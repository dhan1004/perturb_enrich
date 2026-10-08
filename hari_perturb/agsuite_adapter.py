"""Inference-only adapter for the supplied exportable single-gene CV bundle.

Imports the existing agsuite implementations. Never trains, replaces weights,
guesses a parameter layout, or treats a multi-gene checkpoint as interchangeable.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import importlib
import inspect
import json
import pickle

import numpy as np

from core import sha256_file

BUNDLE_TYPE = "alphagenome_single_gene_celltype_cv_ensemble"
SCHEMA_KEYS = ("checkpoint", "organism", "embedding_key", "bp_per_bin",
               "pool_half_windows_bp", "pool_stat", "downsample_1bp_to",
               "pooled_feature_dim", "required_sequence_length")


def plain(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def load_bundle(path, gene_id, schema_path=None):
    # Trusted local pickle only: caller requires explicit --trust-pickle.
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if bundle.get("format_version") != 1 or bundle.get("bundle_type") != BUNDLE_TYPE:
        raise ValueError("Unsupported checkpoint. Expected the supplied single-gene cell-type CV inference bundle; multi-gene weights need their own verified adapter/export.")
    if bundle["gene_id"] != gene_id:
        raise ValueError(f"Wrong gene bundle: {bundle['gene_id']} != {gene_id}")
    cov = np.asarray(bundle["donor_covariates_raw"])
    if cov.ndim != 2 or cov.shape != (len(bundle["sample_ids"]), len(bundle["covariate_names"])):
        raise ValueError("Saved donor-covariate dimensions disagree")
    if len(set(map(str, bundle["sample_ids"]))) != len(bundle["sample_ids"]):
        raise ValueError("Duplicate sample IDs in inference bundle")
    if not bundle["cell_types"]:
        raise ValueError("No cell-type models in bundle")
    generation = bundle.get("embedding_generation", {})
    meta = bundle.get("embedding_shard_metadata", {})
    schema = generation.get("training_embedding_schema") or meta.get("embedding_schema")
    if schema_path:
        with open(schema_path) as f:
            supplied = json.load(f)
        supplied = supplied.get("embedding_schema", supplied)
        if schema and plain(schema) != plain(supplied):
            raise ValueError("External schema conflicts with bundle's recorded training schema")
        schema = supplied
    if not schema:
        raise ValueError("No authoritative training embedding_schema. Export it from the training shard metadata.json and provide --schema-json. CLI 'auto'/fallback defaults are insufficient.")
    schema = plain(schema)
    missing = set(SCHEMA_KEYS) - set(schema)
    if missing or schema["embedding_key"] == "auto":
        raise ValueError(f"Incomplete/unresolved training embedding schema: missing={sorted(missing)}")
    if schema.get("n_haplotypes") != 2:
        raise ValueError("Expected a diploid H=2 training embedding schema")
    if int(schema["pooled_feature_dim"]) != int(bundle["raw_haplotype_embedding_dim"]):
        raise ValueError("Pooled feature width disagrees with inference bundle")
    if int(schema["required_sequence_length"]) < 1:
        raise ValueError("Invalid recorded model input length")
    if generation.get("channel_order", "ACGT") != "ACGT":
        raise ValueError("This reference FASTA workflow requires the training ACGT channel order")
    dtype = meta.get("output_dtype")
    return bundle, schema, dtype


def validate_schema(expected, actual):
    actual = plain(actual)
    for key in SCHEMA_KEYS:
        if plain(expected[key]) != actual.get(key):
            raise ValueError(f"Embedding contract mismatch for {key}: trained={expected[key]!r}, actual={actual.get(key)!r}")
    if "embedding_probe_shape" in expected:
        if list(expected["embedding_probe_shape"])[1:] != list(actual.get("embedding_probe_shape", []))[1:]:
            raise ValueError("Embedding positional/channel dimensions differ from training")
    # Probe batch size and H differ deliberately: compute one reference haplotype,
    # then duplicate its EMBEDDING for both homologues, avoiding duplicate GPU work.
    if int(actual.get("n_haplotypes", 0)) != 1:
        raise ValueError("The scan embedder must return single-haplotype features")


def validate_embedder_class_length(schema, class_path):
    module_name, name = class_path.split(":", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, name):
        raise ValueError(f"Embedder class not found: {class_path}")
    if class_path == "agsuite.alphagenome:AlphaGenomePooledEmbedder":
        required = getattr(module, "AG_REQUIRED_L", None)
        if required != schema["required_sequence_length"]:
            raise ValueError(f"Installed legacy embedder uses {required} bp, but bundle was trained with {schema['required_sequence_length']} bp. Supply its matching --embedder-class; never widen/crop silently.")


class SuiteEmbedder:
    def __init__(self, schema, class_path="agsuite.alphagenome:AlphaGenomePooledEmbedder", allow_cpu=False):
        import jax
        if not allow_cpu and not any(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError("AlphaGenome scan needs a JAX-visible GPU; use --allow-cpu-trunk only deliberately")
        self.expected = schema
        validate_embedder_class_length(schema, class_path)
        module_name, name = class_path.split(":", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, name)
        self.engine = cls(checkpoint=schema["checkpoint"], organism=schema["organism"],
                          embedding_key=schema["embedding_key"], half_windows_bp=schema["pool_half_windows_bp"],
                          pool_stat=schema["pool_stat"], downsample_1bp_to=schema["downsample_1bp_to"], trunk_batch=1)
        self.checked = False

    def embed(self, seq):
        from agsuite.sequences import SequenceStore
        seq = np.asarray(seq, dtype=np.float32)
        if seq.shape != (int(self.expected["required_sequence_length"]), 4):
            raise ValueError("Sequence length differs from the training contract")
        store = SequenceStore(array=seq[None, None], sample_ids=np.array(["REFERENCE"]),
                              source_key="reference", original_shape=seq[None, None].shape)
        output = np.asarray(self.engine.embed(store), dtype=np.float32)
        if output.shape != (1, 1, int(self.expected["pooled_feature_dim"])) or not np.isfinite(output).all():
            raise ValueError(f"Invalid pooled embedding: {output.shape}")
        if not self.checked:
            validate_schema(self.expected, self.engine.schema)
            self.checked = True
        return output[0, 0]


def source_provenance(embedder_class):
    names = ["agsuite.models", "agsuite.data", "agsuite.embedding_store", "agsuite.sequences",
             embedder_class.split(":")[0]]
    result = {}
    for name in dict.fromkeys(names):
        module = importlib.import_module(name)
        path = inspect.getsourcefile(module)
        if not path:
            raise ValueError(f"Cannot fingerprint source module {name}")
        result[name] = {"path": path, "sha256": sha256_file(path)}
    return result


def validate_transform(transform, fields, width):
    for field in fields:
        x = np.asarray(transform[field])
        if x.shape != (width,) or not np.isfinite(x).all():
            raise ValueError(f"Invalid saved {field}; expected finite vector of length {width}")
        if field.endswith("std") and np.any(x <= 0):
            raise ValueError(f"Nonpositive scale in {field}")


def make_context(bundle, cell, fold, mode, external=None, max_contexts=0, seed=42):
    """Marginalize whole profiles, never each covariate independently."""
    from agsuite.data import FeatureTransform
    transform = FeatureTransform(**fold["covariate_transform"])
    if mode == "mean":
        raw = np.asarray(transform.cov_mean)[None]
        ids = ["TRAINING_MEAN_PROFILE"]
    elif mode == "fixed":
        if external is None or len(external) != 1:
            raise ValueError("fixed mode requires exactly one row in --contexts-tsv")
        raw, ids = external.to_numpy(dtype=np.float32), ["FIXED_PROFILE"]
    elif external is not None:
        raw, ids = external.to_numpy(dtype=np.float32), [f"EXTERNAL_{i}" for i in range(len(external))]
    else:
        idx = np.asarray(fold["train_store_donor_indices"], dtype=np.int64)
        n = len(bundle["sample_ids"])
        if len(idx) == 0 or len(np.unique(idx)) != len(idx) or np.any(idx < 0) or np.any(idx >= n):
            raise ValueError(f"Invalid training donor indices: {cell}, fold {fold['fold']}")
        raw = np.asarray(bundle["donor_covariates_raw"], dtype=np.float32)[idx]
        ids = np.asarray(bundle["sample_ids"])[idx].astype(str).tolist()
    if len(raw) == 0:
        raise ValueError("Empty context population")
    if max_contexts and len(raw) > max_contexts:
        # Same external rows for every cell/fold when a common panel is supplied.
        rng = np.random.default_rng(seed if external is not None else seed + int(fold["fold"]))
        chosen = np.sort(rng.choice(len(raw), max_contexts, replace=False))
        raw, ids = raw[chosen], [ids[i] for i in chosen]
    scaled = transform.transform_covariates(raw)
    if not np.isfinite(scaled).all():
        raise ValueError("Nonfinite transformed covariates")
    return scaled, ids


class FoldPredictor:
    def __init__(self, bundle, cell, fold, context_mode, external, max_contexts, seed, batch_size, device):
        import jax
        import jax.numpy as jnp
        from agsuite.models import CovariateCellBaseline, ResidualFiLMTower
        from agsuite.data import FeatureTransform, GroupedTargetTransform
        from flax.core import unfreeze
        from flax.traverse_util import flatten_dict

        self.jax, self.jnp, self.device = jax, jnp, device
        self.fold, self.cell, self.batch_size = fold, cell, batch_size
        self.accepted = bool(fold["residual_accepted"])
        width = len(bundle["covariate_names"])
        validate_transform(fold["covariate_transform"], ("cov_median", "cov_mean", "cov_std"), width)
        self.cov, self.context_ids = make_context(bundle, cell, fold, context_mode, external, max_contexts, seed)
        self.target = GroupedTargetTransform(**fold["target_transform"])
        if int(self.target.n_groups) != 1 or np.asarray(self.target.means).shape != (1,) or np.asarray(self.target.stds).shape != (1,):
            raise ValueError("Expected one-group target transform")
        self.mean, self.std = float(self.target.means[0]), float(self.target.stds[0])
        if not np.isfinite([self.mean, self.std]).all() or self.std <= 0:
            raise ValueError("Invalid target transform")
        self.mode = fold["diploid_combine"]
        self.feature = None
        fdim = int(bundle["raw_haplotype_embedding_dim"])
        if self.accepted:
            if self.mode not in {"mean", "mean_absdiff", "concat"}:
                raise ValueError("Unsupported diploid combination")
            dim = fdim if self.mode == "mean" else 2 * fdim
            validate_transform(fold["embedding_transform"], ("emb_mean", "emb_std"), dim)
            self.feature = FeatureTransform(**fold["embedding_transform"])

        def build(residual):
            cfg = fold["residual_config" if residual else "baseline_config"]
            kwargs = dict(hidden_dims=tuple(cfg["hidden_dims"]), refine_dim=int(cfg["refine_dim"]),
                          dropout_rate=float(cfg["dropout"]), n_cell_types=1,
                          cell_embedding_dim=int(cfg["cell_embedding_dim"]), head_mode=cfg["head_mode"])
            if residual:
                kwargs.update(film_scale=cfg["film_scale"], residual_output_init=cfg["residual_output_init"])
            model = (ResidualFiLMTower if residual else CovariateCellBaseline)(**kwargs)
            params = fold["residual_params" if residual else "baseline_params"]
            if params is None:
                raise ValueError("Missing accepted model weights")
            inputs = [jax.ShapeDtypeStruct((1, width), jnp.float32), jax.ShapeDtypeStruct((1,), jnp.int32)]
            if residual:
                inputs.insert(0, jax.ShapeDtypeStruct((1, dim), jnp.float32))
            with jax.default_device(device):
                expected = jax.eval_shape(lambda *x: model.init(jax.random.PRNGKey(0), *x, training=False), *inputs)["params"]
            want, got = flatten_dict(unfreeze(expected)), flatten_dict(unfreeze(params))
            if set(want) != set(got):
                raise ValueError(f"Parameter keys incompatible: {cell}, fold {fold['fold']}; missing={set(want)-set(got)}, extra={set(got)-set(want)}")
            for key in want:
                if want[key].shape != np.asarray(got[key]).shape or not np.isfinite(got[key]).all():
                    raise ValueError(f"Invalid parameter shape/value at {key}")
            params = jax.tree_util.tree_map(lambda a: jax.device_put(np.asarray(a), device), params)
            return jax.jit(lambda *x: model.apply({"params": params}, *x, training=False))

        self.baseline_fn = build(False)
        self.residual_fn = build(True) if self.accepted else None
        self.baseline_scaled = self._apply(self.baseline_fn)
        self.reference_residual = None

    def _apply(self, fn, combined=None):
        result = []
        for start in range(0, len(self.cov), self.batch_size):
            cov = self.cov[start:start+self.batch_size]
            n = len(cov)
            if n < self.batch_size:
                cov = np.concatenate([cov, np.repeat(cov[-1:], self.batch_size-n, axis=0)])
            inputs = [cov, np.zeros(self.batch_size, dtype=np.int32)]
            if combined is not None:
                inputs.insert(0, np.broadcast_to(combined, (self.batch_size, len(combined))))
            inputs = [self.jax.device_put(x, self.device) for x in inputs]
            result.append(np.asarray(fn(*inputs), dtype=np.float64)[:n])
        result = np.concatenate(result)
        if result.shape != (len(self.cov),) or not np.isfinite(result).all():
            raise ValueError("Invalid model predictions")
        return result

    def residual(self, haploid_embedding):
        from agsuite.embedding_store import combine_haplotypes
        if not self.accepted:
            return np.zeros(len(self.cov), dtype=np.float64)
        diploid = np.stack([haploid_embedding, haploid_embedding])[None]
        combined = combine_haplotypes(diploid, self.mode)
        scaled = self.feature.transform_embeddings(combined)[0]
        if not np.isfinite(scaled).all():
            raise ValueError("Invalid standardized embedding")
        return self._apply(self.residual_fn, scaled)

    def set_reference(self, embedding):
        self.reference_residual = self.residual(embedding)
        return float(np.mean((self.baseline_scaled + self.reference_residual) * self.std + self.mean))

    def delta(self, embedding):
        # Pair FIRST, at identical profiles; baseline and target offsets cancel.
        delta_z = self.residual(embedding) - self.reference_residual
        return float(delta_z.mean() * self.std), float(delta_z.mean()), float(np.mean(np.abs(delta_z)) * self.std)


class Ensemble:
    def __init__(self, bundle, cells, mode="marginal", external=None, max_contexts=0, seed=42,
                 batch_size=128, device_name="cpu"):
        import jax
        self.cells, self.folds, self.baseline_rows = cells, {}, []
        device = jax.devices(device_name)[0]
        for cell in cells:
            saved = bundle["cell_types"][cell]["fold_models"]
            if not saved or len({int(f["fold"]) for f in saved}) != len(saved):
                raise ValueError(f"Empty/duplicate ensemble folds for {cell}")
            self.folds[cell] = [FoldPredictor(bundle, cell, f, mode, external, max_contexts, seed, batch_size, device)
                                for f in saved]

    def set_reference(self, embedding):
        self.baseline_rows = []
        self.reference = []
        for cell in self.cells:
            preds = []
            for f in self.folds[cell]:
                pred = f.set_reference(embedding)
                preds.append(pred)
                self.baseline_rows.append(dict(cell_type=cell, fold=f.fold["fold"], residual_accepted=f.accepted,
                    baseline_prediction_adjusted=pred, n_contexts=len(f.cov), target_std=f.std,
                    target_mean=f.mean, context_ids=f.context_ids))
            self.reference.append(float(np.mean(preds)))
        self.reference = np.asarray(self.reference)

    def delta(self, embedding):
        raw, z, context_abs, fold_sd = [], [], [], []
        for cell in self.cells:
            values = np.asarray([f.delta(embedding) for f in self.folds[cell]])
            raw.append(values[:, 0].mean())
            # Standardize AFTER ensembling, by the mean saved training-target SD.
            # Positive rescaling preserves the sign of the actual predicted delta.
            z.append(values[:, 0].mean() / np.mean([f.std for f in self.folds[cell]]))
            context_abs.append(values[:, 2].mean())
            fold_sd.append(values[:, 0].std(ddof=0))
        return np.asarray([raw, z, context_abs, fold_sd])
