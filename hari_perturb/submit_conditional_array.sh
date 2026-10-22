#!/usr/bin/env bash
# Usage (from anywhere):
#   ./submit_conditional_array.sh --genes candidate_genes.tsv [--partition gpu_h200] [--max-concurrent 2] \
#       -- --trust-pickle [--n-shuffles 5] [--n-null 2] [--scope hits|all] [--dry-run] ...
#
# candidate_genes.tsv: a gene_id column (header), or one gene_id per line; blank and # lines ignored.
# gene_ids must match the scan directory names of the original HARI scan (Ensembl IDs), not symbols.
# Everything after `--` goes to `conditional_scan.py run`; --trust-pickle is required there
# (the bundles are pickles: only use your own).
set -Eeuo pipefail
cond_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${CONDITIONAL_CONFIG:-$cond_root/conditional_config.sh}"
genes=""; partition="gpu_h200"; concurrency=2
while (($#)); do
    case "$1" in
        --genes) genes="$2"; shift 2 ;;
        --partition) partition="$2"; shift 2 ;;
        --max-concurrent) concurrency="$2"; shift 2 ;;
        --) shift; break ;;
        *) printf 'Unknown submission option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n "$genes" && -f "$genes" ]] || { printf 'Provide --genes FILE\n' >&2; exit 2; }
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid --max-concurrent\n' >&2; exit 2; }
: "${REF_SCAN_ROOT:?Set REF_SCAN_ROOT (original HARI scan output root) in conditional_config.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Freeze the gene list so later edits cannot reassign array tasks to different genes.
manifest_dir="${COND_MANIFEST_DIR:-$PWD/conditional_manifests}"
mkdir -p -- "$manifest_dir" logs
snapshot="$(mktemp "$manifest_dir/genes.XXXXXXXX.tsv")"
cp -- "$genes" "$snapshot"
snapshot="$(cd -- "$(dirname -- "$snapshot")" && pwd)/$(basename -- "$snapshot")"
n="$("$PYTHON_BIN" - "$cond_root" "$snapshot" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from conditional_scan import read_gene_ids
print(len(read_gene_ids(sys.argv[2])))
PY
)"
printf 'Submitting %s genes (max %s concurrent) on %s; gene list frozen at %s\n' "$n" "$concurrency" "$partition" "$snapshot"
export COND_ROOT="$cond_root" COND_GENES_TSV="$snapshot"
sbatch --partition="$partition" --array="0-$((n-1))%$concurrency" --export=ALL \
    "$cond_root/run_conditional_array.sbatch" "$@"
