"""CPU logic/integration tests. Synthetic predictors never validate AlphaGenome."""
import collections
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import scan_native_alphagenome as scan
import native_rna_backend as backend
from native_rna_backend import Region, build_exon_mask, load_gene_exons


def write_fasta(folder, sequence, newline=b"\n", wrap=61):
    path = folder / "reference.fa"
    header = b">chr1" + newline
    with open(path, "wb") as f:
        f.write(header)
        for i in range(0, len(sequence), wrap):
            f.write(sequence[i:i+wrap].encode() + newline)
    Path(str(path)+".fai").write_text(f"chr1\t{len(sequence)}\t{len(header)}\t{wrap}\t{wrap+len(newline)}\n")
    return path


class FakeNative:
    calls = 0
    fail_on = None
    seen_batches = []

    def __init__(self, checkpoint):
        self.provenance = {"checkpoint": checkpoint, "weights_and_state_sha256": "synthetic-test-only"}
        self.track_metadata = pd.DataFrame({"name": ["CL:0000127 total RNA-seq"]*2, "strand": ["+", "-"]})

    def encode(self, seq):
        return np.stack([np.frombuffer(seq.encode(), dtype="S1") == b for b in (b"A", b"C", b"G", b"T")], axis=1).astype(np.float32)

    def scores(self, batch, exon_mask, strand):
        FakeNative.calls += 1
        if FakeNative.fail_on == FakeNative.calls:
            raise RuntimeError("simulated interruption")
        FakeNative.seen_batches.append(batch.copy())
        # Sequence position sensitivity permits meaningful scramble contrasts.
        positional = np.linspace(0.01, 0.03, batch.shape[1])
        channel = np.array([1.0, 2.0, 5.0, 9.0])
        out = np.einsum("blc,l,c->b", batch, positional, channel)/batch.shape[1]
        return out + float(exon_mask.sum())/1000 + (0.1 if strand == "-" else 0)

    def baseline(self, sequence, onehot, mask, strand, validate_public):
        value = self.scores(onehot[None], mask, strand)[0]
        return float(value), np.zeros((len(sequence), 2), dtype=np.float32)


class LogicTests(unittest.TestCase):
    def test_native_track_selection_uses_tensor_positions_and_exact_strands(self):
        metadata = pd.DataFrame({
            'name': ['other', 'CL:0000127 total RNA-seq', 'CL:0000679 total RNA-seq', 'CL:0000127 total RNA-seq'],
            'strand': ['.', '-', '+', '+'],
        }, index=['a', 'b', 'c', 'd'])
        with mock.patch.object(backend, 'RNA_OUTPUT', 'RNA_SEQ', create=True):
            mask, selected, indices = backend.select_target_rna_tracks({'RNA_SEQ': metadata})
            self.assertEqual(mask.tolist(), [False, True, False, True])
            self.assertEqual(selected.original_track_index.tolist(), [1, 3])
            self.assertEqual(indices[('Astrocyte', '-')], 0)
            self.assertEqual(indices[('Astrocyte', '+')], 1)
            with self.assertRaisesRegex(RuntimeError, 'expected exactly one'):
                backend.select_target_rna_tracks({'RNA_SEQ': pd.concat([metadata, metadata.iloc[[1]]])})

    def test_fasta_index_newlines_boundaries_and_alias(self):
        with tempfile.TemporaryDirectory() as d:
            sequence = "ACGT"*500 + "AT"
            for newline in (b"\n", b"\r\n"):
                fasta = scan.IndexedFasta(write_fasta(Path(d), sequence, newline))
                self.assertEqual(fasta.resolve("1"), "chr1")
                for a,b in [(0,1), (60,62), (61,400), (0,len(sequence)), (len(sequence)-2,len(sequence))]:
                    self.assertEqual(fasta.fetch("chr1",a,b), sequence[a:b])
                self.assertEqual(fasta.padded("chr1",-10,12), "N"*10+sequence[:12])

    def test_gtf_exon_union_coordinate_endpoints(self):
        with tempfile.TemporaryDirectory() as d:
            gtf=Path(d)/"genes.gtf"
            gtf.write_text('\n'.join([f'chr1\tx\texon\t{a}\t{b}\t.\t-\t.\tgene_id "ENSG1.2";' for a,b in [(10,12),(12,15),(20,21)]]))
            ann=load_gene_exons(str(gtf),"ENSG1")
            mask, bp, frac=build_exon_mask(ann,Region("1",11,20),10)
            self.assertEqual(ann.exon_intervals,[(10,15),(20,21)])
            self.assertEqual(bp,6)
            self.assertEqual(frac,0.75)
            self.assertEqual(np.flatnonzero(mask).tolist(),[0,1,2,3,4,9])

    def test_ambiguous_tss_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"tss.bed"
            p.write_text('chr1\t1\t2\tENSG1\t0\t+\nchr1\t3\t4\tENSG1.2\t0\t+\nchr1\t8\t9\tENSG2\t0\t-\n')
            rows,skipped=scan.bed_targets(p,["ENSG1","ENSG2","ENSG3"])
            self.assertEqual([r['gene_id'] for r in rows],["ENSG2"])
            self.assertEqual([r['reason'] for r in skipped],["ambiguous_tss","missing_tss"])

    def test_tiling_full_edge_window(self):
        windows=scan.windows_for(0,1048576,5000,1000)
        self.assertEqual(len(windows),1045)
        self.assertEqual(windows[-1].tolist(),[1043576,1048576])
        self.assertTrue(np.all(windows[:,1]-windows[:,0]==5000))

    def test_shuffle_counts_and_seed_reproducibility(self):
        original=np.random.default_rng(5).integers(0,4,size=5000)
        counts=collections.Counter(zip(original[:-1],original[1:]))
        changes=[]
        for seed in range(20):
            shuffled=scan.dinucleotide_shuffle(original,np.random.default_rng(seed))
            self.assertEqual(counts,collections.Counter(zip(shuffled[:-1],shuffled[1:])))
            np.testing.assert_array_equal(np.bincount(original,minlength=4),np.bincount(shuffled,minlength=4))
            changes.append(not np.array_equal(original,shuffled))
        self.assertTrue(all(changes))
        self.assertEqual(scan.window_seed(42,"g","chr1",0,1000,2),scan.window_seed(42,"g","chr1",0,1000,2))

    def test_delta_units_and_cancellation(self):
        delta=scan.effects(np.array([[8.,12.]]),10.)
        self.assertEqual(delta['raw'].mean(),0)
        self.assertEqual(np.abs(delta['raw']).mean(),2)
        np.testing.assert_allclose(delta['fractional'],[[-0.2,0.2]])
        np.testing.assert_allclose(delta['log1p'],np.log1p([[8.,12.]])-np.log1p(10))
        self.assertTrue(np.isnan(scan.effects(np.array([[1.]]),0.)['fractional']).all())

    def test_overlap_masking_threshold_and_union_length(self):
        windows=np.array([[0,10],[5,15],[20,30]])
        seg=scan.disjoint_segments(windows,np.ones(3,dtype=bool),np.array([2.,4.,6.]),np.array([2.,-4.,6.]))
        self.assertEqual(seg[:3],[(0,5,2.,2.),(5,10,3.,-1.),(10,15,4.,-4.)])
        meta={'exon_union_1based_inclusive':[(1,2)]}
        target={'gene_id':'G','chrom':'chr1','tss0':2,'strand':'+'}
        calls=scan.call_regions(seg,1.,11,target,meta,5,'raw')
        self.assertEqual(len(calls),1)
        self.assertEqual(calls.iloc[0]['length_bp'],15)
        self.assertEqual(calls.iloc[0]['direction'],'mixed')
        self.assertAlmostEqual(calls.iloc[0]['mean_absolute_effect'],3.)
        self.assertEqual(scan.disjoint_segments(windows,np.array([True,False,True]),np.ones(3),np.ones(3)),[(0,5,1.,1.),(20,30,1.,1.)])
        self.assertEqual(len(scan.call_regions(seg,6.,1,target,meta,5,'raw')),0)

    def test_smoothing_does_not_bridge_missing_window_or_gap(self):
        windows=np.array([[0,10],[10,20],[20,30],[40,50],[50,60]])
        smooth=scan.smooth_valid(np.array([1.,99.,9.,3.,5.]),np.array([True,False,True,True,True]),windows,3)
        np.testing.assert_allclose(smooth,[1.,np.nan,9.,4.,4.],equal_nan=True)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.folder=Path(self.tmp.name)
        sequence='ACGT'*10000
        # First window is not perturbable, second includes N, remaining windows are informative.
        sequence=sequence[:15000]+'A'*1000+'N'+sequence[16001:]
        self.fasta=write_fasta(self.folder,sequence)
        self.gtf=self.folder/'genes.gtf'
        self.gtf.write_text('chr1\tx\texon\t20001\t20100\t.\t-\t.\tgene_id "ENSG1.2";\nchr1\tx\texon\t21001\t21100\t.\t-\t.\tgene_id "ENSG1.2";\n')
        self.targets=self.folder/'targets.tsv'
        self.targets.write_text('gene_id\tchrom\ttss0\tstrand\tbundle_path\nENSG1\t1\t20000\t-\t/path/does/not/exist.pkl\n')
        self.out=self.folder/'out'
        self.argv=['scan','--targets',str(self.targets),'--fasta',str(self.fasta),'--gtf',str(self.gtf),
                   '--assembly','GRCh38','--out-dir',str(self.out),'--sequence-length','16384',
                   '--scan-half-window-bp','5000','--max-windows','4','--window-bp','1000',
                   '--stride-bp','1000','--repeats','3','--batch-size','2','--threshold','0.00001','--effect-units','raw']
        FakeNative.calls=0
        FakeNative.fail_on=None
        FakeNative.seen_batches=[]

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self,argv=None):
        with contextlib.redirect_stdout(io.StringIO()):
            scan.main(self.argv if argv is None else argv,FakeNative)

    def test_full_run_outputs_noop_unknown_and_recall(self):
        self.invoke(self.argv+['--preflight'])
        self.assertFalse(self.out.exists())
        self.assertEqual(FakeNative.calls,0)
        self.invoke()
        folder=self.out/'ENSG1'
        with np.load(folder/'scan_state.npz') as state:
            self.assertEqual(state['effects'].shape,(4,3,3,1))
            self.assertEqual(state['status'].tolist(),['unchanged_all_scrambles','ambiguous_bases','tested','tested'])
            np.testing.assert_array_equal(state['predictions'][0],np.repeat(state['baseline'],3))
        results=folder/'results'/'Astrocyte'
        for filename in ['windows.tsv.gz','absolute.bedGraph','signed.bedGraph','regions.tsv','regions.bed','perturbation_track.png','log.json']:
            self.assertTrue((results/filename).exists(),filename)
        meta=json.loads((folder/'scan_metadata.json').read_text())
        self.assertEqual(meta['target']['strand'],'-')
        self.assertEqual(meta['target']['chrom'],'chr1')
        self.assertNotIn('bundle_path',meta['target'])
        # All edits confined to the window and preserve one-hot base counts.
        reference=FakeNative('x').encode(scan.IndexedFasta(self.fasta).padded('chr1',11808,28192))
        for batch in FakeNative.seen_batches[1:]:
            for example in batch:
                np.testing.assert_array_equal(example.sum(axis=0),reference.sum(axis=0))
                outside=np.ones(len(reference),bool)
                outside[17000-11808:19000-11808]=False
                np.testing.assert_array_equal(example[outside],reference[outside])
        calls=FakeNative.calls
        self.invoke(['recalculate','--gene-dir',str(folder),'--threshold','999','--effect-units','log1p'])
        self.assertEqual(FakeNative.calls,calls)
        self.assertEqual(len(pd.read_csv(results/'regions.tsv',sep='\t')),0)
        self.assertTrue(json.loads((folder/'completion.json').read_text())['complete'])

    def test_interruption_resume_reproducibility_and_contract(self):
        FakeNative.fail_on=3
        with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
            self.invoke()
        FakeNative.fail_on=None
        self.invoke(self.argv+['--resume'])
        with np.load(self.out/'ENSG1'/'scan_state.npz') as s:
            resumed=s['predictions'].copy()
        fresh=self.folder/'fresh'
        other=list(self.argv)
        other[other.index('--out-dir')+1]=str(fresh)
        self.invoke(other)
        with np.load(fresh/'ENSG1'/'scan_state.npz') as s:
            np.testing.assert_array_equal(resumed,s['predictions'])
        changed=list(self.argv)+['--resume','--seed','10']
        with self.assertRaisesRegex(ValueError,'Resume contract mismatch'):
            self.invoke(changed)
        with self.assertRaisesRegex(ValueError,'Output exists'):
            self.invoke()

    def test_dinucleotide_scan(self):
        self.invoke(self.argv+['--shuffle','dinucleotide'])
        with np.load(self.out/'ENSG1'/'scan_state.npz') as state:
            self.assertTrue(np.all(state['status'] != 'pending'))
            # Alternating ACGT repeat cannot be altered while preserving all dinucleotides.
            self.assertTrue(np.all(state['status'][2:] == 'unchanged_all_scrambles'))


if __name__ == '__main__':
    unittest.main()
