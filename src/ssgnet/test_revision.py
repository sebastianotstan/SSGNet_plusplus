"""Focused regression tests; temporary fixtures never alter REAL/SYNTH."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from common import binary_metrics, summarize, write_json, validate_manifest
from prepare_splits import make_split


def fixtures(root):
    rng = np.random.default_rng(9)
    for split, count in [('train',8),('val',10)]:
        for name in ('images','masks'):
            (root/'REAL'/'toy'/split/name).mkdir(parents=True)
        for i in range(count):
            pixels = rng.integers(0,256,(64,64,3),dtype=np.uint8)
            target = np.zeros((64,64),np.uint8)
            target[10+i:30+i,15:40] = 255
            Image.fromarray(pixels).save(root/'REAL'/'toy'/split/'images'/f'{i}.png')
            Image.fromarray(target).save(root/'REAL'/'toy'/split/'masks'/f'{i}.png')
    folder = root/'SYNTH'/'toy'/'train'/'images';folder.mkdir(parents=True)
    for i in range(8):
        Image.fromarray(rng.integers(0,256,(64,64,3),dtype=np.uint8)).save(folder/f'{i}.png')


class MetricsTests(unittest.TestCase):
    def test_known_counts(self):
        r=binary_metrics([1,1,0,0],[1,0,1,0])
        self.assertEqual(r['dice'],0.5)
        self.assertEqual(r['sensitivity'],0.5)
        self.assertAlmostEqual(r['foreground_iou'],1/3)

    def test_empty_policy(self):
        r=binary_metrics([0,0],[0,0])
        self.assertEqual(r['dice'],1)
        self.assertIsNone(r['sensitivity'])
        self.assertEqual(binary_metrics([1,0],[0,0])['dice'],0)

    def test_bound_random(self):
        rng=np.random.default_rng(5)
        for _ in range(100):
            r=binary_metrics(rng.random(100)>0.7,rng.random(100)>0.7)
            self.assertLessEqual(r['dice'],2*r['sensitivity']/(1+r['sensitivity'])+1e-12)

    def test_equal_image_average(self):
        rows=[binary_metrics([1],[1]),binary_metrics([1],[1]),binary_metrics([0],[1])]
        self.assertAlmostEqual(summarize(rows)['macro_image']['dice'],2/3)

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            binary_metrics([1,0],[1])


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        fixtures(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fixed_split_disjoint(self):
        a=make_split(self.root/'REAL','toy',0.5,2026)
        b=make_split(self.root/'REAL','toy',0.5,2026)
        self.assertEqual(a,b)
        self.assertEqual(a['counts'],{'train':8,'val':5,'test':5})
        validate_manifest(a)

    def test_duplicate_quarantine(self):
        src=self.root/'REAL'/'toy'/'train'/'images'/'0.png'
        dest=self.root/'REAL'/'toy'/'val'/'images'/'0.png'
        dest.write_bytes(src.read_bytes())
        with self.assertRaises(ValueError):
            make_split(self.root/'REAL','toy',0.5,2026)
        manifest=make_split(self.root/'REAL','toy',0.5,2026,exclude_train_duplicates=True)
        self.assertEqual(manifest['excluded_old_val_count'],1)
        validate_manifest(manifest)

    def test_refuse_overwrite(self):
        p=self.root/'result.json'
        write_json(p,{'a':1})
        with self.assertRaises(FileExistsError):
            write_json(p,{'a':2})


class TorchTests(unittest.TestCase):
    def test_weighted_loss_gradient(self):
        import torch
        from engine import loss_per_image, loss_per_batch
        logits=torch.zeros((2,1,4,4),requires_grad=True)
        target=torch.ones_like(logits)
        (loss_per_image(logits,target,0.5)*torch.tensor([1.,0.])).mean().backward()
        self.assertGreater(logits.grad[0].abs().sum().item(),0)
        self.assertEqual(logits.grad[1].abs().sum().item(),0)

    def test_legacy_batch_loss_matches_ultimate_formula(self):
        import torch
        import torch.nn.functional as F
        from engine import loss_per_batch, load_config
        logits=torch.tensor([[[[0.2, -0.4], [1.0, -1.0]]],
                             [[[0.1, 0.3], [-0.2, 0.7]]]], requires_grad=True)
        target=torch.tensor([[[[1., 0.], [1., 0.]]],
                             [[[0., 1.], [0., 1.]]]])
        p=torch.sigmoid(logits).clamp(1e-7, 1-1e-7)
        eps=1e-6
        dice=1-((2*(p*target).sum((2,3))+eps)/((p+target).sum((2,3))+eps)).mean()
        expected=0.5*F.binary_cross_entropy_with_logits(logits,target)+0.5*dice
        self.assertTrue(torch.allclose(loss_per_batch(logits,target,0.5),expected))
        self.assertEqual(load_config({})['loss_weighting'],'legacy_batch')

    def test_augmentation_seed(self):
        from engine import SegData
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);fixtures(root)
            rows=make_split(root/'REAL','toy',0.5,2026)['rows'][:4]
            a=SegData(rows,root/'REAL',64,seed=42,epoch=1,aug=True)
            b=SegData(rows,root/'REAL',64,seed=42,epoch=1,aug=True)
            self.assertTrue(np.array_equal(a[0][0].numpy(),b[0][0].numpy()))
            c=SegData(rows,root/'REAL',64,seed=123,epoch=1,aug=True)
            self.assertFalse(np.array_equal(a[0][0].numpy(),c[0][0].numpy()))

    def test_config_synthetic_limit_is_explicit_and_bounded(self):
        from engine import load_config
        self.assertEqual(load_config({'synthetic_limit':64})['synthetic_limit'],64)
        with self.assertRaises(ValueError):
            load_config({'synthetic_limit':-1})

    def test_multihead_and_tk(self):
        import torch
        from engine import build_model,load_config,predict
        torch.set_num_threads(1)
        c=load_config({'models':['fpn:efficientnet-b0'],'encoder_weights':None,'multihead':True,
                       'heads':[[4,0,1],[4,0.1,2]],'tk':True,'tk_iters':2})
        m=build_model(c['models'][0],c,torch.device('cpu'))
        logits=m(torch.randn(2,3,64,64))
        sum(x.mean() for x in logits).backward()
        _,raw,refined=predict([m],torch.randn(2,3,64,64),c,use_tk=True)
        self.assertEqual(tuple(refined.shape),(2,1,64,64))
        self.assertTrue(torch.isfinite(refined).all())

    def test_jbf_no_silent_fallback(self):
        from engine import load_config,refine
        c=load_config({})
        rng=np.random.default_rng(8)
        result=refine(rng.integers(0,256,(64,64,3),dtype=np.uint8),rng.random((32,32)).astype(np.float32),c,'jbf')
        self.assertEqual(result.shape,(32,32))
        self.assertTrue(np.isfinite(result).all())



class PairedTests(unittest.TestCase):
    def inputs(self):
        a={}; b={}
        for seed in (42,123,2024,3407):
            run=dict(dataset='toy',split_sha256='fixed')
            rows={str(i):dict(id=str(i),group=str(i),dice=0.5+i/100) for i in range(5)}
            a[seed]=(run,rows,0)
            b[seed]=(run,{k:dict(v,dice=v['dice']+0.01) for k,v in rows.items()},1)
        return a,b

    def test_paired_difference(self):
        from compare_seeds import compare
        a,b=self.inputs()
        r=compare(a,b,'dice',200,2026)
        self.assertAlmostEqual(r['mean_difference'],0.01)
        self.assertAlmostEqual(r['paired_seed_t_ci95'][0],0.01)
        self.assertAlmostEqual(r['group_bootstrap_ci95_conditional_on_trained_seeds'][1],0.01)

    def test_reject_unpaired_seed(self):
        from compare_seeds import compare
        a,b=self.inputs();b.pop(42)
        with self.assertRaises(ValueError):
            compare(a,b,'dice',200,2026)

    def test_reject_changed_split(self):
        from compare_seeds import compare
        a,b=self.inputs()
        run,rows,r=b[42]
        b[42]=(dict(run,split_sha256='changed'),rows,r)
        with self.assertRaises(ValueError):
            compare(a,b,'dice',200,2026)

if __name__ == '__main__':
    unittest.main()
