"""Standard TorchMetrics/torch-fidelity Inception Score for an image-only folder."""
import argparse
import importlib.metadata
from pathlib import Path
import random
import numpy as np
from PIL import Image
from common import EXTENSIONS, file_hash, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--splits', type=int, default=10)
    p.add_argument('--seed', type=int, default=42, help='IS sample/split seed; not a training seed')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--max_images', type=int, default=0, help='0 = all; otherwise random subset before scoring')
    p.add_argument('--recursive', action='store_true', help='Only use on an image-only tree, never an experiment output tree')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--check_only', action='store_true', help='Verify inputs without loading Inception or calculating a score')
    a = p.parse_args()
    if a.out.exists():
        p.error('Output already exists; choose a new output filename')
    if not a.images.is_dir() or a.batch < 1 or a.splits < 2 or a.max_images < 0:
        p.error('Check image folder, batch, splits >=2, and max_images >=0')
    paths = sorted(x for x in (a.images.rglob('*') if a.recursive else a.images.iterdir())
                   if x.is_file() and x.suffix.lower() in EXTENSIONS)
    total = len(paths)
    if a.max_images and total > a.max_images:
        paths = sorted(random.Random(a.seed).sample(paths, a.max_images))
    if len(paths) < 2 * a.splits:
        p.error('Need at least two images per split; reduce --splits for a small check')
    inventory = []
    for path in paths:
        with Image.open(path) as im:
            if im.mode not in ('RGB', 'L'):
                raise ValueError(f'{path}: expected 8-bit RGB/grayscale; found {im.mode}')
            im.load()
            inventory.append(dict(file=path.relative_to(a.images).as_posix(), sha256=file_hash(path),
                                  mode=im.mode, size=list(im.size)))
    report = dict(metric='Inception Score', status='inputs_checked_only' if a.check_only else 'computed',
                  n_available=total, n_scored=0 if a.check_only else len(paths), n_selected=len(paths),
                  subset=bool(total != len(paths)), splits=a.splits, split_seed=a.seed,
                  preprocessing='8-bit RGB; grayscale replicated to RGB; extractor performs 299x299 resize',
                  interpretation='Supplementary ImageNet-space score; not clinical realism, privacy, or downstream utility',
                  uncertainty='Split standard deviation is not a training-seed CI',
                  images=inventory)
    if not a.check_only:
        import torch
        from torchmetrics.image.inception import InceptionScore
        if a.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --device cpu')
        torch.set_num_threads(2)
        torch.manual_seed(a.seed)
        metric = InceptionScore(feature='logits_unbiased', splits=a.splits, normalize=False).to(a.device)
        pending, shape = [], None
        with torch.inference_mode():
            for path in paths:
                with Image.open(path) as im:
                    pixels = torch.from_numpy(np.array(im.convert('RGB'), copy=True)).permute(2,0,1)
                if pending and (pixels.shape != shape or len(pending) >= a.batch):
                    metric.update(torch.stack(pending).to(a.device)); pending = []
                pending.append(pixels); shape = pixels.shape
            if pending:
                metric.update(torch.stack(pending).to(a.device))
            # Reset immediately before random split assignment, independent of model initialization.
            torch.manual_seed(a.seed)
            mean, std = metric.compute()
        report.update(mean=float(mean), split_std=float(std),
                      versions={k: importlib.metadata.version(k) for k in ('torch','torchvision','torchmetrics','torch-fidelity','Pillow')})
    write_json(a.out, report)
    print({k:v for k,v in report.items() if k != 'images'})


if __name__ == '__main__':
    main()
