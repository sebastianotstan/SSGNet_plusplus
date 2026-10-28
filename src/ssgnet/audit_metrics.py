"""Recompute consistent foreground metrics from saved masks or probability arrays."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from common import list_images, mask_key, binary_metrics, summarize, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--targets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--probabilities', action='store_true', help='Read *.npy or *.prob.npy instead of PNG masks')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--resize_target', action='store_true', help='Explicit nearest-neighbour resize to prediction grid')
    a = p.parse_args()
    if a.out.exists() or not 0 < a.threshold < 1:
        p.error('Use a new output filename and threshold in (0,1)')
    targets = {}
    for path in list_images(a.targets):
        key = mask_key(path)
        if key in targets:
            raise ValueError('Ambiguous target names')
        targets[key] = path
    paths = sorted(a.predictions.glob('*.npy')) if a.probabilities else list_images(a.predictions)
    predictions = {}
    for path in paths:
        key = path.stem.removesuffix('.prob')
        if key in predictions:
            raise ValueError('Duplicate prediction names')
        predictions[key] = path
    if set(predictions) != set(targets):
        raise ValueError(f'Prediction/target IDs differ: missing={len(set(targets)-set(predictions))}, '
                         f'extra={len(set(predictions)-set(targets))}')
    records = []
    for key in sorted(targets):
        if a.probabilities:
            prob = np.load(predictions[key], allow_pickle=False)
            if prob.ndim != 2 or not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
                raise ValueError(f'Expected finite HxW probabilities in [0,1]: {key}')
            pred = prob >= a.threshold
        else:
            with Image.open(predictions[key]) as image:
                pred = np.array(image.convert('L')) > 127
        with Image.open(targets[key]) as image:
            image = image.convert('L')
            if image.size != (pred.shape[1],pred.shape[0]):
                if not a.resize_target:
                    raise ValueError('Grid mismatch; explicitly specify --resize_target if justified')
                image = image.resize((pred.shape[1],pred.shape[0]),Image.Resampling.NEAREST)
            target = np.array(image) > 127
        records.append(dict(id=key, **binary_metrics(pred,target)))
    write_json(a.out, dict(summary=summarize(records), per_image=records,
                          threshold=a.threshold if a.probabilities else 'saved mask >127',
                          target_resize=a.resize_target))
    print(summarize(records))


if __name__ == '__main__':
    main()
