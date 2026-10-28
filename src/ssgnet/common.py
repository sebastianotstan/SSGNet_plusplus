"""Portable manifests and explicit binary-foreground metrics (no torch needed)."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
METRICS = ('dice', 'foreground_iou', 'sensitivity', 'specificity', 'accuracy')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def image_hash(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        h = hashlib.sha256(str(rgb.size).encode())
        h.update(rgb.tobytes())
        return h.hexdigest()


def resolve(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Path escapes data root: {relative}')
    return path


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f'Missing image directory: {folder}')
    paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError(f'Duplicate image stems in {folder}')
    if not paths:
        raise ValueError(f'No images in {folder}')
    return paths


def mask_key(path):
    key = Path(path).stem
    for suffix in ('_segmentation', '_mask', '_gt'):
        if key.endswith(suffix):
            key = key[:-len(suffix)]
    return key


def collect_pairs(root, dataset, split):
    root = Path(root)
    folder = root / dataset / split
    masks = {}
    for path in list_images(folder / 'masks'):
        key = mask_key(path)
        if key in masks:
            raise ValueError(f'Ambiguous mask key: {path}')
        masks[key] = path
    rows = []
    for path in list_images(folder / 'images'):
        if path.stem not in masks:
            raise ValueError(f'Missing mask for {path}')
        mask = masks[path.stem]
        with Image.open(path) as image, Image.open(mask) as target:
            if image.size != target.size:
                raise ValueError(f'Image/mask size mismatch: {path}')
            values = np.unique(np.asarray(target.convert('L')))
            # Preserve the original code's grayscale >127 definition, including antialiased masks.
            if target.mode not in ('L', 'RGB', 'RGBA', 'P', '1'):
                raise ValueError(f'Unsupported mask format: {mask} ({target.mode})')
        rows.append(dict(id=f'{split}/{path.name}', image=path.relative_to(root).as_posix(),
                         mask=mask.relative_to(root).as_posix(), original_split=split,
                         image_sha256=file_hash(path), mask_sha256=file_hash(mask),
                         pixel_sha256=image_hash(path), mask_threshold=127))
    return rows


def validate_manifest(manifest):
    rows = manifest['rows']
    if manifest.get('schema') != 1:
        raise ValueError('Unsupported split manifest schema')
    for split in ('train', 'val', 'test'):
        if not any(r['split'] == split for r in rows):
            raise ValueError(f'Empty {split} split')
    ids, images, groups, hashes = set(), set(), {}, {}
    for r in rows:
        if r['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Unknown split')
        if r['id'] in ids or r['image'] in images:
            raise ValueError('Duplicate ID or path in manifest')
        ids.add(r['id']); images.add(r['image'])
        for value, table, label in ((r['group'], groups, 'group'), (r['pixel_sha256'], hashes, 'image content')):
            if value in table and table[value] != r['split']:
                raise ValueError(f'Cross-split {label} overlap: {r["id"]}')
            table[value] = r['split']


def verify_files(rows, root):
    for r in rows:
        for kind in ('image', 'mask'):
            path = resolve(root, r[kind])
            if file_hash(path) != r[kind + '_sha256']:
                raise ValueError(f'File changed since split was frozen: {path}')


def binary_metrics(pred, target):
    pred, target = np.asarray(pred, dtype=bool), np.asarray(target, dtype=bool)
    if pred.shape != target.shape or pred.size == 0:
        raise ValueError('Prediction and target must have identical nonempty shapes')
    return metrics_from_counts(int(np.count_nonzero(pred & target)),
                               int(np.count_nonzero(pred & ~target)),
                               int(np.count_nonzero(~pred & target)),
                               int(np.count_nonzero(~pred & ~target)))


def metrics_from_counts(tp, fp, fn, tn):
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    if sensitivity is not None and dice > 2 * sensitivity / (1 + sensitivity) + 1e-12:
        raise AssertionError('Dice/sensitivity bound violated')
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, dice=dice, foreground_iou=iou,
                sensitivity=sensitivity, specificity=specificity,
                accuracy=(tp + tn) / (tp + fp + fn + tn))


def summarize(rows):
    if not rows:
        raise ValueError('Cannot summarize an empty evaluation')
    def mean(key, subset):
        values = [r[key] for r in subset if r[key] is not None]
        return float(np.mean(values)) if values else None
    nonempty = [r for r in rows if r['tp'] + r['fn'] > 0]
    return dict(n_images=len(rows), n_nonempty_targets=len(nonempty),
                macro_image={k: mean(k, rows) for k in METRICS},
                macro_nonempty_target={k: mean(k, nonempty) for k in METRICS},
                micro_pixels=metrics_from_counts(*(sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn', 'tn'))),
                definitions={'foreground': 'grayscale ground truth >127; prediction probability >= configured threshold', 'averaging': 'equal weight per image',
                             'both_empty_dice_iou': 1, 'undefined_rates': 'null; excluded from that macro rate',
                             'iou': 'foreground IoU, not foreground/background class mean',
                             'bound_check': 'per image and micro; macro bound uses nonempty targets only'})
