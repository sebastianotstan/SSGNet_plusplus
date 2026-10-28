"""Shared training/inference implementation for the revision commands."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import csv
import hashlib
import importlib.metadata
import random
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from common import binary_metrics, summarize, write_json, resolve, file_hash
import models as legacy

DEFAULTS = dict(
    models=['fpn:efficientnet-b0', 'fpn:efficientnet-b1', 'fpn:efficientnet-b2'],
    multihead=False, heads=[[64, 0.1, 1], [96, 0.0, 2], [128, 0.2, 3]],
    img_size=256, batch=8, epochs0=30, epochs=20, rounds=2, lr=0.001,
    bce_weight=0.5, loss_weighting='legacy_batch', weight_decay=0.0001, grad_clip=1.0, amp=False,
    encoder_weights='imagenet', fuse_mode='logit', eval_threshold=0.5,
    pseudo_threshold=0.55, th_method='otsu', pseudo_ratios=[0.2, 0.4],
    pseudo_weights=[0.099, 0.198], pseudo_max_count=1000, synthetic_limit=0,
    selection='random', select_use_raw=True, min_area=0.002, max_area=0.6, ent_w=0.3,
    refine='none', final_refine='none', tk=False, tk_iters=10, tk_fg=0.9, tk_bg=0.1,
    tk_use_xy=True, tk_xy_weight=0.1, tk_soft=False, tk_temp=0.1, tk_feat_norm=True,
    jbf_d=7, jbf_sigma_color=50.0, jbf_sigma_space=7.0,
    crf_iters=5, crf_sxy_gaussian=3.0, crf_compat_gaussian=3.0,
    crf_sxy_bilateral=50.0, crf_srgb_bilateral=10.0, crf_compat_bilateral=5.0,
    deterministic=True, strict_determinism=False
)


def load_config(value):
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown config keys: {sorted(unknown)}')
    c = dict(DEFAULTS, **value)
    for key in ('img_size', 'batch', 'epochs0', 'epochs', 'tk_iters'):
        if not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if c['batch'] < 2 or c['img_size'] < 32 or c['img_size'] % 32:
        raise ValueError('batch >= 2 and img_size a multiple of 32 are required')
    if not isinstance(c['rounds'], int) or c['rounds'] < 0:
        raise ValueError('rounds must be a nonnegative integer')
    if not isinstance(c['synthetic_limit'], int) or c['synthetic_limit'] < 0:
        raise ValueError('synthetic_limit must be a nonnegative integer (0 means all images)')
    if not c['models'] or len(c['models']) != len(set(c['models'])):
        raise ValueError('models must be a nonempty unique list')
    for spec in c['models']:
        if len(spec.split(':')) != 2:
            raise ValueError(f'Expected arch:encoder, got {spec}')
    for key, choices in [('selection', ['random', 'confidence']), ('fuse_mode', ['prob', 'logit']),
                         ('th_method', ['fixed', 'otsu']), ('refine', ['none','jbf','crf','both']),
                         ('final_refine', ['none','jbf','crf','both'])]:
        if c[key] not in choices:
            raise ValueError(f'Invalid {key}')
    for key in ('eval_threshold', 'pseudo_threshold'):
        if not 0 < c[key] < 1:
            raise ValueError(f'{key} must lie strictly between 0 and 1')
    for key in ('pseudo_ratios', 'pseudo_weights'):
        if len(c[key]) < c['rounds'] or any(x < 0 for x in c[key]):
            raise ValueError(f'{key} must provide one nonnegative entry per round')
    if c['lr'] <= 0 or not 0 <= c['bce_weight'] <= 1 or c['grad_clip'] < 0:
        raise ValueError('Invalid learning-rate/loss/gradient settings')
    if c['loss_weighting'] not in ('legacy_batch', 'per_image'):
        raise ValueError("loss_weighting must be 'legacy_batch' or 'per_image'")
    if not 0 <= c['min_area'] < c['max_area'] <= 1:
        raise ValueError('Invalid foreground area bounds')
    if not 0 <= c['tk_bg'] < c['tk_fg'] <= 1 or c['tk_temp'] <= 0:
        raise ValueError('Invalid TK parameters')
    if not c['heads'] or any(len(h) != 3 or h[0] < 1 or not 0 <= h[1] < 1 or h[2] < 1 for h in c['heads']):
        raise ValueError('Invalid multi-head definitions')
    return c


def preflight(c):
    if any(c[k] in ('jbf', 'both') for k in ('refine', 'final_refine')) and not legacy._HAS_XIMGPROC:
        raise RuntimeError('JBF requires opencv-contrib-python-headless')
    if any(c[k] in ('crf', 'both') for k in ('refine', 'final_refine')) and legacy.dcrf is None:
        raise RuntimeError('CRF requires pydensecrf. Install it on HPC or explicitly choose a different variant.')


def derived_seed(seed, *parts):
    token = '|'.join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], 'little') % (2**31)


def set_seed(seed, deterministic=True, strict=False):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=not strict)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed); random.seed(seed)


def versions():
    packages = ['torch','torchvision','numpy','Pillow','albumentations',
                'segmentation-models-pytorch','timm','torchmetrics','torch-fidelity',
                'opencv-contrib-python-headless','pydensecrf']
    result = {}
    for name in packages:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    result['cuda_runtime'] = torch.version.cuda
    result['cudnn'] = torch.backends.cudnn.version()
    result['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return result


def code_hashes():
    return {name: file_hash(Path(__file__).parent / name) for name in
            ('common.py', 'models.py', 'engine.py', 'train.py', 'evaluate.py')}


class SegData(Dataset):
    def __init__(self, rows, root, size, seed=0, epoch=0, aug=False, round_idx=0):
        self.rows, self.root, self.seed, self.epoch = rows, root, seed, epoch
        transforms = [A.Resize(size, size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)]
        if aug:
            transforms += [A.HorizontalFlip(p=0.5),
                           A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.15, rotate_limit=10,
                                              p=0.5, border_mode=cv2.BORDER_REFLECT)]
            if round_idx == 0:
                transforms += [A.RandomBrightnessContrast(0.1, 0.1, p=0.25), A.GaussianBlur(3, p=0.1)]
        transforms += [A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()]
        self.tf = A.Compose(transforms)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = Path(row['image_absolute']) if 'image_absolute' in row else resolve(self.root, row['image'])
        with Image.open(path) as image:
            rgb = np.array(image.convert('RGB'))
        mask_path = Path(row['mask_absolute']) if 'mask_absolute' in row else (
            resolve(self.root, row['mask']) if 'mask' in row else None)
        target = None
        if mask_path is not None:
            with Image.open(mask_path) as image:
                target = (np.array(image.convert('L')) > row.get('mask_threshold', 127)).astype(np.uint8)
            if target.shape != rgb.shape[:2]:
                # Pseudo masks are generated on the model grid; bring them to source size with nearest interpolation.
                if not row.get('pseudo'):
                    raise ValueError(f'Real image/mask shape mismatch: {path}')
                target = cv2.resize(target, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        # Item-specific seed is independent of worker assignment and consumption order.
        self.tf.set_random_seed(derived_seed(self.seed, self.epoch, row['id']))
        sample = self.tf(image=rgb, **({'mask': target} if target is not None else {}))
        if target is None:
            return sample['image'], index
        return sample['image'], sample['mask'].float().unsqueeze(0), index, float(row.get('weight', 1.0))


def loader(rows, root, c, workers, seed=0, epoch=0, train=False, round_idx=0):
    ds = SegData(rows, root, c['img_size'], seed, epoch, train, round_idx)
    generator = torch.Generator().manual_seed(derived_seed(seed, 'loader', epoch))
    # A size-one training batch can break BatchNorm. Preserve historical drop_last=True.
    if train and len(rows) < c['batch']:
        raise ValueError('Training set smaller than batch; reduce batch size')
    return DataLoader(ds, batch_size=c['batch'], shuffle=train, num_workers=workers,
                      drop_last=train, worker_init_fn=seed_worker, generator=generator,
                      pin_memory=torch.cuda.is_available())


def tag(spec, c):
    return spec.replace(':', '_') + ('_mh' if c['multihead'] else '_sh')


def build_model(spec, c, device, pretrained=True):
    arch, enc = spec.split(':')
    weights = c['encoder_weights'] if pretrained else None
    if c['multihead']:
        model = legacy.MultiHeadSMP(arch, enc, head_cfgs=tuple(tuple(h) for h in c['heads']),
                                   encoder_weights=weights)
    else:
        model = legacy.SingleHeadSMP(arch, enc, encoder_weights=weights)
    return model.to(device)


def load_models(folder, c, device):
    result = []
    for spec in c['models']:
        model = build_model(spec, c, device, pretrained=False)
        state = torch.load(Path(folder) / (tag(spec, c) + '.pth'), map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.eval()
        result.append(model)
    return result


def fuse(probs, mode):
    if mode == 'prob':
        return torch.stack(probs).mean(0)
    return torch.sigmoid(torch.stack([torch.logit(p.clamp(1e-6, 1-1e-6)) for p in probs]).mean(0))


@torch.no_grad()
def predict(models, images, c, use_tk=False):
    raw, refined = [], []
    for model in models:
        model.eval()
        prob = torch.sigmoid(legacy.fuse_heads_logits(model(images))).clamp(1e-6, 1-1e-6)
        raw.append(prob)
        if use_tk:
            feat = legacy.extract_decoder_feat_smp(model.base, images)
            if c['tk_feat_norm']:
                feat = F.normalize(feat, p=2, dim=1)
            feat = F.interpolate(feat, size=prob.shape[-2:], mode='bilinear', align_corners=False)
            maps = [legacy.transductive_kmeans_refine(
                feat[i].permute(1,2,0).contiguous(), prob[i,0].contiguous(),
                num_iters=c['tk_iters'], fg_thresh=c['tk_fg'], bg_thresh=c['tk_bg'],
                use_xy=c['tk_use_xy'], xy_weight=c['tk_xy_weight'],
                soft=c['tk_soft'], temperature=c['tk_temp']) for i in range(len(images))]
            refined.append(torch.stack(maps).unsqueeze(1))
    raw_fused = fuse(raw, c['fuse_mode'])
    refined_fused = fuse(refined, c['fuse_mode']) if use_tk else raw_fused
    if not torch.isfinite(raw_fused).all() or not torch.isfinite(refined_fused).all():
        raise FloatingPointError('Nonfinite model probabilities')
    return raw, raw_fused, refined_fused


def refine(rgb, prob, c, method):
    kwargs = {k: c[k] for k in c if k.startswith(('jbf_', 'crf_'))}
    result = legacy.refine_prob(rgb, prob, method, **kwargs)
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite refined probabilities')
    return result


def loss_per_image(logits, target, bce_weight):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean((1,2,3))
    prob = torch.sigmoid(logits)
    dice = 1 - (2 * (prob * target).sum((1,2,3)) + 1e-6) / (
        prob.sum((1,2,3)) + target.sum((1,2,3)) + 1e-6)
    return bce_weight * bce + (1 - bce_weight) * dice


def loss_per_batch(logits, target, bce_weight, dice_eps=1e-6):
    """Match ultimate.py's scalar BCE + batch-mean Dice objective."""
    prob = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    numerator = 2 * (prob * target).sum((2, 3)) + dice_eps
    denominator = (prob + target).sum((2, 3)) + dice_eps
    dice = 1 - (numerator / denominator).mean()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce_weight * bce + (1 - bce_weight) * dice


def train_epoch(model, batches, optimizer, device, c):
    model.train()
    total, seen = 0.0, 0
    amp = c['amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    for image, target, _, weight in batches:
        image, target = image.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            heads = model(image)
            if c['loss_weighting'] == 'legacy_batch':
                loss = torch.stack([loss_per_batch(h, target, c['bce_weight']) for h in heads]).mean()
                # Preserve ultimate.py: scalar batch loss scaled by mean sample weight.
                loss = (loss * weight.to(device=device, dtype=loss.dtype).view(-1, 1, 1, 1)).mean()
            else:
                losses = torch.stack([loss_per_image(h, target, c['bce_weight']) for h in heads]).mean(0)
                loss = (losses * weight.to(device=device, dtype=losses.dtype)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss; no optimizer step taken')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if c['grad_clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), c['grad_clip'], error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
        total += loss.item() * len(image); seen += len(image)
    return total / seen


@torch.no_grad()
def evaluate(models, rows, root, c, device, workers, out=None, include_bases=False,
             postprocess=True, save_masks=False, save_probabilities=False):
    names = ([tag(s, c) for s in c['models']] if include_bases else []) + ['ensemble']
    records = {name: [] for name in names}
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        if save_masks:
            (out / 'masks').mkdir()
        if save_probabilities:
            (out / 'probs').mkdir()
    for images, targets, indices, _ in loader(rows, root, c, workers):
        base, fused, _ = predict(models, images.to(device), c)
        streams = (base if include_bases else []) + [fused]
        for name, tensor in zip(names, streams):
            for j, index in enumerate(indices.tolist()):
                row = rows[index]
                prob = tensor[j,0].cpu().numpy().astype(np.float32)
                if name == 'ensemble' and postprocess and c['final_refine'] != 'none':
                    with Image.open(resolve(root, row['image'])) as image:
                        prob = refine(np.array(image.convert('RGB')), prob, c, c['final_refine'])
                pred = prob >= c['eval_threshold']
                record = dict(id=row['id'], group=row['group'], **binary_metrics(pred, targets[j,0].numpy()))
                records[name].append(record)
                if out and name == 'ensemble' and (save_masks or save_probabilities):
                    key = hashlib.sha256(row['id'].encode()).hexdigest()[:20]
                    if save_masks:
                        Image.fromarray(pred.astype(np.uint8) * 255).save(out / 'masks' / (key + '.png'))
                    if save_masks or save_probabilities:
                        record['prediction_file'] = key
                    if save_probabilities:
                        np.save(out / 'probs' / (key + '.npy'), prob)
    result = {}
    for name, values in records.items():
        result[name] = dict(summary=summarize(values), per_image=values)
    if out:
        write_json(out / 'metrics.json', result)
    return result
