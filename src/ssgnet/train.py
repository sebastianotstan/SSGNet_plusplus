"""Train fresh seeded rounds using train/validation only. Test is a separate command."""
import argparse
import csv
import gc
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from common import read_json, write_json, file_hash, image_hash, list_images, validate_manifest, verify_files
from engine import (load_config, preflight, set_seed, derived_seed, versions, code_hashes,
                    loader, build_model, tag, load_models, train_epoch, evaluate, predict, refine)
import models as legacy


def tau(prob, c, round_idx):
    if c['th_method'] == 'fixed':
        return max(c['pseudo_threshold'], 0.55) if round_idx == 1 else c['pseudo_threshold']
    import cv2
    value, _ = cv2.threshold((np.clip(prob, 0, 1) * 255).astype(np.uint8),
                            0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return float(value / 255)


def make_pseudo(models, synth, c, device, workers, seed, round_idx, nreal, folder):
    folder.mkdir(parents=True, exist_ok=False)
    masks_dir = folder / 'masks'
    masks_dir.mkdir()
    records = []
    for images, indices in loader(synth, '.', c, workers):
        _, raw, tk = predict(models, images.to(device), c, use_tk=c['tk'])
        for j, index in enumerate(indices.tolist()):
            row = synth[index]
            raw_prob = raw[j,0].cpu().numpy()
            tk_prob = tk[j,0].cpu().numpy()
            selection_prob = raw_prob if c['select_use_raw'] else tk_prob
            selection_tau = tau(selection_prob, c, round_idx)
            score, area = legacy.pseudo_confidence(selection_prob, selection_tau,
                                                  min_area=c['min_area'], max_area=c['max_area'], ent_w=c['ent_w'])
            records.append(dict(id=row['id'], image_absolute=row['image_absolute'],
                                mask_file='', pseudo=True,
                                score=score, area=area, selection_tau=selection_tau,
                                training_tau=None, kept=False))
    count = int(nreal * c['pseudo_ratios'][round_idx - 1])
    if c['pseudo_max_count'] > 0:
        count = min(count, c['pseudo_max_count'])
    if c['selection'] == 'random':
        candidates = list(range(len(records)))
        rng = np.random.RandomState(derived_seed(seed, 'pseudo', round_idx))
        rng.shuffle(candidates)
    else:
        candidates = sorted((i for i, r in enumerate(records) if r['score'] >= 0),
                            key=lambda i: (-records[i]['score'], records[i]['id']))
    chosen = candidates[:count]
    for index in chosen:
        records[index]['kept'] = True
    if count > 0 and not chosen:
        raise ValueError('No pseudo images passed selection; do not silently report a synthetic-data experiment')

    # Re-run inference only for selected images to materialize training masks.
    # This avoids creating one persistent file per synthetic candidate.
    selected_rows = [synth[index] for index in chosen]
    for images, indices in loader(selected_rows, '.', c, workers):
        _, _, tk = predict(models, images.to(device), c, use_tk=c['tk'])
        for j, selected_index in enumerate(indices.tolist()):
            record_index = chosen[selected_index]
            row = selected_rows[selected_index]
            tk_prob = tk[j,0].cpu().numpy()
            with Image.open(row['image_absolute']) as image:
                training_prob = refine(np.array(image.convert('RGB')), tk_prob, c, c['refine'])
            training_tau = tau(training_prob, c, round_idx)
            mask_name = Path(row['image_absolute']).stem + '.png'
            mask_path = masks_dir / mask_name
            Image.fromarray((training_prob >= training_tau).astype(np.uint8) * 255).save(mask_path)
            records[record_index]['mask_file'] = mask_name
            records[record_index]['training_tau'] = training_tau
            records[record_index]['mask_absolute'] = str(mask_path.resolve())

    # Keep the full selection audit but omit temporary absolute mask paths.
    csv_records = [{k: v for k, v in record.items() if k != 'mask_absolute'} for record in records]
    with (folder / 'selection.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_records[0]))
        w.writeheader(); w.writerows(csv_records)
    print(f'Round {round_idx}: selected {len(chosen)}/{len(records)} synthetic images', flush=True)
    return [dict(records[i], weight=c['pseudo_weights'][round_idx - 1]) for i in chosen]


def run_one(a, c, manifest, synth, seed):
    out = a.out_root / manifest['dataset'] / f'seed_{seed}'
    if out.exists():
        raise FileExistsError(f'Refusing to reuse run directory: {out}. Use a new output root for fresh runs.')
    train_rows = [r for r in manifest['rows'] if r['split'] == 'train']
    val_rows = [r for r in manifest['rows'] if r['split'] == 'val']
    # Only training/validation image and mask bytes are read here.
    verify_files(train_rows + val_rows, a.real_root)
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu for a local check')
    preflight(c)
    out.mkdir(parents=True)
    write_json(out / 'split.json', manifest)
    write_json(out / 'run.json', dict(config=c, seed=seed, dataset=manifest['dataset'],
                                    split_sha256=file_hash(a.split), code_sha256=code_hashes(),
                                    versions=versions(), workers=a.workers,
                                    synthetic_files=synth,
                                    pseudo_mask_policy='Selected masks are temporary during round training; selection CSVs remain for audit and masks can be regenerated from the prior-round checkpoint.',
                                    note='Fresh run; no resume; test never evaluated during training.'))
    source_dir = out / 'source'
    source_dir.mkdir()
    for name in code_hashes():
        shutil.copy2(Path(__file__).parent / name, source_dir / name)
    summaries = []
    for round_idx in range(c['rounds'] + 1):
        rdir = out / f'round_{round_idx}'
        checkpoints = rdir / 'checkpoints'
        checkpoints.mkdir(parents=True)
        pseudo = []
        if round_idx and c['pseudo_ratios'][round_idx - 1] > 0:
            teacher = load_models(out / f'round_{round_idx-1}' / 'checkpoints', c, device)
            pseudo = make_pseudo(teacher, synth, c, device, a.workers, seed, round_idx,
                                 len(train_rows), rdir / 'pseudo')
            del teacher; gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        stage_rows = train_rows + pseudo
        for spec in c['models']:
            stage_seed = derived_seed(seed, round_idx, spec)
            set_seed(stage_seed, c['deterministic'], c['strict_determinism'])
            model = build_model(spec, c, device, pretrained=(round_idx == 0))
            if round_idx:
                previous = out / f'round_{round_idx-1}' / 'checkpoints' / (tag(spec, c) + '.pth')
                model.load_state_dict(torch.load(previous, map_location='cpu', weights_only=True))
            optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
            best, best_epoch, history = -1.0, None, []
            epochs = c['epochs0'] if round_idx == 0 else c['epochs']
            for epoch in range(1, epochs + 1):
                batches = loader(stage_rows, a.real_root, c, a.workers, stage_seed, epoch,
                                 train=True, round_idx=round_idx)
                loss = train_epoch(model, batches, optimizer, device, c)
                val = evaluate([model], val_rows, a.real_root, c, device, a.workers, postprocess=False)
                metrics = val['ensemble']['summary']['macro_image']
                history.append(dict(epoch=epoch, loss=loss, **metrics))
                if metrics['dice'] > best:
                    best, best_epoch = metrics['dice'], epoch
                    # Serialize immediately: no mutable CPU state_dict alias as training continues.
                    torch.save({k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                               checkpoints / (tag(spec, c) + '.pth'))
                print(f'seed={seed} R{round_idx} {spec} epoch={epoch} loss={loss:.5f} valDice={metrics["dice"]:.5f}', flush=True)
            write_json(rdir / (tag(spec, c) + '_history.json'),
                       dict(best_epoch=best_epoch, best_dice=best, history=history))
            del model, optimizer; gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        # Pseudo masks are no longer needed after every model in this round has trained.
        for row in pseudo:
            Path(row['mask_absolute']).unlink(missing_ok=True)
        models = load_models(checkpoints, c, device)
        values = evaluate(models, val_rows, a.real_root, c, device, a.workers,
                          out=rdir / 'validation', include_bases=True, save_masks=False)
        summary = dict(round=round_idx, pseudo_count=len(pseudo),
                       ensemble_validation=values['ensemble']['summary']['macro_image'])
        summaries.append(summary)
        write_json(rdir / 'summary.json', summary)
        del models; gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    selected = max(summaries, key=lambda x: x['ensemble_validation']['dice'])['round']
    hashes = {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.glob('round_*/checkpoints/*.pth'))}
    write_json(out / 'complete.json', dict(selected_round_by_validation=selected,
                                         rounds=summaries, checkpoint_sha256=hashes,
                                         run_sha256=file_hash(out / 'run.json'),
                                         split_sha256=file_hash(out / 'split.json')))
    print(f'Completed {out}. Validation-selected round={selected}. Test remains unevaluated.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--real_root', type=Path, default=Path('REAL'))
    p.add_argument('--synth_root', type=Path, default=Path('SYNTH'))
    p.add_argument('--out_root', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42,123,2024,3407])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if len(a.seeds) != len(set(a.seeds)) or any(s < 0 or s >= 2**32 for s in a.seeds) or a.workers < 0:
        p.error('Seeds must be unique integers in [0, 2**32); workers must be nonnegative')
    c, manifest = load_config(read_json(a.config)), read_json(a.split)
    validate_manifest(manifest)
    torch.set_num_threads(1)
    synth = []
    if c['rounds'] and any(c['pseudo_ratios'][:c['rounds']]):
        folder = a.synth_root / manifest['dataset']
        folder = folder / 'train' / 'images' if (folder / 'train' / 'images').is_dir() else folder / 'images'
        heldout_hashes = {r['pixel_sha256'] for r in manifest['rows'] if r['split'] != 'train'}
        synthetic_paths = list_images(folder)
        if c['synthetic_limit']:
            synthetic_paths = synthetic_paths[:c['synthetic_limit']]
            print(f'Pilot limit: using {len(synthetic_paths)} of the sorted synthetic images in {folder}', flush=True)
        for path in synthetic_paths:
            if image_hash(path) in heldout_hashes:
                raise ValueError(f'Synthetic image exactly duplicates a held-out real image: {path}')
            synth.append(dict(id='synthetic/' + path.name, image_absolute=str(path.resolve()), sha256=file_hash(path)))
    for seed in a.seeds:
        if (a.out_root / manifest['dataset'] / f'seed_{seed}').exists():
            p.error('Output run already exists; no automatic resume or overwrite is allowed')
    for seed in a.seeds:
        run_one(a, c, manifest, synth, seed)


if __name__ == '__main__':
    main()
