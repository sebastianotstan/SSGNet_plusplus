"""Regenerate selected pseudo-mask PNGs from a completed preceding-round checkpoint."""
import argparse
from pathlib import Path
import torch

from common import read_json, write_json, file_hash
from engine import load_models, preflight, set_seed, tag
from train import make_pseudo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True,
                        help='Completed run directory, e.g. revision/external_runs/mb/cbis/seed_42')
    parser.add_argument('--round', type=int, choices=(1, 2), required=True,
                        help='Pseudo-label round to recreate; uses the previous round checkpoint')
    parser.add_argument('--out', type=Path, required=True,
                        help='New output directory for selection.csv and selected mask PNGs')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    run = read_json(args.run / 'run.json')
    config, seed = run['config'], int(run['seed'])
    round_idx = args.round
    if round_idx > config['rounds'] or config['pseudo_ratios'][round_idx - 1] <= 0:
        parser.error(f'Run config has no pseudo-training for round {round_idx}')
    if args.workers < 0:
        parser.error('--workers must be nonnegative')
    if args.out.exists():
        parser.error(f'Output already exists: {args.out}; choose a new directory')

    synth = run.get('synthetic_files', [])
    if not synth:
        parser.error('Run metadata has no synthetic file list')
    for row in synth:
        path = Path(row['image_absolute'])
        if not path.is_file() or file_hash(path) != row['sha256']:
            raise ValueError(f'Synthetic image missing or changed since training: {path}')

    split = read_json(args.run / 'split.json')
    nreal = sum(row['split'] == 'train' for row in split['rows'])
    checkpoint_dir = args.run / f'round_{round_idx - 1}' / 'checkpoints'
    complete_path = args.run / 'complete.json'
    if complete_path.exists():
        hashes = read_json(complete_path).get('checkpoint_sha256', {})
        for model_spec in config['models']:
            relative = f'round_{round_idx - 1}/checkpoints/{tag(model_spec, config)}.pth'
            checkpoint = args.run / relative
            if hashes.get(relative) != file_hash(checkpoint):
                raise ValueError(f'Checkpoint does not match the completed run record: {checkpoint}')

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu')
    torch.set_num_threads(1)
    set_seed(seed, config['deterministic'], config['strict_determinism'])
    preflight(config)
    models = load_models(checkpoint_dir, config, device)
    selected = make_pseudo(models, synth, config, device, args.workers, seed,
                           round_idx, nreal, args.out)
    write_json(args.out / 'regeneration.json', dict(
        run=str(args.run.resolve()), round=round_idx,
        teacher_checkpoint_round=round_idx - 1,
        selected_count=len(selected), synthetic_count=len(synth),
        seed=seed, config=config,
        note='Regenerated selected pseudo masks from the saved teacher checkpoint.'))
    print(f'Regenerated {len(selected)} selected pseudo masks in {args.out}', flush=True)


if __name__ == '__main__':
    main()
