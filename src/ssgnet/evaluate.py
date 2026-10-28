"""Evaluate frozen runs on test only after all experimental choices are fixed."""
import argparse
from pathlib import Path
import torch
from common import read_json, write_json, file_hash, validate_manifest, verify_files
from engine import load_config, load_models, evaluate, preflight, set_seed, code_hashes, versions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True, help='.../dataset/seed_42')
    p.add_argument('--real_root', type=Path, default=Path('REAL'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--save-probabilities', action='store_true',
                   help='Also save per-image float probability arrays as .npy files (off by default)')
    a = p.parse_args()
    run, complete = read_json(a.run / 'run.json'), read_json(a.run / 'complete.json')
    for name in ('run', 'split'):
        if file_hash(a.run / (name + '.json')) != complete[name + '_sha256']:
            raise ValueError(f'Frozen {name} metadata was modified')
    manifest = read_json(a.run / 'split.json')
    validate_manifest(manifest)
    if run['code_sha256'] != code_hashes():
        raise ValueError('Revision code changed since training; use the code version saved with this experiment')
    c = load_config(run['config'])
    preflight(c)
    for relative, digest in complete['checkpoint_sha256'].items():
        if file_hash(a.run / relative) != digest:
            raise ValueError(f'Checkpoint changed: {relative}')
    rows = [r for r in manifest['rows'] if r['split'] == 'test']
    verify_files(rows, a.real_root)
    device = torch.device(a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; choose --device cpu')
    torch.set_num_threads(1)
    set_seed(run['seed'], c['deterministic'], c['strict_determinism'])
    output = a.run / 'test'
    output.mkdir(exist_ok=False)
    summaries = []
    for round_idx in range(c['rounds'] + 1):
        models = load_models(a.run / f'round_{round_idx}' / 'checkpoints', c, device)
        result = evaluate(models, rows, a.real_root, c, device, a.workers,
                          out=output / f'round_{round_idx}', include_bases=True,
                          save_masks=True, save_probabilities=a.save_probabilities)
        summaries.append(dict(round=round_idx, **result['ensemble']['summary']['macro_image']))
        del models
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    selected = complete['selected_round_by_validation']
    write_json(output / 'summary.json', dict(dataset=run['dataset'], seed=run['seed'],
               selected_round_by_validation=selected, selected_test=summaries[selected],
               all_predeclared_rounds=summaries, versions=versions(),
               provenance=manifest['provenance'], patient_independence_verified=False,
               prediction_grid=[c['img_size'],c['img_size']], threshold=c['eval_threshold'],
               prediction_masks_saved=True, probability_maps_saved=a.save_probabilities))
    print(f'Test evaluation saved to {output}. Do not select a new winner using these results.')


if __name__ == '__main__':
    main()
