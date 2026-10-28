"""One-command preflight, fresh multi-round training, and final held-out test evaluation."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--split', type=Path, required=True)
    p.add_argument('--out_root', type=Path, required=True)
    p.add_argument('--real_root', type=Path, default=Path('REAL'))
    p.add_argument('--synth_root', type=Path, default=Path('SYNTH'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--save-probabilities', action='store_true',
                   help='Also save per-image float probability arrays as .npy files (off by default)')
    a = p.parse_args()
    if not 0 <= a.seed < 2**32 or a.workers < 0:
        p.error('Seed must be in [0, 2**32); workers must be nonnegative')

    script_dir = Path(__file__).resolve().parent
    run = a.out_root / __import__('json').loads(a.split.read_text(encoding='utf-8'))['dataset'] / f'seed_{a.seed}'
    common = [
        '--config', str(a.config), '--split', str(a.split),
        '--real_root', str(a.real_root), '--synth_root', str(a.synth_root),
        '--device', a.device,
    ]

    # Stop before training if the selected device, data, dependencies, or model forward pass fails.
    subprocess.run([sys.executable, str(script_dir / 'preflight.py'), *common,
                    '--forward'], check=True)
    subprocess.run([sys.executable, str(script_dir / 'train.py'), *common,
                    '--out_root', str(a.out_root), '--seeds', str(a.seed),
                    '--workers', str(a.workers)], check=True)
    evaluate_command = [sys.executable, str(script_dir / 'evaluate.py'),
                    '--run', str(run), '--real_root', str(a.real_root),
                    '--device', a.device, '--workers', str(a.workers)]
    if a.save_probabilities:
        evaluate_command.append('--save-probabilities')
    subprocess.run(evaluate_command, check=True)
    print(f'Complete: training, validation-based round selection, and held-out evaluation are in {run}')


if __name__ == '__main__':
    main()
