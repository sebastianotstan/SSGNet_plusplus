"""Paired seed differences and image/group bootstrap intervals on saved test predictions."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import t
from common import read_json, write_json, METRICS


def t_interval(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError('At least two paired seeds are required for a seed confidence interval')
    mean = float(values.mean())
    half = float(t.ppf(0.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return [mean-half, mean+half]


def load_runs(root, requested_round, predictor, metric):
    data = {}
    for folder in sorted(Path(root).glob('seed_*')):
        run = read_json(folder / 'run.json')
        read_json(folder / 'test' / 'summary.json')  # require complete evaluation
        selected = read_json(folder / 'complete.json')['selected_round_by_validation']
        round_idx = selected if requested_round == 'selected' else int(requested_round)
        result = read_json(folder / 'test' / f'round_{round_idx}' / 'metrics.json')[predictor]
        if run['seed'] in data:
            raise ValueError('Duplicate seed')
        records = result['per_image']
        if len({r['id'] for r in records}) != len(records):
            raise ValueError('Duplicate evaluation image')
        data[run['seed']] = (run, {r['id']:r for r in records}, round_idx)
    if not data:
        raise ValueError(f'No seed runs under {root}')
    return data


def compare(a, b, metric, nboot, bootstrap_seed):
    if set(a) != set(b):
        raise ValueError('Both comparisons must contain exactly the same seeds; no silent dropping')
    seeds = sorted(a)
    matrices, groups, sample_ids, deltas = [], None, None, []
    expected_split = None
    for seed in seeds:
        arun, av, _ = a[seed]
        brun, bv, _ = b[seed]
        if arun['dataset'] != brun['dataset'] or arun['split_sha256'] != brun['split_sha256']:
            raise ValueError('Paired runs must use the same dataset and frozen split')
        if expected_split is not None and arun['split_sha256'] != expected_split:
            raise ValueError('Split changed between training seeds')
        expected_split = arun['split_sha256']
        if set(av) != set(bv):
            raise ValueError('Evaluation image IDs differ')
        valid = sorted(k for k in av if av[k][metric] is not None and bv[k][metric] is not None)
        if not valid or (sample_ids is not None and valid != sample_ids):
            raise ValueError('No valid cases, or metric eligibility changed between seeds')
        if any(av[k]['group'] != bv[k]['group'] for k in valid):
            raise ValueError('Group mappings differ')
        sample_ids = valid
        new_groups = [av[k]['group'] for k in valid]
        if groups is not None and groups != new_groups:
            raise ValueError('Group mapping changed between seeds')
        groups = new_groups
        row = np.array([bv[k][metric] - av[k][metric] for k in valid])
        matrices.append(row)
        deltas.append(dict(seed=seed, a=float(np.mean([av[k][metric] for k in valid])),
                           b=float(np.mean([bv[k][metric] for k in valid])),
                           difference=float(row.mean()), round_a=a[seed][2], round_b=b[seed][2]))
    differences = [r['difference'] for r in deltas]
    seed_ci = t_interval(differences)
    matrix = np.stack(matrices)
    unique_groups = sorted(set(groups))
    indices = [np.flatnonzero(np.array(groups) == g) for g in unique_groups]
    sums = np.array([matrix[:, ix].mean(0).sum() for ix in indices])
    sizes = np.array([len(ix) for ix in indices])
    rng = np.random.default_rng(bootstrap_seed)
    boot = []
    for _ in range(nboot):
        sampled = rng.integers(0, len(indices), len(indices))
        boot.append(float(sums[sampled].sum() / sizes[sampled].sum()))
    return dict(metric=metric, direction='B minus A', units='fraction; multiply by 100 for percentage points',
                n_seeds=len(seeds), n_images=len(sample_ids), n_groups=len(indices),
                mean_difference=float(np.mean(differences)), paired_seed_t_ci95=seed_ci,
                per_seed=deltas,
                group_bootstrap_ci95_conditional_on_trained_seeds=np.quantile(boot,[0.025,0.975]).tolist(),
                bootstrap_replicates=nboot, bootstrap_seed=bootstrap_seed,
                interpretation='Seed t-CI assumes independent run differences with approximately normal errors. '
                'Small seed counts give imprecise intervals. Group bootstrap conditions on these trained seeds; '
                'without patient IDs it describes image/content-group uncertainty, not patient uncertainty. '
                'Intervals are unadjusted for multiple comparisons.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a', type=Path, required=True, help='Dataset directory containing seed_* runs')
    p.add_argument('--b', type=Path, required=True)
    p.add_argument('--round_a', default='0', help='Integer or selected (by validation)')
    p.add_argument('--round_b', default='2')
    p.add_argument('--predictor_a', default='ensemble')
    p.add_argument('--predictor_b', default='ensemble')
    p.add_argument('--metric', choices=METRICS, default='dice')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.bootstrap < 100 or args.out.exists():
        p.error('Use at least 100 bootstrap replicates and a new output filename')
    result = compare(load_runs(args.a,args.round_a,args.predictor_a,args.metric),
                     load_runs(args.b,args.round_b,args.predictor_b,args.metric),
                     args.metric,args.bootstrap,args.seed)
    write_json(args.out,result)
    print('B - A:', result['mean_difference'], 'paired seed 95% CI:',result['paired_seed_t_ci95'])


if __name__ == '__main__':
    main()
