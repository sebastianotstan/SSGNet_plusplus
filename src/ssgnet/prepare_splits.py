"""Freeze an image-level holdout from existing validation files; never move data."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import random
from common import collect_pairs, validate_manifest, write_json


def make_split(root, dataset, fraction, seed, groups_csv=None, exclude_train_duplicates=False):
    if not 0 < fraction < 1:
        raise ValueError('test_fraction must be between zero and one')
    rows = collect_pairs(root, dataset, 'train') + collect_pairs(root, dataset, 'val')
    mapping = {}
    if groups_csv:
        with open(groups_csv, newline='', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r['image'] in mapping or not r['group'].strip():
                    raise ValueError('Duplicate image or blank group in groups CSV')
                mapping[r['image']] = r
    for r in rows:
        if groups_csv and r['image'] not in mapping:
            raise ValueError(f'Missing group mapping: {r["image"]}')
        m = mapping.get(r['image'], {})
        r['group'] = m.get('group', r['pixel_sha256'])
        r['stratum'] = m.get('class', '').strip() or 'unknown'
        r['split'] = r['original_split']
    train_groups = {r['group'] for r in rows if r['original_split'] == 'train'}
    train_pixels = {r['pixel_sha256'] for r in rows if r['original_split'] == 'train'}
    excluded = [r for r in rows if r['original_split'] == 'val'
                and (r['group'] in train_groups or r['pixel_sha256'] in train_pixels)]
    if excluded and not exclude_train_duplicates:
        raise ValueError(f"{dataset}: {len(excluded)} old validation images overlap training. "
                         "Use --exclude_train_duplicates to quarantine them in the new manifest.")
    excluded_ids = {r['id'] for r in excluded}
    rows = [r for r in rows if r['id'] not in excluded_ids]
    pool = defaultdict(list)
    for r in rows:
        if r['original_split'] == 'val':
            pool[r['group']].append(r)
    strata = defaultdict(list)
    for group, members in pool.items():
        strata[tuple(sorted({r['stratum'] for r in members}))].append(group)
    rng = random.Random(seed)
    for label, groups in sorted(strata.items()):
        groups.sort(); rng.shuffle(groups)
        if len(groups) < 2:
            raise ValueError(f'Stratum {label} has fewer than two groups')
        ntest = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
        for group in groups[:ntest]:
            for r in pool[group]:
                r['split'] = 'test'
    result = dict(schema=1, dataset=dataset, split_seed=seed, test_fraction_of_old_val=fraction,
                  counts=dict(Counter(r['split'] for r in rows)),
                  grouping='provided_case_ids' if groups_csv else 'image_content_only',
                  patient_independence_verified=False,
                  provenance='Test carved from previously used validation data; not historically untouched.',
                  gan_training_provenance='Not verified; confirm GAN used original training images only.',
                  excluded_old_val=excluded, excluded_old_val_count=len(excluded), rows=rows)
    validate_manifest(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real_root', type=Path, default=Path('REAL'))
    p.add_argument('--datasets', nargs='+', default=['cbis', 'isic2017', 'isic2018', 'polyps'])
    p.add_argument('--out', type=Path, default=Path('revision/splits'))
    p.add_argument('--test_fraction', type=float, default=0.5, help='Fraction of OLD validation')
    p.add_argument('--seed', type=int, default=2026, help='Fixed split seed, separate from training seeds')
    p.add_argument('--groups_csv', type=Path, help='Optional columns: image (relative to REAL),group,class')
    p.add_argument('--exclude_train_duplicates', action='store_true', help='Quarantine old-val images identical to training; original files stay untouched')
    a = p.parse_args()
    root = a.real_root.resolve()
    for ds in a.datasets:
        if (a.out / f'{ds}.json').exists():
            p.error(f'Manifest already exists: {a.out / (ds + ".json")}')
    results = [make_split(root, ds, a.test_fraction, a.seed, a.groups_csv, a.exclude_train_duplicates) for ds in a.datasets]
    for result in results:
        write_json(a.out / (result['dataset'] + '.json'), result)
        print(result['dataset'], result['counts'], result['grouping'], 'excluded old-val:', result['excluded_old_val_count'])
    print('Original data untouched. Image-level splitting does not establish patient independence.')


if __name__ == '__main__':
    main()
