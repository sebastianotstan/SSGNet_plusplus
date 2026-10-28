"""Verify exported pairs and report within-cohort perceptual neighbours."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from prepare_external_data import sha, pixel_hash, phash, write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=Path('external_test'))
    a=p.parse_args();reports=a.folder/'reports'
    rows=list(csv.DictReader((reports/'pairs.csv').open(encoding='utf-8')))
    refs=list(csv.DictReader((reports/'original_images.csv').open(encoding='utf-8')))
    originals={r['pixel_sha256'] for r in refs}
    included=[r for r in rows if r['status']=='included']
    counts=Counter();unique={};hashes={};empty_model_grid=[]
    try:
        import cv2
    except ImportError:
        cv2=None
    for r in included:
        ip=a.folder/'REAL'/r['image'];mp=a.folder/'REAL'/r['mask']
        assert sha(ip)==r['image_sha256'] and sha(mp)==r['mask_sha256'],str(ip)
        with Image.open(ip) as image,Image.open(mp) as mask:
            assert image.mode=='RGB' and mask.mode=='L'
            assert image.size==mask.size
            values=np.unique(np.asarray(mask));assert set(values)=={0,255},(str(mp),values)
            digest=pixel_hash(image);assert digest==r['pixel_sha256'] and digest not in originals
            if r['dataset']=='cbis':assert image.size==(512,512)
            # Use the model's OpenCV resize when available; otherwise use the
            # equivalent binary-mask nearest-neighbour sanity check via Pillow.
            grid=(cv2.resize(np.asarray(mask),(256,256),interpolation=cv2.INTER_NEAREST)
                  if cv2 is not None else np.asarray(mask.resize((256,256),Image.Resampling.NEAREST)))
            if not grid.any():
                empty_model_grid.append(r['image'])
            unique[(r['cohort'],r['source_id'])]=(r,phash(image))
        counts[r['dataset']]+=1
        hashes[(r['dataset'],r['source_id'])]=(r['image_sha256'],r['mask_sha256'])
    for r in included:
        if r['dataset']=='isic2017':
            assert hashes[('isic2017',r['source_id'])]==hashes[('isic2018',r['source_id'])]
    for dataset,count in counts.items():
        root=a.folder/'REAL'/dataset/'external'
        assert len(list((root/'images').glob('*.png')))==count
        assert len(list((root/'masks').glob('*.png')))==count
    neighbours=[]
    for cohort in ['ph2','neopolyp','dmid']:
        items=[v for (c,_),v in unique.items() if c==cohort]
        for i,(r,h) in enumerate(items):
            for s,j in items[i+1:]:
                distance=(int(h,16)^int(j,16)).bit_count()
                if distance<=6:
                    neighbours.append(dict(cohort=cohort,id_a=r['source_id'],id_b=s['source_id'],
                                           distance=distance,note='Similarity flag only; may be related views or false positive. Not proof of patient identity.'))
    write_csv(reports/'within_cohort_similarity.csv',neighbours)
    source_status={ (r['cohort'],r['source_id']):r['status'] for r in rows }
    with (a.folder/'external_test_checksums.sha256').open('x',encoding='utf-8',newline='\n') as f:
        for r in included:
            for kind in ['image','mask']:
                f.write(r[kind+'_sha256']+'  REAL/'+r[kind]+'\n')
    result=dict(passed=True,counts=dict(counts),verified_exported_pairs=len(included),
                unique_external_source_images=len(unique),ph2_exports_identical=True,
                source_status_counts=dict(Counter(c+':'+status for (c,_),status in source_status.items())),
                exact_overlap_with_originals=0,within_cohort_similarity_flags=len(neighbours),
                empty_at_model_256_grid=empty_model_grid,
                model_grid_check='OpenCV nearest-neighbour' if cv2 is not None else 'Pillow nearest-neighbour fallback',
                notes='All exported SHA256, decoded image hashes, pairing, dimensions, binary labels and PH2 shared identity verified. Within-cohort near neighbours are retained and documented; frame count is not patient count.')
    (reports/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
