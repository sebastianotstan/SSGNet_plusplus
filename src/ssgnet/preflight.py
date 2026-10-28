"""Check one revision configuration before submitting an expensive HPC job."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from common import read_json, validate_manifest, verify_files, list_images, image_hash
from engine import load_config, preflight, versions, build_model, predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--real_root',type=Path,default=Path('REAL'))
    p.add_argument('--synth_root',type=Path,default=Path('SYNTH'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--forward',action='store_true',help='Forward pass with random weights; never downloads pretrained encoders')
    a=p.parse_args()
    c=load_config(read_json(a.config)); manifest=read_json(a.split)
    validate_manifest(manifest)
    preflight(c)
    # Verify every frozen file, including the external test cohort, before a
    # potentially long training allocation is started. Evaluation rechecks
    # test hashes later, but should not be the first place missing test files
    # are discovered.
    verify_files(manifest['rows'],a.real_root)
    print('Splits:',manifest['counts'])
    print('Versions:',versions())
    print('Grouping:',manifest['grouping'])
    print('Test provenance:',manifest['provenance'])
    if c['rounds'] and any(c['pseudo_ratios'][:c['rounds']]):
        folder=a.synth_root/manifest['dataset']
        folder=folder/'train'/'images' if (folder/'train'/'images').is_dir() else folder/'images'
        paths=list_images(folder)
        selected=paths[:c['synthetic_limit']] if c['synthetic_limit'] else paths
        sample=selected if c['synthetic_limit'] else selected[:min(64,len(selected))]
        heldout={r['pixel_sha256'] for r in manifest['rows'] if r['split']!='train'}
        duplicates=[]
        for image_path in sample:
            with Image.open(image_path) as image:
                image.convert('RGB').load()
            if image_hash(image_path) in heldout:
                duplicates.append(image_path.name)
        if duplicates:
            raise ValueError(f'Synthetic images duplicate held-out real images: {duplicates[:5]}')
        count_note=f'{len(selected)} will be used'
        if len(sample)<len(selected):
            count_note+=f'; first {len(sample)} checked for readability and exact held-out duplicates'
        print(f'Synthetic images: {len(paths)} found in {folder}; {count_note}')
    if a.forward:
        device=torch.device(a.device)
        if device.type=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(1)
        for spec in c['models']:
            model=build_model(spec,c,device,pretrained=False)
            _,raw,refined=predict([model],torch.zeros(2,3,c['img_size'],c['img_size'],device=device),c,use_tk=c['tk'])
            print(spec,'forward OK',tuple(raw.shape))
            del model
    print('Preflight passed. Patient-level independence cannot be checked without case identifiers.')


if __name__=='__main__':
    main()
