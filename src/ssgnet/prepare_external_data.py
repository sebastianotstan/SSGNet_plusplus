"""Reproducible external cohort export. Sources are never modified."""
import argparse
import csv
import hashlib
import io
import json
import math
import struct
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_image(path):
    """Repair DMID's invalid SampleFormat count in memory, not on disk."""
    path = Path(path)
    repaired = False
    if path.suffix.lower() in ('.tif', '.tiff'):
        data = bytearray(path.read_bytes())
        endian = '<' if data[:2] == b'II' else '>'
        if struct.unpack_from(endian + 'H', data, 2)[0] != 42:
            raise ValueError('Only classic TIFF supported: ' + str(path))
        offset = struct.unpack_from(endian + 'I', data, 4)[0]
        count = struct.unpack_from(endian + 'H', data, offset)[0]
        entries = {}
        for i in range(count):
            pos = offset + 2 + 12 * i
            tag, typ, n, value = struct.unpack_from(endian + 'HHII', data, pos)
            entries[tag] = (pos, typ, n, value)
        samples = entries.get(277, (0, 3, 1, 1))[3] & 65535
        if 339 in entries and entries[339][2] != samples:
            pos, typ, n, value = entries[339]
            if typ != 3 or n != 3 or samples != 4:
                raise ValueError('Unexpected SampleFormat defect: ' + str(path))
            vals = struct.unpack_from(endian + '3H', data, value)
            if vals != (1, 1, 1):
                raise ValueError('Unexpected signed TIFF values: ' + str(path))
            if len(data) % 2:
                data.append(0)
            new_offset = len(data)
            data.extend(struct.pack(endian + '4H', 1, 1, 1, 1))
            struct.pack_into(endian + 'II', data, pos + 4, 4, new_offset)
            repaired = True
        with Image.open(io.BytesIO(data)) as image:
            return image.convert('RGB'), repaired
    with Image.open(path) as image:
        return image.convert('RGB'), repaired


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pixel_hash(image):
    return hashlib.sha256(str(image.size).encode() + image.convert('RGB').tobytes()).hexdigest()


def contact_sheet(pairs, path, columns=4):
    """Image alongside binary-mask overlay; labels are source IDs."""
    tile_w, tile_h = 384, 216
    canvas = Image.new('RGB', (columns * tile_w, math.ceil(len(pairs)/columns) * tile_h), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (name, image, mask) in enumerate(pairs):
        x, y = (i % columns) * tile_w, (i // columns) * tile_h
        image = image.convert('RGB').resize((192, 192))
        m = np.array(mask.convert('L').resize((192, 192), Image.Resampling.NEAREST)) > 127
        overlay = np.array(image).copy()
        overlay[m] = (overlay[m].astype(float)*.55 + np.array([255,0,0])*.45).astype('uint8')
        canvas.paste(image, (x,y+22)); canvas.paste(Image.fromarray(overlay), (x+192,y+22))
        draw.text((x+3,y+4),name,fill='black')
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path)


def phash(image):
    a = np.asarray(image.convert('L').resize((32,32), Image.Resampling.LANCZOS), dtype=float)
    # Orthonormal DCT-II for the low-frequency block, implemented with NumPy
    # so this preparation script does not require SciPy on the local machine.
    n = 32
    x = np.arange(n, dtype=float)
    u = np.arange(8, dtype=float)[:, None]
    basis = np.cos(np.pi * (2 * x[None, :] + 1) * u / (2 * n))
    basis[0] *= math.sqrt(1 / n)
    basis[1:] *= math.sqrt(2 / n)
    coeff = (basis @ a @ basis.T).ravel()
    return np.packbits(coeff > np.median(coeff[1:])).tobytes().hex()


def fingerprints(image):
    return [phash(v) for v in [image, image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), image.transpose(Image.Transpose.ROTATE_180),
        image.transpose(Image.Transpose.ROTATE_90), image.transpose(Image.Transpose.ROTATE_270),
        image.transpose(Image.Transpose.TRANSPOSE), image.transpose(Image.Transpose.TRANSVERSE)]]


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys or ['status'])
        w.writeheader();w.writerows(rows)


def reference_index(real, cache):
    paths=sorted(p for p in real.rglob('*') if p.is_file() and p.parent.name=='images'
                 and 'external' not in p.parts and p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'})
    if not paths: raise ValueError('No original REAL images found')
    old={}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r=json.loads(line);old[r['path']]=r
    result=[]
    cache.parent.mkdir(parents=True,exist_ok=True)
    with cache.open('a',encoding='utf-8') as f:
        for i,p in enumerate(paths):
            key=p.as_posix();st=p.stat();r=old.get(key)
            if not r or r['size_bytes']!=st.st_size or r['mtime_ns']!=st.st_mtime_ns:
                im,_=load_image(p)
                r=dict(path=key,size_bytes=st.st_size,mtime_ns=st.st_mtime_ns,
                       width=im.width,height=im.height,sha256=sha(p),pixel_sha256=pixel_hash(im),phash=phash(im))
                f.write(json.dumps(r)+'\n');f.flush()
            result.append(r)
            if (i+1)%250==0: print('Reference images indexed:',i+1,'/',len(paths),flush=True)
    return result


class Screen:
    def __init__(self, refs, distance):
        self.refs=refs;self.distance=distance
        self.exact=defaultdict(list)
        for r in refs:self.exact[r['pixel_sha256']].append(r['path'])
        self.bits=np.array([list(bytes.fromhex(r['phash'])) for r in refs],dtype=np.uint8)
        self.lut=np.array([int(i).bit_count() for i in range(256)],dtype=np.uint8)

    def check(self, image):
        exact=self.exact.get(pixel_hash(image),[])
        best=np.full(len(self.refs),65,dtype=np.int16)
        for h in fingerprints(image):
            b=np.frombuffer(bytes.fromhex(h),dtype=np.uint8)
            d=self.lut[np.bitwise_xor(self.bits,b)].sum(axis=1)
            best=np.minimum(best,d)
        idx=np.flatnonzero(best<=self.distance)
        return exact,[dict(path=self.refs[i]['path'],distance=int(best[i])) for i in idx],int(best.min())


def candidates(source):
    ph2=source/'ph2'/'PH2Dataset'/'PH2 Dataset images'
    for case in sorted(ph2.iterdir()):
        if not case.is_dir():continue
        name=case.name
        yield dict(cohort='ph2',id=name,image=case/(name+'_Dermoscopic_Image')/(name+'.bmp'),
                   mask=case/(name+'_lesion')/(name+'_lesion.bmp'),tasks=['isic2017','isic2018'])
    images=source/'polyps'/'train'/'train';masks=source/'polyps'/'train_gt'/'train_gt'
    for p in sorted(images.glob('*.jpeg')):
        yield dict(cohort='neopolyp',id=p.stem,image=p,mask=masks/p.name,tasks=['polyps'])
    masks=source/'dmid'/'ROI Masks'/'ROI Masks'
    for p in sorted(masks.glob('*.tif')):
        yield dict(cohort='dmid',id=p.stem,image=source/'dmid'/'TIFF Images'/'TIFF Images'/p.name,
                   mask=p,tasks=['cbis'])


def convert(candidate):
    image,fix_image=load_image(candidate['image']);mask,fix_mask=load_image(candidate['mask'])
    if image.size!=mask.size:raise ValueError('Image/mask dimensions differ: '+candidate['id'])
    rgb=np.asarray(mask)
    if candidate['cohort']=='neopolyp':
        # JPEG colour masks: threshold chroma against black; merge red and green.
        rgb=rgb.astype(np.int16)
        fg=(np.maximum(rgb[:,:,0],rgb[:,:,1])>127)&(np.maximum(rgb[:,:,0],rgb[:,:,1])-rgb[:,:,2]>64)
        rule='max(R,G)>127 and max(R,G)-B>64; red/green union'
    else:
        fg=np.asarray(mask.convert('L'))>127;rule='grayscale>127'
    if not fg.any():raise ValueError('Empty source mask: '+candidate['id'])
    details=dict(original_width=image.width,original_height=image.height,mask_rule=rule,
                 source_foreground_pixels=int(fg.sum()),tiff_tag_repaired=fix_image or fix_mask,
                 crop_box='',crop_side='')
    raw=image
    mask=Image.fromarray(fg.astype('uint8')*255)
    if candidate['cohort']=='dmid':
        ys,xs=np.nonzero(fg)
        bbox=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)
        side=max(512,math.ceil(1.5*max(bbox[2]-bbox[0],bbox[3]-bbox[1])))
        left=math.floor((bbox[0]+bbox[2]-side)/2);top=math.floor((bbox[1]+bbox[3]-side)/2)
        box=(left,top,left+side,top+side)
        cropped_mask=mask.crop(box)
        if np.count_nonzero(np.array(cropped_mask))!=int(fg.sum()):raise AssertionError('Crop lost source lesion pixels')
        image=image.crop(box).resize((512,512),Image.Resampling.LANCZOS)
        mask=cropped_mask.resize((512,512),Image.Resampling.NEAREST)
        details.update(crop_box=json.dumps(box),crop_side=side,lesion_bbox=json.dumps(bbox),
                       black_padding=left<0 or top<0 or box[2]>raw.width or box[3]>raw.height)
    if not np.asarray(mask).any():raise ValueError('Mask vanished during resizing')
    details.update(output_width=image.width,output_height=image.height,
                   output_foreground_pixels=int(np.count_nonzero(np.asarray(mask))))
    return image,mask,raw,details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('new_dataset'))
    p.add_argument('--real',type=Path,default=Path('REAL'))
    p.add_argument('--out',type=Path,default=Path('external_test'))
    p.add_argument('--cache',type=Path,default=Path('revision/checks/external_index.jsonl'))
    p.add_argument('--near-distance',type=int,default=6)
    a=p.parse_args()
    if a.out.exists():p.error('Output already exists; choose a new --out. No overwriting.')
    if not 0<=a.near_distance<=16:p.error('Invalid perceptual distance')
    refs=reference_index(a.real,a.cache)
    a.out.mkdir();reports=a.out/'reports';reports.mkdir()
    write_csv(reports/'original_images.csv',refs)
    screen=Screen(refs,a.near_distance)
    records=[];overlaps=[];seen={};previews=defaultdict(list);review=[]
    all_candidates=list(candidates(a.source))
    for i,c in enumerate(all_candidates):
        image,mask,raw,detail=convert(c)
        exact,near,minimum=screen.check(image)
        if c['cohort']=='dmid':
            e,n,m=screen.check(raw);exact+=e;near+=n;minimum=min(minimum,m)
        key=pixel_hash(image)
        internal=seen.get((c['cohort'],key))
        status='included'
        if exact:status='excluded_exact_overlap'
        elif near:status='quarantined_perceptual_candidate'
        elif internal:status='excluded_internal_duplicate'
        row=dict(cohort=c['cohort'],source_id=c['id'],source_image=c['image'].as_posix(),
                 source_mask=c['mask'].as_posix(),source_image_sha256=sha(c['image']),
                 source_mask_sha256=sha(c['mask']),source_group=c['cohort']+'/'+c['id'],
                 patient_id='unknown',status=status,nearest_original_phash_distance=minimum,
                 duplicate_of=internal or '',**detail)
        row['pixel_sha256']=key
        if status=='included':
            seen[(c['cohort'],key)]=c['id']
            for task in c['tasks']:
                folder=a.out/'REAL'/task/'external'
                for kind in ['images','masks']:(folder/kind).mkdir(parents=True,exist_ok=True)
                name=c['cohort']+'_'+c['id']+'.png'
                ip=folder/'images'/name;mp=folder/'masks'/name
                image.save(ip);mask.save(mp)
                records.append(dict(row,dataset=task,image=ip.relative_to(a.out/'REAL').as_posix(),
                                    mask=mp.relative_to(a.out/'REAL').as_posix(),
                                    image_sha256=sha(ip),mask_sha256=sha(mp)))
            if len(previews[c['cohort']])<16:previews[c['cohort']].append((c['id'],image.copy(),mask.copy()))
        else:
            records.append(dict(row,dataset=';'.join(c['tasks'])))
            for path in exact:overlaps.append(dict(cohort=c['cohort'],source_id=c['id'],kind='exact',original=path,distance=0))
            for hit in near:overlaps.append(dict(cohort=c['cohort'],source_id=c['id'],kind='perceptual_candidate',original=hit['path'],distance=hit['distance']))
            if near and len(review)<24:
                match,_=load_image(Path(min(near,key=lambda n:n['distance'])['path']))
                review.append((c['cohort']+'/'+c['id'],image.copy(),Image.new('L',image.size)))
                review.append(('original '+min(near,key=lambda n:n['distance'])['path'],match,Image.new('L',match.size)))
        if (i+1)%25==0:print('External candidates processed:',i+1,'/',len(all_candidates),flush=True)
    write_csv(reports/'pairs.csv',records);write_csv(reports/'overlap_candidates.csv',overlaps)
    # Source files lacking segmentation ground truth are not treated as negatives.
    annotated={c['image'] for c in all_candidates}
    unused=[dict(path=p.as_posix(),reason='no paired segmentation mask in supplied source')
            for p in sorted((a.source/'dmid'/'TIFF Images'/'TIFF Images').glob('*.tif')) if p not in annotated]
    unused += [dict(path=p.as_posix(),reason='official challenge test image; no ground truth supplied')
               for p in sorted((a.source/'polyps'/'test'/'test').glob('*.jpeg'))]
    write_csv(reports/'unused_sources.csv',unused)
    for cohort,pairs in previews.items():contact_sheet(pairs,reports/(cohort+'_preview.jpg'))
    if review:contact_sheet(review,reports/'overlap_review.jpg',columns=2)
    counts=Counter(r['dataset'] for r in records if r['status']=='included')
    summary=dict(counts=dict(counts),source_candidates=len(all_candidates),
                 original_images_screened=len(refs),status_counts=dict(Counter((r['cohort']+':'+r['status']) for r in records)),
                 near_distance=a.near_distance,patient_independence_verified=False,
                 overlap_scope='All original REAL image folders, excluding external. RGB decoded SHA256 plus 64-bit pHash with 8 orientations. DMID source full mammograms and output crops both checked. Does not rule out arbitrary crops, related frames, or shared patients.',
                 selection='All eligible labelled cases. No sampling or model-performance selection. Perceptual candidates conservatively quarantined.',
                 dmid_crop='One crop per masked mammogram. Union foreground bbox, square side=max(512,ceil(1.5*max(bbox width,height))), black padding if needed, Lanczos image and nearest-neighbour mask resize to 512. Ground-truth-assisted ROI segmentation, not detection.',
                 ph2='Identical accepted PH2 cohort exported to both ISIC tasks; not two independent external cohorts.',
                 code_sha256=sha(Path(__file__)))
    (reports/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    (a.out/'READY.json').write_text(json.dumps(dict(counts=dict(counts),pairs_sha256=sha(reports/'pairs.csv')),indent=2))


if __name__=='__main__':main()
