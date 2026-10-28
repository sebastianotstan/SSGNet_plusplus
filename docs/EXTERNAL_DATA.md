# External cohort data preparation

The raw external datasets are not included. After obtaining them under their source terms, place them in a local `new_dataset/` folder and run from the repository root. This creates a new export directory and never overwrites an existing one:

```bash
python src/ssgnet/prepare_external_data.py \
  --source new_dataset --real REAL --out external_test \
  --cache outputs/checks/external_index.jsonl
python src/ssgnet/verify_external_data.py --folder external_test
```

Review `external_test/reports/` and the generated images/masks. Then copy the contents of `external_test/REAL/` into the repository's `REAL/` tree. The expected dataset layout and exact PH2, DMID, and BKAI-IGH NeoPolyp preprocessing rules are documented below. The DMID procedure is ground-truth-assisted ROI segmentation, not whole-mammogram detection. PH2 is one shared cohort for both ISIC tasks, not two independent cohorts.

Create external manifests in a fresh directory (the included manifests are frozen and the tools refuse to overwrite files):

```bash
python src/ssgnet/prepare_external_split.py \
  --real_root REAL --datasets cbis isic2017 isic2018 polyps \
  --external_split external --out data/manifests/external_fresh
```

Use that path when running a job with `SPLIT_ROOT=data/manifests/external_fresh`. The default scripts use the packaged manifests under `data/manifests/external`. The external test cohort is used only for final evaluation after validation-based decisions. Image similarity checks do not verify patient-level independence because patient identifiers were unavailable.

## Transfer layout

```text
REAL/cbis/external/{images,masks}/
REAL/isic2017/external/{images,masks}/
REAL/isic2018/external/{images,masks}/
REAL/polyps/external/{images,masks}/
```

Do not replace existing training or validation folders. Keep preparation reports with your provenance records, but review them for sensitive source paths before sharing publicly.

## Preprocessing protocol

PH2: use each case's Dermoscopic_Image BMP and its `_lesion.bmp`, not colour ROI annotations. Convert images to RGB PNG without resizing; convert the lesion mask to binary 0/255 with grayscale >127. Export the same accepted cases to both ISIC tasks.

NeoPolyp: use eligible paired images and masks from the supplied training-labelled portion as external test-only cases. Challenge test images without masks cannot yield segmentation metrics. Merge red/green polyp annotations into foreground using the documented mask rule in the preparation script; preserve image dimensions.

DMID: use clean TIFF images and ROI masks, not coloured pixel-annotation overlays. The script crops each annotated mammogram around the union of nonzero ROI regions, with black padding as needed, then resizes images and masks to 512×512 using Lanczos and nearest-neighbour interpolation respectively. This is ground-truth-assisted ROI segmentation; it does not model whole-mammogram lesion detection. The loader resizes these inputs to the experiment's 256×256 grid.

Exact-image and perceptual-similarity screening is a precaution, not proof of patient-disjoint data. Review and retain the generated `reports/` with the local experiment provenance. Do not publish private paths or identifiers contained in source records.
