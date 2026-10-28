# StyleGAN3 usage notes

This folder documents the StyleGAN3 component used to create synthetic image pools. It contains no NVIDIA source code, pretrained weights, medical images, or generated samples. Obtain the official implementation and follow its separate licence and dependency instructions:

- [Official repository and setup](https://github.com/NVlabs/stylegan3)
- [Training options](https://github.com/NVlabs/stylegan3/blob/main/docs/train-help.txt)
- [Configurations and training-time guidance](https://github.com/NVlabs/stylegan3/blob/main/docs/configs.md)
- [Dataset preparation tool](https://github.com/NVlabs/stylegan3/blob/main/dataset_tool.py)
- [NVIDIA StyleGAN3 source-code licence](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt)
- Paper: Karras et al., “Alias-Free Generative Adversarial Networks,” NeurIPS 2021.

## Project data layout

Keep StyleGAN3 inputs and outputs separate from the segmentation pipeline's validation and test data. For class-specific generators, prepare one image-only folder per dataset and class:

```text
stylegan3/
  data/
    source/<dataset>/<class>/*.png
    archives/<dataset>_<class>_<resolution>.zip
  runs/<dataset>/<class>/<run-id>/
  generated/<dataset>/<class>/
```

Populate each source folder using **only real images from the original training partition**. Do not include validation or test images, segmentation masks, pseudo-masks, or generated images. Keep one model per class, or document the labels if you instead train a conditional model. Record class mapping, image count, resolution, preprocessing, split provenance, seed, complete command, and training duration.

StyleGAN3 can read an image directory directly; NVIDIA recommends a ZIP of uncompressed PNG images with `dataset.json`. From the official StyleGAN3 repository root, set paths and prepare the dataset, for example:

```bash
DATASET=cbis
CLASS=benign
RESOLUTION=512x512
STYLEGAN3_ROOT=/path/to/NVlabs/stylegan3
SOURCE_DIR="/path/to/stylegan3/data/source/$DATASET/$CLASS"
ARCHIVE="/path/to/stylegan3/data/archives/${DATASET}_${CLASS}_${RESOLUTION}.zip"

cd "$STYLEGAN3_ROOT"
python dataset_tool.py \
  --source="$SOURCE_DIR" \
  --dest="$ARCHIVE" \
  --resolution="$RESOLUTION"
```

The dataset/class and 512×512 resolution above are examples; use the class mapping and resolution chosen for your actual image pool. Inputs should be consistently sized RGB images. Masks are not supplied to the GAN. After generating images, place the selected samples in `SYNTH/<dataset>/train/images/`, which is the layout expected by the SSGNet++ segmentation pipeline, and preserve their model/class provenance separately.

## Training command

The upstream command requires an output directory, configuration, dataset, GPU count, total batch size, and R1 regularisation weight (`gamma`). This template intentionally leaves the hardware and dataset-specific settings to be filled in:

```bash
GPUS=SET_GPU_COUNT
TOTAL_BATCH=SET_TOTAL_BATCH
GAMMA=SET_DATASET_SPECIFIC_GAMMA
KIMG=SET_TRAINING_DURATION
SEED=42
RUN_DIR="/path/to/stylegan3/runs/$DATASET/$CLASS"

python train.py \
  --outdir="$RUN_DIR" \
  --cfg=stylegan3-t \
  --data="$ARCHIVE" \
  --gpus="$GPUS" \
  --batch="$TOTAL_BATCH" \
  --gamma="$GAMMA" \
  --kimg="$KIMG" \
  --seed="$SEED" \
  --metrics=none
```

This is a command template, not a validated medical-image configuration. Choose resolution, configuration, batch, `gamma`, augmentation, mirror-flipping, and GPU count for the data and hardware; consult `python train.py --help` and the upstream configuration guide. For example, upstream `gamma=8.2` is for AFHQv2 at 512×512 and should not be reused as a medical-image setting without validation. The upstream default duration is 25,000 kimg. Its guide notes that around 5,000 kimg can look reasonable in typical cases, but that is not a convergence guarantee or a medical-image quality threshold. Save the full command and logs for every run.

`--metrics=none` disables periodic FID evaluation. With default FID enabled, NVIDIA reports that metric computation typically adds up to about 5% to training time. ImageNet-feature FID or IS are supplementary diagnostics; they do not establish clinical realism or downstream segmentation utility.

## Runtime planning

There is no universal wall-time target. Runtime depends on resolution, configuration, GPU type and count, batch size, data loading, and periodic metrics. NVIDIA's guide gives a StyleGAN3-T example for AFHQv2 at 512×512: about **18.47 seconds/kimg on 8 V100 GPUs** or **12.29 seconds/kimg on 8 A100 GPUs**. At those example rates, 25,000 kimg is about 5.4 days on the 8×V100 setup or 3.6 days on the 8×A100 setup before extra overhead; 5,000 kimg is about 25.7 or 17.1 hours, respectively. These are upstream reference figures, not estimates for these medical datasets, an L40S, or a single GPU.

For a practical estimate, run a short pilot at the intended resolution and batch settings, read seconds per kimg from the training log, then calculate:

```text
estimated hours = observed seconds per kimg × target kimg ÷ 3600
```

Add time for data preparation, checkpoint writes, restarts, and metrics. A smoke run only checks that training starts; it does not show that the training budget is sufficient for image quality. For exact reproduction of a paper image pool, use its recorded StyleGAN3 settings and duration if available. Do not present a new duration as the historical run setting.

## Licensing and attribution

NVIDIA distributes StyleGAN3 under its own source-code licence, which limits use of the upstream work and derivatives to non-commercial research or evaluation. A licence selected for SSGNet++ does not replace NVIDIA's terms. If you include official StyleGAN3 code in a repository, retain its licence and notices and clearly separate third-party code from SSGNet++ code. This folder only links to the upstream source; check the current terms before cloning, modifying, or redistributing it.

Cite the StyleGAN3 paper and official implementation. Dataset licences apply separately to source images and generated outputs.
