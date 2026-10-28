# Environment setup

Use Python 3.10 or later. Install PyTorch and torchvision first, choosing builds compatible with your local CUDA runtime or HPC modules. Then install the remaining pinned/minimum dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

JBF configurations require `opencv-contrib-python-headless` with `cv2.ximgproc.jointBilateralFilter`; DenseCRF configurations additionally require `pydensecrf`. Avoid installing conflicting OpenCV wheels into one environment. Verify the stack before running:

```bash
python -c "import cv2; print(cv2.__version__, hasattr(cv2, 'ximgproc'))"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Standard configurations request ImageNet-pretrained encoders. Provide/cache the weights if compute nodes do not have network access. The smoke configuration uses random weights and does not verify pretrained-weight availability.

For HPC, copy `scripts/hpc_env.example.sh` to the ignored `scripts/hpc_env.sh`; configure the site module, Conda activation, Python path, GPU device, and worker count. Review `scripts/run_hpc.pbs` resource directives for current cluster policy. Test GPU visibility and package availability inside a scheduled GPU job before submitting a large matrix. Environment versions, CUDA and hardware can affect numerical results even with the same random seed.
