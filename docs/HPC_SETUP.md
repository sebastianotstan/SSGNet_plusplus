# HPC environment example

Copy `scripts/hpc_env.example.sh` to `scripts/hpc_env.sh`. Load your site's CUDA module, activate the environment containing a compatible PyTorch build, set `PYTHON` to that environment's executable, and verify `torch.cuda.is_available()` inside a scheduled GPU allocation. The example PBS requests are placeholders based on the prior CX3 runs and must be checked against current local policy.

Install base Python dependencies with `requirements.txt` after installing the site-appropriate Torch and torchvision builds. JBF requires OpenCV contrib with `cv2.ximgproc`; DenseCRF configurations additionally require a compatible `pydensecrf` installation. Do not install a second OpenCV wheel that overwrites the contrib build.
