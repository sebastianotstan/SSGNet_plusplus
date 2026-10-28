# Copy to scripts/hpc_env.sh and edit for your cluster. Do not commit that file.
# Load your site CUDA/module and activate the environment before setting PYTHON.
# module load <CUDA_MODULE>
# source <CONDA_BASE>/etc/profile.d/conda.sh
# conda activate <ENVIRONMENT_NAME>
export PYTHON="/path/to/your/environment/bin/python"
export DEVICE="cuda:0"
export WORKERS=4
# export TORCH_HOME="/path/to/cluster/torch-cache"
"$PYTHON" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
