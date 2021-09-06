# ml_zoo
Paper summary and code implementation

## Environments
- Python 3.8.8 with Anaconda (Check `environment.yaml` to see packages I used!)
- GPU(NVIDIA Geforce GTX 1070 Ti) + CUDA 10.2, CUDA Toolkit 10.2
- For reproducibility, set environment variables below:
    - `CUDA_LAUNCH_BLOCKING=1`
    - `CUBLAS_WORKSPACE_CONFIG=:16:8` or `CUBLAS_WORKSPACE_CONFIG=:4096:2`
