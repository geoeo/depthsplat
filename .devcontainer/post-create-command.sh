#!/usr/bin/env sh

set -eu

CONDA_NUMBER_CHANNEL_NOTICES=0
CONDA_TOS_ACCEPTED=yes
export CONDA_NUMBER_CHANNEL_NOTICES CONDA_TOS_ACCEPTED

conda config --set remote_read_timeout_secs 120
# Install all Python requirements except diff-gaussian-rasterization first.
# The diff-gaussian package imports torch at build time, so install it
# separately without build isolation to reuse torch from the env.
conda run -n depthsplat sh -c "grep -v '^git+https://github.com/dcharatan/diff-gaussian-rasterization-modified$' requirements.txt > /tmp/requirements-no-dgrm.txt"
conda run -n depthsplat pip install -r /tmp/requirements-no-dgrm.txt
conda run -n depthsplat pip install --no-build-isolation git+https://github.com/dcharatan/diff-gaussian-rasterization-modified

# Guard the ABI. torch/torchvision/xformers are installed in the Dockerfile from
# the cu126 index because only those wheels are built with
# _GLIBCXX_USE_CXX11_ABI=1 (see .github/agents/pytorch-cxx11-abi.md). Several
# requirements above declare a torch dependency, so a future pin could quietly
# pull the pre-cxx11 PyPI wheel over it and break every C++ consumer and every
# AOTInductor package. Fail the build here rather than at link time.
conda run -n depthsplat python -c "\
import torch, sys; \
ok = torch.compiled_with_cxx11_abi(); \
print(f'torch {torch.__version__} cxx11_abi={ok}'); \
sys.exit(0 if ok and torch.version.cuda and torch.version.cuda.startswith('12.6') else 1)"

# Make conda activation available in future bash shells and avoid duplicate lines.
conda --no-plugins init bash
if ! grep -qxF "conda activate depthsplat" /home/ubuntu/.bashrc; then
	echo "conda activate depthsplat" >> /home/ubuntu/.bashrc
fi