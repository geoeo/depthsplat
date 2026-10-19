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
# Its rasterizer_impl.h uses std::uintptr_t/uint32_t without <cstdint>. GCC 11
# (Ubuntu 22.04) included that transitively; GCC 13 (Ubuntu 24.04) does not, so
# force-include it rather than fork the repo. See .github/agents/migration.md.
conda run -n depthsplat env NVCC_PREPEND_FLAGS="--pre-include cstdint" \
	pip install --no-build-isolation git+https://github.com/dcharatan/diff-gaussian-rasterization-modified

# Guard the torch build. torch/torchvision are installed in the Dockerfile from
# the cu132 index. Several requirements above declare a torch dependency, so a
# future pin could quietly pull PyPI's torch over it -- the same version string
# but built for CUDA 13.0 -- and every AOTInductor package and C++ consumer is
# locked to the build that produced it. Fail the build here rather than later.
# (compiled_with_cxx11_abi() is no longer checked: since torch 2.7 every wheel is
# cxx11 and the function just returns True.)
conda run -n depthsplat python -c "\
import torch, sys; \
print(f'torch {torch.__version__} cuda={torch.version.cuda}'); \
sys.exit(0 if torch.version.cuda == '13.2' and torch.__version__.endswith('+cu132') else 1)"

# Make conda activation available in future bash shells and avoid duplicate lines.
conda --no-plugins init bash
if ! grep -qxF "conda activate depthsplat" /home/ubuntu/.bashrc; then
	echo "conda activate depthsplat" >> /home/ubuntu/.bashrc
fi