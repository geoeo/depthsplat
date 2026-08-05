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

# Make conda activation available in future bash shells and avoid duplicate lines.
conda --no-plugins init bash
if ! grep -qxF "conda activate depthsplat" /home/ubuntu/.bashrc; then
	echo "conda activate depthsplat" >> /home/ubuntu/.bashrc
fi