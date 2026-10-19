FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive

ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
	locales \
	curl \
	ca-certificates \
	bzip2 \
	libgl1 \
	libglib2.0-0t64 \
	libsm6 \
	libxext6 \
	libxrender1 \
	&& locale-gen en_US.UTF-8 \
	&& update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
	&& rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
	&& bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
	&& rm -f /tmp/miniconda.sh \
	&& conda config --system --set auto_update_conda false \
	&& conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
	&& conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r \
	&& conda install -y -n base pip \
	&& conda clean -afy

RUN conda create -y -n depthsplat python=3.10 \
	&& conda clean -afy

# torch/torchvision come from the cu132 index, never PyPI. PyPI's torch==2.14.1
# is the CUDA 13.0 build under the same version string, so --index-url, not
# --extra-index-url, or pip would be free to pick it. (Every torch >= 2.7 wheel
# is cxx11 ABI, so the ABI concern in .github/agents/pytorch-cxx11-abi.md is
# moot; the CUDA build is what has to be pinned now.)
#
# xformers has no cu132 build. 0.0.35 is taken from the cu130 index, with
# --no-deps so its torch>=2.10 requirement cannot pull in a different torch. It
# runs on cu132 torch; see .github/agents/migration.md.
RUN conda run -n depthsplat pip install --no-cache-dir --upgrade pip \
	&& conda run -n depthsplat pip install --no-cache-dir \
		torch==2.14.1 torchvision==0.29.1 \
		--index-url https://download.pytorch.org/whl/cu132 \
	&& conda run -n depthsplat pip install --no-cache-dir --no-deps \
		xformers==0.0.35 \
		--index-url https://download.pytorch.org/whl/cu130

COPY host_groups /


