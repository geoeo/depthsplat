FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

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
	libglib2.0-0 \
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

RUN conda run -n depthsplat pip install --no-cache-dir --upgrade pip \
	&& conda run -n depthsplat pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

COPY host_groups /


