# FROM nvidia/cuda:9.2-devel-ubuntu16.04
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04


ARG solaris_branch='master'
   
# prep apt-get and cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils  && \
    rm -rf /var/lib/apt/lists/*

# install requirements
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    emacs \
    git \
    less \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
    libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools \
    tree \
    unzip \
    vim \
    wget \
    xterm \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH


# install anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc


# use conda-forge instead of default channel
RUN conda update conda && \
    conda config --add channels conda-forge


RUN pip install --upgrade pip
RUN pip install torch==1.4.0 -f https://download.pytorch.org/whl/torch_stable
RUN pip install torchvision==0.5.0 --no-deps
RUN pip install albumentations==0.4.1
RUN pip install numba
RUN pip install geopandas
RUN pip install affine
RUN pip install rasterio
    
#RUN pip uninstall apex
RUN git clone https://github.com/NVIDIA/apex
RUN sed -i 's/check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/pass/g' apex/setup.py
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  /apex
RUN conda install pandas

RUN pip install efficientnet_pytorch
RUN pip install segmentation-models-pytorch
RUN pip install Pillow==6.1

WORKDIR /root/
RUN git clone https://github.com/cosmiq/solaris.git && \
    cd solaris && \
    git checkout tags/0.2.1 && \
    conda env create -f environment-gpu.yml
ENV PATH /opt/conda/envs/solaris/bin:$PATH

RUN source activate solaris && pip install git+git://github.com/toblerity/shapely.git
RUN cd solaris && pip install .

# INSERT COPY COMMANDS HERE TO COPY FILES TO THE WORKING DIRECTORY.
COPY . /root/

# SET PERMISSIONS FOR EXECUTION OF SHELL SCRIPTS
RUN chmod a+x /root/train.sh && chmod a+x /root/test.sh && chmod a+x /root/settings.sh && chmod a+x /root/train_models.sh && chmod a+x /root/test_models.sh
ENV PATH $PATH:/root/
