FROM nvidia/cuda:10.1-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /quinv
WORKDIR /quinv

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /quinv
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda create -y --name py36 python=3.6.9 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN /home/user/miniconda/bin/conda install conda-build=3.18.9=py36_3 \
 && /home/user/miniconda/bin/conda clean -ya


# CUDA 10.1-specific steps
RUN conda install -y -c pytorch \
   cudatoolkit=10.1 \
   "pytorch=1.4.0=py3.6_cuda10.1.243_cudnn7.6.3_0" \
   "torchvision=0.5.0=py36_cu101" \
&& conda clean -ya

# Install Libraries with Pip
RUN pip install \
    python-socketio \
    eventlet \
    backcall==0.1.0 \
    colorama==0.4.1 \
    cycler==0.10.0 \
    decorator==4.4.0 \
    fire==0.1.3 \
    got10k==0.1.3 \
    imutils==0.5.3 \
    ipdb==0.12 \
    ipython==7.5.0 \
    ipython-genutils==0.2.0 \
    jedi==0.13.3 \
    kiwisolver==1.1.0 \
    llvmlite==0.29.0 \
    matplotlib==3.1.0 \
    numba==0.44.0 \
    opencv-python==4.1.1.26 \
    pandas==0.24.2 \
    parso==0.4.0 \
    pexpect==4.7.0 \
    pickleshare==0.7.5 \
    prompt-toolkit==2.0.9 \
    ptyprocess==0.6.0 \
    pygments==2.4.2 \
    pyparsing==2.4.0 \
    python-dateutil==2.8.0 \
    pytz==2019.1 \
    scipy==1.1.0 \
    shapely==1.6.4.post2 \
    tqdm==4.32.1 \
    traitlets==4.3.2 \
    wcwidth==0.1.7 \
    wget==3.2 \
    yacs==0.1.6 \
    cupy-cuda101==6.5.0 \
    Cython==0.29.14 \
    easydict==1.9 \
    Flask==1.1.1 \
    Flask-Cors==3.0.8 \
    h5py==2.10.0 \
    Pillow==6.2.1 \
    scikit-image==0.16.2 \
    requests==2.22.0 \
    six==1.13.0 \
    tensorboard==1.14.0 \
    tensorboardX==1.9 \
    urllib3==1.25.6 \
    visdom==0.1.8.9 


ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH='/src/:$PYTHONPATH'

# Add general lib
RUN sudo apt-get update 
RUN sudo apt-get install -y libsm6 
RUN sudo apt-get update
RUN sudo apt install build-essential -y
RUN sudo apt-get install nvidia-cuda-toolkit -y

# Clone
RUN git clone https://github.com/nguyenquibk1996/FasterRCNN.git

# Add project
ADD models FasterRCNN/models
RUN cd FasterRCNN && mkdir data


# Activate run file
RUN sudo chmod +x /quinv/FasterRCNN/run.sh

# Install Requirement
RUN pip install -r /quinv/FasterRCNN/requirements.txt
RUN pip install torchvision==0.4.0

# Build C++ lib
WORKDIR /quinv/FasterRCNN/lib
RUN python setup.py build develop

# Build COCOAPI
WORKDIR /quinv
RUN git clone https://github.com/pdollar/coco.git
RUN sudo mv ./coco ./FasterRCNN/data/
RUN sudo chmod -R 777 ./FasterRCNN/data/coco/PythonAPI 
WORKDIR /quinv/FasterRCNN/data/coco/PythonAPI
RUN python setup.py install
RUN python setup.py build_ext --inplace

# Auto run
WORKDIR /quinv/FasterRCNN
CMD ["./run.sh"]

