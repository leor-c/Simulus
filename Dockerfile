FROM nvidia/cuda:12.1.0-base-ubuntu22.04

WORKDIR /REM

RUN apt-get update && apt-get -y install gcc

RUN apt-get update && apt-get install -yq \
        bison \
        build-essential \
        cmake \
        curl \
        flex \
        git \
        libbz2-dev \
        ninja-build \
        wget \
        tmux

RUN apt-get install ffmpeg libsm6 libxext6 -y

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN mkdir -p ~/miniconda3 && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && rm -rf ~/miniconda3/miniconda.sh

RUN ~/miniconda3/bin/conda init bash && ~/miniconda3/bin/conda init zsh

RUN conda install -y python=3.10

RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY ./get_lpips.py .
RUN python get_lpips.py

# RUN pip install minihack

RUN pip install git+https://github.com/chernyadev/bigym

RUN conda install -y conda-forge::mesalib
RUN apt-get install -y libgl1-mesa-glx libosmesa6

RUN pip install craftax

RUN pip install git+https://github.com/leor-c/Kinetix-CPU.git

# RUN apt install msttcorefonts -qq

#RUN groupadd -r rem_users && useradd -r -g rem_users rem_user
#USER rem_user
#CMD ["python", "src/main.py"]
