FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt update && \
    apt upgrade -y && \
    apt install -y wget git

RUN wget https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh -O anaconda.sh -q && \
	chmod +x anaconda.sh && \
	./anaconda.sh -b -p /usr/local/anaconda && \
	rm anaconda.sh
ENV PATH /usr/local/anaconda/bin:$PATH
RUN conda update conda conda-build

RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

RUN pip install astor nltk records

RUN mkdir working_dir
WORKDIR /working_dir
