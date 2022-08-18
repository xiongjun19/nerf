FROM nvcr.io/nvidia/tensorflow:20.09-tf1-py3
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
    
RUN curl \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh -o "Miniconda3-py37_4.12.0-Linux-x86_64.sh" \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.12.0-Linux-x86_64.sh -b

COPY dock_env.yml .
RUN conda env create -f dock_env.yml

