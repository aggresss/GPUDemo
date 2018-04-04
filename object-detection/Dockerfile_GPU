#
# Dockerfile for tensorflow object dection API
#

# Use tensorflow image as parent image
FROM tensorflow/tensorflow:1.6.0-gpu

LABEL AUTHOR aggresss
ENV DEBIAN_FRONTEND noninteractive

EXPOSE 8888 
EXPOSE 6006
VOLUME /root/volume
USER root

# Modify apt-get to aliyun mirror
RUN sed -i 's/archive.ubuntu/mirrors.aliyun/g' /etc/apt/sources.list
RUN apt-get update


# Modify timezone to GTM+8
ENV TZ=Asia/Shanghai
RUN apt-get -y install tzdata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Modify locale
RUN apt-get -y install locales
RUN locale-gen en_US.UTF-8
RUN echo "LANG=\"en_US.UTF-8\"" > /etc/default/locale && \
    echo "LANGUAGE=\"en_US:en\"" >> /etc/default/locale && \
    echo "LC_ALL=\"en_US.UTF-8\"" >> /etc/default/locale

# Modify pip mirror
RUN mkdir -p /root/.pip
RUN echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url=http://mirrors.aliyun.com/pypi/simple/" >> /root/.pip/pip.conf && \
    echo "[install]" >> /root/.pip/pip.conf && \
    echo "trusted-host=mirrors.aliyun.com" >> /root/.pip/pip.conf

# Modify Jupter run arguments
RUN mkdir -p /root/.jupyter
RUN echo "# Jupyter config file" > /root/.jupyter/jupyter_config.py && \
    echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_config.py && \
    echo "c.NotebookApp.notebook_dir = u'/GPUDemo/object-detection/'" >> /root/.jupyter/jupyter_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_config.py && \
    echo "c.NotebookApp.allow_root = True">> /root/.jupyter/jupyter_config.py && \
    echo "c.NotebookApp.port = 8888">> /root/.jupyter/jupyter_config.py && \
    echo "# default password: 12345678">> /root/.jupyter/jupyter_config.py && \
    echo "c.NotebookApp.password = u'sha1:d501736a80f9:2bf882737f5ded39b8d1803b0c3ca385325fbfa8'" >> \
    /root/.jupyter/jupyter_config.py

# Install necessary library
RUN apt-get -y install apt-utils python python-dev python-pip \
    lib32z1 libglib2.0-dev libsm6 libxrender1 \
    libxext6 libice6 libxt6 libfontconfig1 libcups2 \
    protobuf-compiler python-tk git vim \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install necessary python-library
RUN pip install --upgrade pip
RUN pip  --no-cache-dir install opencv-python Cython lxml flask

# Clone the repository
RUN git clone https://github.com/aggresss/GPUDemo.git /GPUDemo
WORKDIR /GPUDemo/object-detection/
RUN protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH /usr/bin/python:/GPUDemo/object-detection:/GPUDemo/object-detection/slim

WORKDIR /
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR /cocoapi/PythonAPI
RUN make
RUN cp -r pycocotools /GPUDemo/object-detection/

# Make startup run file
RUN echo '#!/bin/sh' > /run.sh && \
    echo "nohup jupyter notebook" >> /run.sh
RUN chmod +x /run.sh
WORKDIR /GPUDemo/object-detection/
CMD /run.sh


