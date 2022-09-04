# syntax=docker/dockerfile:experimental

# OS
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04


# debians
RUN apt-get -y update \
	&& apt-get install -y curl software-properties-common \
  	&& apt-get -y update

# mrg apt packages
RUN apt-get -y update
RUN apt-get install -y ruby
RUN /usr/bin/env gem install aharrison24-git-external --no-rdoc --no-ri --force
RUN apt-get install -y build-essential
RUN apt-get install -y git tmux wget

# FFMPEG
RUN apt-get install -y ffmpeg

# python, prev. 3.6, deadsnakes/ppa
RUN  add-apt-repository ppa:jblgf0/python \
	&& apt-get -y update \
	&& apt-get -y install python3.6 \
	&& wget https://bootstrap.pypa.io/pip/3.6/get-pip.py \
        && wget https://bootstrap.pypa.io/pip/3.6/get-pip.py \
	&& python3.6 get-pip.py \
	&& rm /usr/local/bin/pip3 \
	&& ln -s /usr/bin/python3.6 /usr/local/bin/python3 \
	&& ln -s /usr/local/bin/pip /usr/local/bin/pip3 \
	&& apt-get install -y python3.6-dev
#3.6
# ssh
RUN mkdir -p -m 0600 /root/.ssh \
	&&  ssh-keyscan -p 7999 mrgbuild.robots.ox.ac.uk >> /root/.ssh/known_hosts

# compiler
ENV CC /usr/bin/clang
ENV CXX /usr/bin/clang++

# alias
RUN echo 'alias python=python3' >> /root/.bashrc \
  && echo 'alias pip=pip3' >> /root/.bashrc

# mcl
RUN mkdir buildsystem
#\
#	&& cd buildsystem \
#	&& cmake ../../src/buildsystem \
#	&& make -j
RUN mkdir datatypes
#\
#	&& cd datatypes \
#	&& cmake ../../src/datatypes \
#	&& make -j
RUN mkdir core-moos
#\
#	&& cd core-moos \
#	&& cmake ../../src/core-moos \
#	&& make -j
RUN mkdir sms
#\
#	&& cd sms \
#	&& cmake ../../src/sms \
#	&& make -j
RUN mkdir ticsync
#\
#	&& cd ticsync \
#	&& cmake ../../src/ticsync \
#	&& make -j
RUN mkdir platform-configurations
#\
#	&& cd platform-configurations \
#	&& cmake ../../src/platform-configurations \
#	&& make -j
RUN mkdir base-cpp
#\
#	&& cd base-cpp \
#	&& cmake ../../src/base-cpp \
#	&& make -j
RUN mkdir database-client
#\
#	&& cd database-client \
#	&& cmake ../../src/database-client \
#	&& make -j
RUN mkdir tools-cpp
#\
#	&& cd tools-cpp \
#	&& cmake ../../src/tools-cpp \
#	&& make -j

RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64


WORKDIR /root/Workspace/causal_natural_language_explanations/
COPY . .

RUN pip install --no-cache-dir -r requirements.txt



# labels
LABEL maintainer="Marc Alexander Kühn"
LABEL contact="marcalexander.kuehn@tum.de"
LABEL description="causal_natural_language_explanations"
