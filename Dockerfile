# Use a base image with Miniconda, or install Miniconda manually if necessary
FROM continuumio/miniconda3

# Set ARGs for configuration
ARG ENV_NAME=sam2
ARG ENV_FILE=environment.yml

# Copy environment.yml file to the Docker image
COPY $ENV_FILE /tmp/environment.yml

# Install dependencies, create and activate the Conda environment
RUN conda update -n base -c defaults conda && \
    conda env create -f /tmp/environment.yml && \
    conda clean -afy
RUN conda run -n $ENV_NAME pip install git+https://github.com/kornia/kornia
RUN conda run -n $ENV_NAME pip install jupyter
RUN conda run -n $ENV_NAME pip install debugpy

EXPOSE 7777

# Set the environment to activate automatically
ENV PATH /opt/conda/envs/$ENV_NAME/bin:$PATH


ARG REPO_DIR=./
ARG USERID=1000
ARG GROUPID=1000
ARG USERNAME=containeruser
ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Berlin
# if using ray for experiments, set the legacy output style (less verbose)
ENV RAY_AIR_NEW_OUTPUT=0
ENV REPO_DIR=${REPO_DIR}

RUN apt-get update -y && apt-get install -y git-flow sudo libffi-dev build-essential ffmpeg libsm6 libxext6
RUN groupadd -g $GROUPID weakFT
RUN useradd -ms /bin/bash -u $USERID -g $GROUPID $USERNAME

# Make sudo easy to use inside the container
RUN echo "$USERNAME ALL=(ALL:ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/$USERNAME

COPY entrypoint.sh /opt/app/entrypoint.sh

ENTRYPOINT [ "bash", "/opt/app/entrypoint.sh" ]
CMD [ "bash" ]



# FROM nvcr.io/nvidia/pytorch:23.10-py3

# # REPO_DIR contains the path of the repo inside the container, which is 
# # expected to be mounted inside and not copied (this is a dev env)
# ARG REPO_DIR=./
# ARG USERID=1000
# ARG GROUPID=1000
# ARG USERNAME=containeruser
# ARG DEBIAN_FRONTEND=noninteractive

# ENV TZ=Europe/Berlin
# # if using ray for experiments, set the legacy output style (less verbose)
# ENV RAY_AIR_NEW_OUTPUT=0
# ENV REPO_DIR=${REPO_DIR}

# RUN apt-get update -y && apt-get install -y git-flow sudo libffi-dev build-essential ffmpeg libsm6 libxext6
# RUN groupadd -g $GROUPID weakFTgroup
# RUN useradd -ms /bin/bash -u $USERID -g $GROUPID $USERNAME

# # Make sudo easy to use inside the container
# RUN echo "$USERNAME ALL=(ALL:ALL) NOPASSWD: ALL" | tee /etc/sudoers.d/$USERNAME

# RUN pip install --upgrade pip
# COPY requirements.txt /opt/app/requirements.txt
# RUN pip install -r /opt/app/requirements.txt
# # RUN pip install git+https://github.com/kornia/kornia

# COPY entrypoint.sh /opt/app/entrypoint.sh

# ENTRYPOINT [ "bash", "/opt/app/entrypoint.sh" ]
# CMD [ "bash" ]

