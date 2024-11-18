#!/bin/bash
# $1: data folder path to mount in the /data path inside the container
# NOTE: home folder is always mounted inside the container
docker stop $USER-weakft
docker rm $USER-weakft
docker run -it -h weakft --name $USER-weakft \
    --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --memory=600g \
    --shm-size=2g \
    -u $(id -u):$(id -g) \
    -d \
    -v /home/$USER:/home/$USER \
    -v /home/$USER/data:/data \
    -w /home/$USER/Codes/WeakFT \
    $USER/weakft:v0.1.2
