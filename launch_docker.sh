#!/bin/bash
# $1: data folder path to mount in the /data path inside the container
# NOTE: home folder is always mounted inside the container

cid=1
img_name=$USER/landseg:v1.0.$cid
container_name=$USER-landseg-$cid
docker stop $container_name
docker rm $container_name

docker image rm $img_name
docker build \
    --build-arg="USERNAME=$USER" \
    -t $img_name .

docker run -it -h landseg --name $container_name \
    --gpus all \
    --memory=600g \
    --shm-size=4g \
    -d \
    -p $((7777+cid)):$((7777+cid)) \
    -v .:/home/$USER \
    -v $1:/data \
    -w /home/$USER \
    $img_name 
