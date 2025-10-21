#!/bin/bash
# $1: data folder path to mount in the /data path inside the container
# NOTE: home folder is always mounted inside the container

cid=1
img_name=$USER/weakft:v0.3.$cid
container_name=$USER-weakft-$cid
docker stop $container_name
docker rm $container_name

docker image rm $img_name
docker build \
    --build-arg="USERNAME=$USER" \
    -t $img_name .

docker run -it -h weakft --name $container_name \
    --gpus all \
    --memory=600g \
    --shm-size=4g \
    -d \
    -p $((7776+cid)):$((7776+cid)) \
    -v .:/home/$USER \
    -v /home/$USER/data:/data \
    -w /home/$USER \
    $img_name 

# docker exec -it $container_name bash -c 'conda init && conda activate sam2 && pip install -e .'
# after that, run the python script using 'docker exec -it $container_name bash test.sh'
