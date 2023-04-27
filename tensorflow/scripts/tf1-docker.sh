#!/bin/bash


[ -x "$MGN_DATA" ] || MGN_DATA=$(realpath ../data)

docker build -t tf1-docker:latest -f Dockerfile .
docker run --gpus all -it --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -v $MGN_DATA:/data -v $PWD:/work -w /work tf1-docker:latest $*


