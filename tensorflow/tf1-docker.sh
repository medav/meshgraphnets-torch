#!/bin/bash

pushd tensorflow
docker build -t tf1-docker:latest -f Dockerfile .
popd

docker run --gpus all -it -v $PWD:/work -w /work tf1-docker:latest bash


