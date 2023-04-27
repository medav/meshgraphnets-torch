#!/bin/bash

MODEL=cfd
ODIR=/data/checkpoints/cfd_cylinder_flow
DATA=/data/cylinder_flow

mkdir -p $ODIR
chmod -r 777 $ODIR

./scripts/tf1-docker.sh python train.py --model=$MODEL --checkpoint_dir=$ODIR --dataset_dir=$DATA
