#!/bin/bash

[ -x "$MGN_DATA" ] || MGN_DATA=$(realpath ../data)
MODEL=cfd
ODIR=checkpoints/cfd_cylinder_flow
DATA=cylinder_flow

mkdir -p $MGN_DATA/$ODIR
chmod -R 777 $MGN_DATA/$ODIR

./scripts/tf1-docker.sh python train.py --model=$MODEL --checkpoint_dir=/data/$ODIR --dataset_dir=/data/$DATA
