#!/bin/bash

[ -x "$MGN_DATA" ] || MGN_DATA=$(realpath ../data)

# dataset_name = sys.argv[1]
# datapath = sys.argv[2]
# checkpoint_dir = sys.argv[3]
# out_dir = sys.argv[4]

./scripts/tf1-docker.sh python np_validate.py $1 /$2 /$3 /$4
