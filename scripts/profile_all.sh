#!/bin/bash

set -x

for DS in deforming_plate cylinder_flow flag_simple ; do
    DATA_PATH=data/${DS}_valid_1024_infer
    [ ! -d "$DATA_PATH" ] && ./scripts/create_infer_data $DATA_PATH $DS valid 1024


    for B in 1 2 4 8 16 32 64 128 256 ; do
        # echo "==== Tensorflow Version ===="
        # pushd tensorflow
        # ./scripts/infer_bench.sh $DS /$DATA_PATH $B
        # popd

        echo "==== PyTorch Version ===="
        ./scripts/infer_profile $DS $DATA_PATH $B 10

    done
done
