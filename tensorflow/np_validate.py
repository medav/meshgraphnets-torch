import sys
import os
import numpy as np
import tensorflow.compat.v1 as tf
import cfd_model
# import cloth_model
import core_model
import np_dataset as dataset
import time

from parameters import *


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(f'Usage: {sys.argv[0]} <dataset> <datapath> <checkpoint_dir> <out_dir>')
        exit(1)

    dataset_name = sys.argv[1]
    datapath = sys.argv[2]
    checkpoint_dir = sys.argv[3]
    out_dir = sys.argv[4]

    os.makedirs(out_dir, exist_ok=True)

    tf.enable_resource_variables()
    tf.disable_eager_execution()

    model = {
        'flag_simple': 'cloth',
        'cylinder_flow': 'cfd',
        'deforming_plate': 'deforming_plate'
    }[dataset_name]

    float_type=tf.float32

    params = PARAMETERS[model]
    learned_model = core_model.EncodeProcessDecode(
        output_size=params['size'],
        latent_size=128,
        num_layers=2,
        message_passing_steps=15)
    model = params['model'].Model(learned_model, float_type=float_type)

    num_samples = len([
        name for name in os.listdir(datapath)
        if os.path.isfile(os.path.join(datapath, name))
    ])

    num_iters = 1

    ds = dataset.load_dataset(datapath, float_type=float_type)
    ds = ds.repeat(None)
    ds = ds.prefetch(4)

    inputs = tf.data.make_one_shot_iterator(ds).get_next()

    graph_op = model._build_graph(inputs, is_training=False)
    pred_op = model._learned_model(graph_op)
    tf.train.create_global_step()

    with tf.train.MonitoredTrainingSession(
        hooks=[],
        checkpoint_dir=checkpoint_dir,
        save_checkpoint_secs=180) as sess:

        for i in range(num_samples):
            [y] = sess.run([pred_op])
            print(y.shape)



