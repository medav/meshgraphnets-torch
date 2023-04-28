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


def benchmark(model, datapath, float_type=tf.float32):
    ds = dataset.load_dataset(datapath, float_type=float_type)
    ds = ds.repeat(None)

    inputs = tf.data.make_one_shot_iterator(ds).get_next()
    loss_op = model.loss(inputs)

    num_samples = len([
        name for name in os.listdir(datapath)
        if os.path.isfile(os.path.join(datapath, name))
    ])
    num_iters = 5

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('running...')
        t0 = time.perf_counter()
        for i in range(num_iters * num_samples): sess.run([loss_op])
        t1 = time.perf_counter()
        print('done')

    print(f'Batch Size: 1')
    print(f'Num Samples: {num_samples}')
    print(f'Num Iters: {num_iters}')
    print(f'Elapsed time: {t1 - t0:.2f} seconds')
    print(f'Throughput: {num_iters * num_samples / (t1 - t0):.2f} samp/sec')



if __name__ == '__main__':
    dataset_name = sys.argv[1]
    datapath = sys.argv[2]
    tf.enable_resource_variables()
    tf.disable_eager_execution()

    model = {
        'flag_simple': 'cloth',
        'cylinder_flow': 'cfd',
        'deforming_plate': 'deforming_plate'
    }[dataset_name]

    float_type=tf.float16

    params = PARAMETERS[model]
    learned_model = core_model.EncodeProcessDecode(
        output_size=params['size'],
        latent_size=128,
        num_layers=2,
        message_passing_steps=15)
    model = params['model'].Model(learned_model, float_type=float_type)

    benchmark(model, datapath, float_type=float_type)

