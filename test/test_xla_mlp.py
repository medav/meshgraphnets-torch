import tensorflow as tf
from tensorflow import keras
import keras.layers

import time


@tf.function(jit_compile=True)
def ffn(x, w1, b1):
    y1 = tf.linalg.matmul(x, w1) + b1
    #y2 = tf.linalg.matmul(y1, w2) + b2
    #y3 = tf.linalg.matmul(y2, w3) + b3
    return y1


if __name__ == '__main__':
    num_iters = 100000

    w1 = tf.random.normal((128, 128), dtype=tf.float16)
    b1 = tf.random.normal((128,), dtype=tf.float16)
    w2 = tf.random.normal((128, 128), dtype=tf.float16)
    b2 = tf.random.normal((128,), dtype=tf.float16)
    w3 = tf.random.normal((128, 128), dtype=tf.float16)
    b3 = tf.random.normal((128,), dtype=tf.float16)

    x = tf.random.uniform((128*1024, 128), dtype=tf.float16)
    t0 = time.perf_counter()
    for _ in range(num_iters):
        y = ffn(x, w1, b1)

    t1 = time.perf_counter()

    flops = 128*1024 * (128*384) * 2

    print(f"Time: {t1-t0}")
    print(f'Avg. Latency: {(t1-t0)/num_iters}')
    print(f'GFLOP/s: {flops * num_iters / (t1-t0) / 1e9}')


