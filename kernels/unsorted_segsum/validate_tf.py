import tensorflow as tf
import torch
from . import kernel
import numpy as np

def torch_unsorted_segment_sum(
    data : np.ndarray,
    indices : np.ndarray,
    num_segments : int
):
    return kernel.unsorted_segment_sum_ref(
        torch.Tensor(data),
        torch.LongTensor(indices),
        num_segments).numpy()

def tf_unsorted_segment_sum(
    data : np.ndarray,
    indices : np.ndarray,
    num_segments : int
):
    return tf.math.unsorted_segment_sum(data, indices, num_segments)


if __name__ == '__main__':
    data = np.random.randn(1000, 128)
    indices = np.random.randint(0, 100, (1000,))
    num_segments = 100

    ref = torch_unsorted_segment_sum(data, indices, num_segments)
    out = tf_unsorted_segment_sum(data, indices, num_segments)

    print(ref)
    print(out)

    print('L2 = ', np.linalg.norm(ref - out))


