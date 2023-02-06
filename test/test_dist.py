import tensorflow as tf
import torch
import enum
import numpy as np


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def torch_arange2d(m, n):
    return torch.stack([
        torch.arange(m).reshape(-1, 1).repeat(1, n),
        torch.arange(n).reshape(1, -1).repeat(m, 1)
    ], dim=2)

def torch_squared_dist(A : torch.Tensor, B : torch.Tensor):
    row_norms_A = A.pow(2).sum(dim=1).reshape(-1, 1) # N, 1
    row_norms_B = B.pow(2).sum(dim=1).reshape(1, -1) # 1, N
    return row_norms_A - 2 * (A @ B.t()) + row_norms_B

def torch_construct_world_edges(
    world_pos : torch.Tensor,
    node_type : torch.Tensor,
    thresh : float = 0.5
) -> torch.Tensor:

    deformable = node_type != NodeType.OBSTACLE
    deformable_idx = torch.arange(node_type.shape[0])[deformable]

    actuator = node_type == NodeType.OBSTACLE
    actuator_idx = torch.arange(node_type.shape[0])[actuator]

    actuator_pos = world_pos[actuator].to(torch.float64) # M, D
    deformable_pos = world_pos[deformable].to(torch.float64) # N, D

    dists = torch_squared_dist(actuator_pos, deformable_pos) # M, N
    M, N = dists.shape

    idxs = torch_arange2d(M, N)
    rel_close_pair_idx = idxs[dists < (thresh ** 2)]

    srcs = actuator_idx[rel_close_pair_idx[:,0]]
    dsts = deformable_idx[rel_close_pair_idx[:,1]]

    return torch.stack([srcs, dsts], dim=0), torch.stack([dsts, srcs], dim=0)


def tf_construct_world_edges(world_pos, node_type):

    deformable_idx = tf.where(tf.not_equal(node_type, NodeType.OBSTACLE))
    actuator_idx = tf.where(tf.equal(node_type, NodeType.OBSTACLE))
    B = tf.squeeze(tf.gather(world_pos, deformable_idx))
    A = tf.squeeze(tf.gather(world_pos, actuator_idx))

    A = tf.cast(A, tf.float64)
    B = tf.cast(B, tf.float64)

    thresh = 0.5

    dists = tf_squared_dist(A, B)
    rel_close_pair_idx = tf.where(tf.math.less(dists, thresh ** 2))

    close_pair_actuator = tf.gather(actuator_idx, rel_close_pair_idx[:,0])
    close_pair_def = tf.gather(deformable_idx, rel_close_pair_idx[:,1])
    close_pair_idx = tf.concat([close_pair_actuator, close_pair_def], 1)

    senders, receivers = tf.unstack(close_pair_idx, 2, axis=1)

    return (tf.concat([senders, receivers], axis=0),
            tf.concat([receivers, senders], axis=0))

def tf_squared_dist(A, B):

  row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
  row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.

  row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
  row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.

  return row_norms_A - 2 * tf.matmul(A, B, False, True) + row_norms_B

world_pos_np = np.random.rand(100, 3)
node_type_np = np.random.randint(0, 2, (100,))

world_pos_torch = torch.from_numpy(world_pos_np)
node_type_torch = torch.from_numpy(node_type_np)

world_pos_tf = tf.convert_to_tensor(world_pos_np)
node_type_tf = tf.convert_to_tensor(node_type_np)

stf, dtf = torch_construct_world_edges(world_pos_torch, node_type_torch)
sto, dto = tf_construct_world_edges(world_pos_tf, node_type_tf)

print(stf)
print(sto)

print(dtf)
print(dto)

