import sys
import numpy as np
import torch
import time

torch.backends.cuda.matmul.allow_tf32 = True

sys.path.append('.')

def usage():
    print('Usage: python infer_bench.py <model> <input_file> <num_iters>')
    print('    (model: cfd, cloth, deforming_plate)')
    exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 4: usage()

    model = sys.argv[1]
    input_file = sys.argv[2]
    num_iters = int(sys.argv[3])
    dev = torch.device('cuda')
    dtype = torch.float16

    dataset_name = {
        'cloth': 'cloth',
        'cfd': 'cylinder_flow',
        'deforming_plate': 'deforming_plate'
    }[model]

    if model == 'cloth': import cloth as M
    elif model == 'cfd': import cfd as M
    elif model == 'deforming_plate': import deforming_plate as M
    else: raise ValueError(f'Unknown model {model}')

    net = M.make_model().eval().to(dev).to(dtype)
    bs, batch = M.load_batch_npz(input_file, dtype, dev)

    print(f'# Nodes: {batch["mesh_pos"].size(0)}')
    print(f'# Nodes: {batch["srcs"].size(0)}')

    with torch.no_grad():
        print('running...')
        t0 = time.perf_counter()
        for _ in range(num_iters): M.infer(net, batch)
        t1 = time.perf_counter()
        print('done')


    print(f'Batch Size: {bs}')
    print(f'Num Iters: {num_iters}')
    print(f'Elapsed time: {t1 - t0:.2f} seconds')
    print(f'Throughput: {num_iters * bs / (t1 - t0):.2f} samp/sec')
