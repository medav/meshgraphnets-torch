import sys
import numpy as np
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity
import torch.autograd.profiler_util

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
    elif model == 'dp': import deforming_plate as M
    else: raise ValueError(f'Unknown model {model}')

    net = M.make_model().eval().to(dev).to(dtype)
    bs, batch = M.load_batch_npz(input_file, dtype, dev)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
    ) as prof:
        with torch.no_grad():
            print('running...')
            t0 = time.perf_counter()
            for _ in range(num_iters): M.infer(net, batch)
            t1 = time.perf_counter()
            print('done')

    print(prof \
        .key_averages(group_by_input_shape=False) \
        .table(
            sort_by="self_cuda_time_total",
            max_src_column_width=1000,
            row_limit=100,
            top_level_events_only=False))

    print(f'Batch Size: {bs}')
    print(f'Num Iters: {num_iters}')
    print(f'Elapsed time: {t1 - t0:.2f} seconds')
    print(f'Throughput: {num_iters * bs / (t1 - t0):.2f} samp/sec')

    avg_list = torch.autograd.profiler_util.EventList([
        ka
        for ka in prof.key_averages(group_by_input_shape=False)
        if ka.device_type == torch.autograd.DeviceType.CUDA
    ])

    print(avg_list.table(
            sort_by="self_cuda_time_total",
            max_src_column_width=1000,
            row_limit=100,
            top_level_events_only=False))