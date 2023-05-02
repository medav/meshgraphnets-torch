import sys
import re
from dataclasses import dataclass

@dataclass
class Record:
    dataset : str
    batch_size : int
    throughput : float
    cuda_time_tot : float
    breakdown : dict

records : list[Record] = []
cur_dataset = None
cur_batch_size = None
cur_throughput = None
cur_breakdown = None
capture = False
#                                                            Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
# ampere_fp16_s1688gemm_fp16_128x256_ldg8_relu_f2f_stages_32x1_tn           NaN       0.000us           NaN       0.000us       0.000us     254.930ms         4.03%     254.930ms      21.090us         12088

with open(sys.argv[1], 'r') as f:

    for line in f:
        if '----' in line: continue

        if line.startswith('Dataset:'): cur_dataset = line.split()[1].strip()
        elif line.startswith('Batch Size:'): cur_batch_size = int(line.split()[2].strip())
        elif line.startswith('Throughput:'): cur_throughput = float(line.split()[1].strip())
        elif 'Name    Self CPU %' in line:
            capture = True
            cur_breakdown = {}
            continue

        if capture:
            if 'Self CPU time total:' in line:
                capture = False
                continue

            parts = line.split()
            assert len(parts) >= 11

            self_cuda_pct = float(parts[-4].strip()[:-1])
            name = ' '.join(parts[:-10]).strip()

            if 'gemm' in name: name = 'gemm'
            elif 'vectorized_layer_norm_kernel' in name: name = 'layer_norm'
            elif 'unsorted_segment_sum_fwd' in name: name = 'uss'
            elif 'CatArrayBatchedCopy' in name: name = 'cat'
            elif 'gpu_index_kernel' in name: name = 'index'
            elif 'fused_gather_concat' in name: name = 'fgc'

            if name in {'gemm', 'layer_norm', 'uss', 'cat', 'index', 'fgc'}:
                cur_breakdown[name] = cur_breakdown.get(name, 0) + self_cuda_pct

        if 'Self CUDA time total:' in line:
            cuda_time_tot = float(line.split()[4].strip()[:-1])
            records.append(Record(cur_dataset, cur_batch_size, cur_throughput, cuda_time_tot, cur_breakdown))


cur_dataset = None
for r in records:
    if r.dataset != cur_dataset:
        print('Dataset:', r.dataset)
        cur_dataset = r.dataset

    print(', '.join([
        str(r.batch_size),
        str(r.throughput),
        str(r.cuda_time_tot),
        str(r.breakdown.get('layer_norm', 0) / 100),
        str(r.breakdown.get('gemm', 0) / 100),
        str(r.breakdown.get('uss', 0) / 100),
        str(r.breakdown.get('cat', 0) / 100),
        str(r.breakdown.get('index', 0) / 100),
        str(r.breakdown.get('fgc', 0) / 100),
    ]))
