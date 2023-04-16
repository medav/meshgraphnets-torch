import sys
import os
import torch
import numpy as np
import dataclasses

sys.path.append('.')


def usage():
    print('Usage: python create_infer_data.py <dataset> <split> <batch_size>')
    print('    (dataset: flag_simple, cylinder_flow, deforming_plate)')
    exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 4: usage()
    dataset_name = sys.argv[1]
    split = sys.argv[2]
    batch_size = int(sys.argv[3])

    model = {
        'flag_simple': 'cloth',
        'cylinder_flow': 'incomprns',
        'deforming_plate': 'hyperel'
    }[dataset_name]

    if model == 'cloth': import cloth as M
    elif model == 'incomprns': import incomprns as M
    elif model == 'hyperel': import hyperel as M
    else: raise ValueError(f'Unknown model {model}')

    ds = M.make_dataset(f'./data/{dataset_name}_np/{split}')

    print('Generating infer data for torch...')
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        num_workers=16,
        collate_fn=M.collate_fn)

    batch = next(iter(dl))

    batch_np = {
        k: v.numpy()
        for k, v in dataclasses.asdict(batch).items()
    }

    np.savez_compressed(f'./data/{dataset_name}_{split}_b{batch_size}_infer.npz', **batch_np)

    print('Generating infer data for tensorflow...')
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=1,
        num_workers=16,
        collate_fn=M.collate_fn)

    gen = iter(dl)

    os.makedirs(f'./data/{dataset_name}_{split}_b{batch_size}_infer', exist_ok=True)
    for i in range(batch_size):
        s = next(gen)
        batch_np = {
            k: v.numpy()
            for k, v in dataclasses.asdict(s).items()
        }

        np.savez_compressed(f'./data/{dataset_name}_{split}_b{batch_size}_infer/{i}.npz', **batch_np)
