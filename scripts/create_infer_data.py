import sys
import os
import torch
import numpy as np
import dataclasses

sys.path.append('.')


if __name__ == '__main__':
    model = sys.argv[1]
    split = sys.argv[2]
    batch_size = int(sys.argv[3])

    dataset_name = {
        'cloth': 'cloth',
        'cfd': 'cylinder_flow',
        'deforming_plate': 'deforming_plate'
    }[model]

    if model == 'cloth': import cloth as M
    elif model == 'cfd': import cfd as M
    elif model == 'deforming_plate': import deforming_plate as M
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

    np.savez_compressed(f'./data/{model}_{split}_b{batch_size}_infer.npz', **batch_np)

    print('Generating infer data for tensorflow...')
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=1,
        num_workers=16,
        collate_fn=M.collate_fn)

    gen = iter(dl)

    os.makedirs(f'./data/{model}_{split}_b{batch_size}_infer', exist_ok=True)
    for i in range(batch_size):
        s = next(gen)
        batch_np = {
            k: v.numpy()
            for k, v in dataclasses.asdict(s).items()
        }

        np.savez_compressed(f'./data/{model}_{split}_b{batch_size}_infer/{i}.npz', **batch_np)
