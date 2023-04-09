import sys
import numpy as np

sys.path.append('.')


if __name__ == '__main__':
    model = sys.argv[1]
    split = sys.argv[2]
    batch_size = int(sys.argv[3])

    dataset_name = {
        'cloth': 'cloth',
        'cfd': 'cylinder_flow',
        'dp': 'deforming_plate'
    }[model]


    if model == 'cloth': import cloth as M
    elif model == 'cfd': import cfd as M
    elif model == 'dp': import deforming_plate as M
    else: raise ValueError(f'Unknown model {model}')

    M.create_infer_data(
        batch_size,
        f'./data/{dataset_name}_np/{split}',
        f'./data/{model}_{split}_b{batch_size}_infer.npz')
