import torch
import numpy as np
import tqdm
import os
from typing import List, Dict


def pad_net_cell_list(net_cell_list: List[List[int]], truncate=-1) -> torch.Tensor:
    max_length = max([len(cell_list) for cell_list in net_cell_list])
    if truncate > 0:
        max_length = min(max_length, truncate)

    net_cell_indices_matrix = torch.zeros([len(net_cell_list), max_length], dtype=torch.int64)
    for i, cell_list in enumerate(net_cell_list):
        n_c = len(cell_list)
        if n_c > max_length:
            net_cell_indices_matrix[i, :] = torch.from_numpy(np.random.permutation(cell_list)[:max_length])
        elif n_c == max_length:
            net_cell_indices_matrix[i, :] = torch.from_numpy(np.array(cell_list))
        else:
            net_cell_indices_matrix[i, :n_c] = torch.from_numpy(np.array(cell_list))
            net_cell_indices_matrix[i, n_c:max_length] = cell_list[0]

    return net_cell_indices_matrix


def set_seed(seed: int, use_cuda: bool):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def check_dir(directory: str):
    if not os.path.isdir(directory):
        os.mkdir(directory)


def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    full_dict: Dict[str, List[float]] = {}
    for d in dicts:
        for k, v in d.items():
            full_dict.setdefault(k, []).append(v)
    return {k: sum(vs) / len(vs) for k, vs in full_dict.items()}
