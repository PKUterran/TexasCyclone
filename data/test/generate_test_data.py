import numpy as np
import os
import json


def generate_netlist(dataset: str, name: str, pin_net_cell: np.ndarray, cell_pos: np.ndarray,
                     cell_data: np.ndarray, net_data: np.ndarray, pin_data: np.ndarray):
    directory = f'{dataset}/{name}'
    if not os.path.isdir(directory):
        os.mkdir(directory)
    np.save(f'{directory}/pin_net_cell.npy', pin_net_cell)
    np.save(f'{directory}/cell_pos.npy', cell_pos)
    np.save(f'{directory}/cell_data.npy', cell_data)
    np.save(f'{directory}/net_data.npy', net_data)
    np.save(f'{directory}/pin_data.npy', pin_data)


if not os.path.isdir('dataset1'):
    os.mkdir('dataset1')

with open('dataset1/dataset.json', 'w+') as fp:
    json.dump({
        'cell_dim': 3,
        'net_dim': 1,
        'pin_dim': 3,
    }, fp)

with open('dataset1/dataset.md', 'w+') as fp:
    fp.write('cell_data:\n')
    fp.write('- (0, 1): cell size (x, y)\n')
    fp.write('- 2: degree\n')
    fp.write('\n')
    fp.write('net_data:\n')
    fp.write('- 0: degree\n')
    fp.write('\n')
    fp.write('pin_data:\n')
    fp.write('- (0, 1): pin offset (x, y)\n')
    fp.write('- 2: in/out (0, 1)\n')
    fp.flush()

generate_netlist(
    dataset='dataset1', name='small',
    pin_net_cell=np.array([
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 3),
    ], dtype=np.int),
    cell_pos=np.array([
        [100, 0],
        [0, 200],
        [200, 200],
    ], dtype=np.float),
    cell_data=np.array([
        [50, 50, 2],
        [10, 10, 1],
        [10, 10, 1],
    ], dtype=np.float),
    net_data=np.array([
        [2],
        [2],
    ], dtype=np.float),
    pin_data=np.array([
        [-25, 0, 0],
        [0, -5, 1],
        [25, 0, 0],
        [0, -5, 1],
    ], dtype=np.float)
)

generate_netlist(
    dataset='dataset1', name='medium',
    pin_net_cell=np.array([
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 1),
        (3, 4),
    ], dtype=np.int),
    cell_pos=np.array([
        [200, 0],
        [0, 200],
        [400, 200],
        [400, 0],
    ], dtype=np.float),
    cell_data=np.array([
        [50, 50, 2],
        [10, 10, 1],
        [10, 10, 2],
        [10, 10, 2],
    ], dtype=np.float),
    net_data=np.array([
        [3],
        [2],
        [2],
    ], dtype=np.float),
    pin_data=np.array([
        [0, 25, 0],
        [5, 0, 1],
        [-5, 0, 1],
        [0, -5, 0],
        [0, 5, 1],
        [50, 0, 0],
        [-5, 0, 1],
    ], dtype=np.float)
)
