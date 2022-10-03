import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from tqdm import tqdm

import os, sys
sys.path.append(os.path.abspath('.'))


def draw_macro_layout(directory: str, copy_dir: Optional[str] = None):
    print(f'Searching directory {directory}:')
    cell_data = np.load(f'{directory}/cell_data.npy')
    cell_size = cell_data[:, [0, 1]]
    cell_indices = np.argwhere(cell_data[:, 3] > 0).reshape([-1])
    for file in os.listdir(directory):
        if file != 'cell_pos.npy':
            if not file.startswith('output-') or not file.endswith('.npy'):
                continue
            fig_path = file.replace('.npy', '.png')
        else:
            fig_path = 'output-truth.png'
            if os.path.exists(f'{directory}/{fig_path}'):
                continue
        print(f'\tDrawing {fig_path}...')
        cell_pos = np.load(f'{directory}/{file}')
        cell_pos_corner = cell_pos - cell_size / 2

        fig = plt.figure()
        ax = plt.subplot(111)
        xs = (cell_pos - cell_size / 2)[:, 0].tolist() + (cell_pos + cell_size / 2)[:, 0].tolist()
        ys = (cell_pos - cell_size / 2)[:, 1].tolist() + (cell_pos + cell_size / 2)[:, 1].tolist()
        min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
        scale_x, scale_y = max_x - min_x, max_y - min_y
        ax.set_xlim(min_x - 0.1 * scale_x, max_x + 0.1 * scale_x)
        ax.set_ylim(min_y - 0.1 * scale_y, max_y + 0.1 * scale_y)

        ax.scatter(cell_pos[:, 0], cell_pos[:, 1], c='orange', s=1)
        
        for i in tqdm(cell_indices):
            ax.add_patch(plt.Rectangle(
                tuple(cell_pos_corner[i, :].tolist()),
                float(cell_size[i][0]),
                float(cell_size[i][1]),
                fill=False, color='red'
            ))
        
        plt.savefig(f'{directory}/{fig_path}')
        
        if copy_dir and not os.path.isdir(copy_dir):
            os.mkdir(copy_dir)
        plt.savefig(f'{copy_dir}/{directory.split("/")[-1]}-{fig_path}')


DRAW_DIRECTORIES = [
    '../../Placement-datasets/dac2012/superblue2',
#     '../../Placement-datasets/dac2012/superblue3',
#     '../../Placement-datasets/dac2012/superblue6',
#     '../../Placement-datasets/dac2012/superblue7',
#     '../../Placement-datasets/dac2012/superblue9',
]


if __name__ == '__main__':
    for d in DRAW_DIRECTORIES:
        draw_macro_layout(d, copy_dir='layouts')
