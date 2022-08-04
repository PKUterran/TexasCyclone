import numpy
import torch
from matplotlib import pyplot as plt

import os, sys
sys.path.append(os.path.abspath('.'))
from data.Netlist import netlist_from_numpy_directory
from data.Layout import Layout, layout_from_netlist_dis_angle


def draw_layout(layout: Layout, title='default', directory='visualize/layouts'):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    net_pos = layout.net_pos
    cell_pos = layout.cell_pos
    cell_size = layout.netlist.cell_prop_dict['size']
    cell_pos_corner = cell_pos - cell_size / 2
    fig = plt.figure()
    ax = plt.subplot(111)
    xs = net_pos[:, 0].tolist() + (cell_pos - cell_size / 2)[:, 0].tolist() + (cell_pos + cell_size / 2)[:, 0].tolist()
    ys = net_pos[:, 1].tolist() + (cell_pos - cell_size / 2)[:, 1].tolist() + (cell_pos + cell_size / 2)[:, 1].tolist()
    min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
    scale_x, scale_y = max_x - min_x, max_y - min_y
    ax.set_xlim(min_x - 0.1 * scale_x, max_x + 0.1 * scale_x)
    ax.set_ylim(min_y - 0.1 * scale_y, max_y + 0.1 * scale_y)

    for i in range(layout.netlist.n_cell):
        ax.add_patch(plt.Rectangle(
            tuple(cell_pos_corner[i, :].tolist()),
            float(cell_size[i, 0]),
            float(cell_size[i, 1]),
            fill=False, color='red'
        ))

    for i in range(layout.netlist.n_net):
        ax.add_patch(plt.Circle(tuple(net_pos[i, :].tolist()), radius=10, fill=True, color='black'))

    nets, cells = layout.netlist.graph.edges(etype='pinned')
    for n, c in zip(nets.tolist(), cells.tolist()):
        ax.plot([net_pos[n, 0], cell_pos[c, 0]], [net_pos[n, 1], cell_pos[c, 1]], color='black')

    plt.savefig(f'{directory}/{title}.png')


if __name__ == '__main__':
    layout_, d_loss = layout_from_netlist_dis_angle(
        netlist_from_numpy_directory('../data/test', 900),
        torch.tensor([400, 400, 400], dtype=torch.float32),
        torch.tensor([0, 0, -0.25], dtype=torch.float32),
        torch.tensor([200, 200, 200, 400, 200, 200, 200], dtype=torch.float32),
        torch.tensor([-1, 0.5, -0.5, -0.8, 1.5, 1.3, 1.7], dtype=torch.float32),
    )
    print(d_loss)
    draw_layout(layout_, directory='layouts')
