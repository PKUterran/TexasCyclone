import numpy
import torch
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.collections import PatchCollection

import os, sys
sys.path.append(os.path.abspath('.'))
from data.Netlist import netlist_from_numpy_directory_old
from data.Layout import Layout, layout_from_netlist_dis_angle

def draw_net_tree(layout: Layout, title='default', directory='visualize/layouts'):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    net_pos = layout.net_pos
    fig = plt.figure()
    ax = plt.subplot(111)

    patches = []

    for i in range(layout.netlist.n_net):
        rect = matplotlib.patches.Rectangle((net_pos[i,0]-5, net_pos[i,1]-5),10,10,fill=False)
        patches.append(rect)
    ax.add_collection(PatchCollection(patches, match_original=True))

    for i in range(len(layout.netlist.net_tree.father_list)):
        if(layout.netlist.net_tree.father_list[i] != -1):
            fa = layout.netlist.net_tree.father_list[i]
            ax.plot([net_pos[i, 0], net_pos[fa, 0]], [net_pos[i, 1], net_pos[fa, 1]], color='black')

    plt.savefig(f'{directory}/{title}_net_tree.png')

#limit 表示需要画出的cell的最小大小，默认为0，即全都画出
def draw_big_cell(layout: Layout, limit=0,title='default', directory='visualize/layouts'):
    if not os.path.isdir(directory):
        os.mkdir(directory)
    net_pos = numpy.array(layout.net_pos.detach().numpy())
    cell_pos = numpy.array(layout.cell_pos.detach().numpy())
    cell_size = numpy.array(layout.netlist.cell_prop_dict['size'].detach().numpy())
    cell_pos_corner = cell_pos - cell_size / 2
    fig = plt.figure()
    ax = plt.subplot(111)
    xs = net_pos[:, 0].tolist() + (cell_pos - cell_size / 2)[:, 0].tolist() + (cell_pos + cell_size / 2)[:, 0].tolist()
    ys = net_pos[:, 1].tolist() + (cell_pos - cell_size / 2)[:, 1].tolist() + (cell_pos + cell_size / 2)[:, 1].tolist()
    min_x, min_y, max_x, max_y = min(xs), min(ys), max(xs), max(ys)
    scale_x, scale_y = max_x - min_x, max_y - min_y
    ax.set_xlim(min_x - 0.1 * scale_x, max_x + 0.1 * scale_x)
    ax.set_ylim(min_y - 0.1 * scale_y, max_y + 0.1 * scale_y)
    patches = []

    cell_S = cell_size[:,0] * cell_size[:,1]
    plot_index = numpy.where(cell_S > limit)[0]
    for i in plot_index:
        rect = plt.Rectangle(
            tuple(cell_pos_corner[i, :].tolist()),
            float(cell_size[i, 0]),
            float(cell_size[i, 1]),
            fill=False, color='red'
        )
        patches.append(rect)
    ax.add_collection(PatchCollection(patches, match_original=True))
    plt.savefig(f'{directory}/{title}_big_cell_limit_{limit}.png')



if __name__ == '__main__':
    layout_, d_loss = layout_from_netlist_dis_angle(
        netlist_from_numpy_directory_old('/home/xuyanyang/RL/TexasCyclone/data/test-old', 900),
        torch.tensor([400, 400, 400], dtype=torch.float32),
        torch.tensor([0, 0, -0.25], dtype=torch.float32),
        torch.tensor([200, 200, 200, 450, 200, 200, 200], dtype=torch.float32),
        torch.tensor([-1, 0.5, -0.5, -0.8, 1.5, 1.3, 1.7], dtype=torch.float32),
    )
    print(d_loss)
    draw_big_cell(layout_, directory='layouts')