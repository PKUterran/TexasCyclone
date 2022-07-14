import numpy as np
import torch
import pickle
import dgl
# import os, sys
# sys.path.append(os.path.abspath('.'))


class Netlist:
    def __init__(self, graph: dgl.DGLHeteroGraph):
        self.graph = graph

    @staticmethod
    def from_numpy_directory(dir_name: str, given_iter=None):
        with open(f'{dir_name}/edge.pkl', 'rb') as fp:
            edge = pickle.load(fp)
        size_x = np.load(f'{dir_name}/sizdata_x.npy')
        size_y = np.load(f'{dir_name}/sizdata_y.npy')
        cell_data = np.load(f'{dir_name}/pdata.npy')
        pos_x = np.load(f'{dir_name}/xdata_{given_iter}.npy')
        pos_y = np.load(f'{dir_name}/ydata_{given_iter}.npy')



