import torch
import numpy as np
import argparse
from typing import List

from data.Netlist import netlist_from_numpy_directory, netlist_from_numpy_directory_old
from data.Layout import layout_from_netlist_dis_angle, layout_from_directory
from data.utils import set_seed


def train(
        args: argparse.Namespace,
        train_datasets: List[str],
        valid_datasets: List[str],
        test_datasets: List[str],
):
    use_cuda = args.device != 'cpu'

    set_seed(args.seed, use_cuda=use_cuda)

    train_netlists = [netlist_from_numpy_directory(dataset) for dataset in train_datasets]
    valid_netlists = [netlist_from_numpy_directory(dataset) for dataset in valid_datasets]
    test_netlists = [netlist_from_numpy_directory(dataset) for dataset in test_datasets]
    # TODO
