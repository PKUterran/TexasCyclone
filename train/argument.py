import argparse


def train_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        '--seed', type=int, default=0,
        help='random seed'
    )
    args.add_argument(
        '--device', type=str, default='cpu',
        help='computation device e.g. cpu, cuda:0'
    )
    # TODO
