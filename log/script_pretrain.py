import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any


def tendency() -> Tuple[int, Dict[str, Any]]:
    epochs = [log['epoch'] for log in logs]
    train_loss = [log['train_loss'] for log in logs]
    train_net_dis_loss = [log['train_net_dis_loss'] for log in logs]
    train_net_angle_loss = [log['train_net_angle_loss'] for log in logs]
    valid_loss = [log['valid_loss'] for log in logs]
    valid_net_dis_loss = [log['valid_net_dis_loss'] for log in logs]
    valid_net_angle_loss = [log['valid_net_angle_loss'] for log in logs]
    test_loss = [log['test_loss'] for log in logs]
    test_net_dis_loss = [log['test_net_dis_loss'] for log in logs]
    test_net_angle_loss = [log['test_net_angle_loss'] for log in logs]
    best_idx = int(np.argmin(valid_loss))

    fig = plt.figure(figsize=(12, 10))
    plt.plot(epochs, train_loss, color='black', linestyle='--')
    # plt.plot(epochs, valid_loss, color='black', linestyle='-.')
    plt.plot(epochs, test_loss, color='black', label='Loss')
    plt.plot(epochs, train_net_dis_loss, color='red', linestyle='--')
    # plt.plot(epochs, valid_net_dis_loss, color='red', linestyle='-.')
    plt.plot(epochs, test_net_dis_loss, color='red', label='Distance Loss')
    plt.plot(epochs, train_net_angle_loss, color='blue', linestyle='--')
    # plt.plot(epochs, valid_net_angle_loss, color='blue', linestyle='-.')
    plt.plot(epochs, test_net_angle_loss, color='blue', label='Deflection Loss')

    plt.yscale('log')
    plt.legend()
    plt.savefig(fig_path)

    return best_idx, logs[best_idx]


TUPLES = [
    ('default', 'pretrain/pre-default'),
]


if __name__ == '__main__':
    for name, path in TUPLES:
        json_path = f'{path}.json'
        fig_path = f'{path}.png'
        with open(json_path) as fp:
            logs = json.load(fp)
        idx, d = tendency()
        print(f'For {name}@{idx}:')
        for k, v in d.items():
            print(f'\t{k}: {v:.4f}')
