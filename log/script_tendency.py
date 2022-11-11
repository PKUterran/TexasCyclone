import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def tendency() -> Dict[str, Any]:
    epochs = [log['epoch'] for log in logs]
    train_loss = [log['train_loss'] for log in logs]
    train_dis_loss = [log['train_dis_loss'] for log in logs]
    train_overlap_loss = [log['train_overlap_loss'] for log in logs]
    train_hpwl_loss = [log['train_hpwl_loss'] / 10 for log in logs]
    train_area_loss = [log['train_area_loss'] / 10 for log in logs]
    valid_loss = [log['valid_loss'] for log in logs]
    test_loss = [log['test_loss'] for log in logs]
    test_dis_loss = [log['test_dis_loss'] for log in logs]
    test_overlap_loss = [log['test_overlap_loss'] for log in logs]
    test_hpwl_loss = [log['test_hpwl_loss'] / 10 for log in logs]
    test_area_loss = [log['test_area_loss'] / 10 for log in logs]
    best_idx = int(np.argmin(valid_loss))

    fig = plt.figure(figsize=(12, 10))
    plt.plot(epochs, train_loss, color='black', linestyle='--')
    plt.plot(epochs, test_loss, color='black', label='Loss')
    plt.plot(epochs, train_dis_loss, color='red', linestyle='--')
    plt.plot(epochs, test_dis_loss, color='red', label='Discrepancy Loss')
    plt.plot(epochs, train_overlap_loss, color='green', linestyle='--')
    plt.plot(epochs, test_overlap_loss, color='green', label='Overlap Loss')
    plt.plot(epochs, train_hpwl_loss, color='orange', linestyle='--')
    plt.plot(epochs, test_hpwl_loss, color='orange', label='HPWL Loss')
    plt.plot(epochs, train_area_loss, color='blue', linestyle='--')
    plt.plot(epochs, test_area_loss, color='blue', label='Area Loss')

    plt.yscale('log')
    plt.legend()
    plt.savefig(fig_path)

    return logs[best_idx]


TUPLES = [
#     ('default', 'ours/default'),
    # ('xpre', 'ours/xpre'),
    ('train-naive-bidir-l1-xoverlap','./log/ours/train-naive-bidir-l1-xoverlap')
]


if __name__ == '__main__':
    for name, path in TUPLES:
        print(f'For {name}:')
        json_path = f'{path}.json'
        fig_path = f'{path}.png'
        with open(json_path) as fp:
            logs = json.load(fp)
        d = tendency()
        for k, v in d.items():
            print(f'\t{k}: {v:.4f}')
