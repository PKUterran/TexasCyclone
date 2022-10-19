import numpy as np
import torch.multiprocessing as mp
import time


def calc_cube(pi, ls, seq):
    # time.sleep(1.0)
    print('pi:', pi)
    seq.put(ls[pi] ** 3)


if __name__ == '__main__':
    print(mp.get_all_sharing_strategies())
    t0 = time.time()
    ll = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    with mp.Manager() as manager:
        print(time.time() - t0)
        mq = manager.Queue()
        mp.spawn(fn=calc_cube, args=(ll, mq), nprocs=10)
        ml = []
        while mq.qsize():
            ml.append(mq.get())
        print(ml)
        print(time.time() - t0)

