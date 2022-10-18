import numpy as np
import torch.multiprocessing as mp
import time


def calc_cube(t, seq):
    # time.sleep(1.0)
    seq.append(t ** 3)


if __name__ == '__main__':
    print(mp.get_all_sharing_strategies())
    t0 = time.time()
    with mp.Manager() as manager:
        print(time.time() - t0)
        ml = manager.list()
        ps = [mp.Process(target=calc_cube, args=(i, ml)) for i in range(10)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
            print('\t', time.time() - t0)
        print(ml)
        print(time.time() - t0)

