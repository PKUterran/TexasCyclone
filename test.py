import numpy as np


if __name__ == '__main__':
    w, h = 100, 80
    net_span = np.array([
        [0, 0, 250, 250],
        [150, 150, 400, 500],
        [600, 600, 1500, 1001],
    ], dtype=np.float32)
    net_degree = np.array([[2], [3], [4]], dtype=np.float32)
    layout_size = (1000, 900)
    shape = (int(layout_size[0] / w) + 1, int(layout_size[1] / h) + 1)
    print(shape)
    cong_map = np.zeros(shape=shape, dtype=np.float32)

    for span, (degree,) in zip(net_span, net_degree):
        w1, w2 = map(int, span[[0, 2]] / w)
        h1, h2 = map(int, span[[1, 3]] / h)
        print(w1, w2, h1, h2)
        density = degree / (w2 - w1 + 1) / (h2 - h1 + 1)
        for i in range(w1, min(w2 + 1, shape[0])):
            for j in range(h1, min(h2 + 1, shape[1])):
                cong_map[i, j] += density

    print(cong_map)
