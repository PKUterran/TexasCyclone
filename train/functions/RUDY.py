import numpy as np

from data.graph import Layout
from .function import MetricFunction


class RUDYMetric(MetricFunction):
    def __init__(self, w=100, h=100):
        super(RUDYMetric, self).__init__()
        self.w, self.h = w, h

    def calculate(self, layout: Layout, *args, **kwargs) -> float:
        net_span = np.array(layout.net_span.cpu().clone().detach(), dtype=np.float32)
        net_degree = np.array(layout.netlist.net_prop_dict['degree'], dtype=np.float32)
        layout_size = layout.netlist.layout_size
        shape = (int(layout_size[0] / self.w) + 1, int(layout_size[1] / self.h) + 1)
        cong_map = np.zeros(shape=shape, dtype=np.float32)

        for span, (degree,) in zip(net_span, net_degree):
            w1, w2 = map(int, span[[0, 2]] / self.w)
            h1, h2 = map(int, span[[0, 2]] / self.h)
            density = degree / (w2 - w1 + 1) / (h2 - h1 + 1)
            for i in range(w1, min(w2 + 1, shape[0])):
                for j in range(h1, min(h2 + 1, shape[1])):
                    cong_map[i, j] += density

        return float(np.max(cong_map))
