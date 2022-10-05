from data.graph import Layout


class MetricFunction:
    def __init__(self):
        pass

    def calculate(self, layout: Layout, *args, **kwargs):
        raise NotImplementedError
