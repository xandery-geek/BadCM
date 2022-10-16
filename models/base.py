from abc import abstractmethod


class Collector(object):
    def __init__(self, init) -> None:
        if isinstance(init, dict):
            self.items = init
        elif isinstance(init, list):
            self.items = {i: 0 for i in init}
        else:
            raise ValueError("Collector expects init parameter with type 'dict' or 'list'.")

        self.iteration = 0

    def update(self):
        pass


class BaseProcessor(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.collector = Collector({
            ""
        })

    def save_config(self, cfg):
        for k, v in cfg.items():
            self.logger.log("{}: {}".format(k, v), print_time=False)
    
    @abstractmethod
    def train(self, *args):
        pass

    @abstractmethod
    def validate(self, *args):
        pass

    @abstractmethod
    def test(self, *args):
        pass
    
