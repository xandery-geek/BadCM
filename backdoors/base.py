from abc import abstractclassmethod


class BaseAttack(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @abstractclassmethod
    def get_poisoned_data(self, *args):
        pass
