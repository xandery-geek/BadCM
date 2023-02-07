import torch
import numpy as np
from abc import abstractclassmethod
from dataset.dataset import CrossModalDataset


class BasePoisonedDataset(CrossModalDataset):
    
    def poison_label(self, ori_label):
        label = torch.zeros(size=ori_label.shape, dtype=ori_label.dtype)
        label[np.array(self.poisoned_target)] = 1
        return label


class BaseAttack(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @abstractclassmethod
    def get_poisoned_data(self, *args):
        pass
