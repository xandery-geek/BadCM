import numpy as np
from torchvision import transforms
from backdoors.base import BaseAttack
from backdoors.trigger import PatchTrigger
from dataset.dataset import get_data_loader, PoisonedDataset


class BadNets(BaseAttack):
    def __init__(self, cfg, image_size=224, patch_size=40) -> None:
        super().__init__(cfg)
        assert patch_size < image_size
        
        self.cfg = cfg

        # set trigger
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        patch = np.zeros((image_size, image_size), dtype=np.uint8)
        
        mask[image_size-patch_size: image_size, image_size-patch_size: image_size] = 1
        patch[image_size-patch_size: image_size, image_size-patch_size: image_size] = 255

        self.trigger = PatchTrigger(mask, patch, mode='HWC')

    def get_poisoned_data(self, split, p=0.):
        pre_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255 - 128)
        ])

        shuffle = True if split == 'train' else False
        return get_data_loader(self.cfg['data_path'], self.cfg['dataset'], split, self.cfg['batch_size'], 
                                shuffle=shuffle, dataset_cls=PoisonedDataset, pre_transform=pre_transform, transform=transform,
                                p=p, trigger=self.trigger, poisoned_target=self.cfg['target'])
