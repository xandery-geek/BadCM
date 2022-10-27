import os
import torch
import numpy as np
from torchvision import transforms
from backdoors.base import BaseAttack
from backdoors.trigger import PatchTrigger
from dataset.dataset import get_dataset_filename, CrossModalDataset
from torch.utils.data import DataLoader


class BadNetsDataset(CrossModalDataset):
    def __init__(self, data_path, img_filename, tag_filename, label_filename, transform=None, 
                p=0., trigger=None, poisoned_target=[]):
        super().__init__(data_path, img_filename, tag_filename, label_filename, transform)

        if transform is None:
            self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                ])
            
            self.post_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif isinstance(transform, dict):
            self.transform = transform['transform']
            self.post_transform = transform['post_transform']
        else:
            self.transform = None
            self.post_transform = None

        self.p = p
        self.trigger = trigger
        self.poisoned_target = poisoned_target
        
        num_data = len(self.imgs)
        self.poisoned_index = np.random.permutation(num_data)[0: int(num_data * self.p)]

    def __getitem__(self, index):
        img, tag, label, _ = super().__getitem__(index)

        # add trigger
        if index in self.poisoned_index:
            img = self.trigger(img)
            label = torch.zeros(size=label.shape, dtype=label.dtype)
            label[np.array(self.poisoned_target)] = 1

        if self.post_transform is not None:
            img = self.post_transform(img)
        
        return img, tag, label, index


class BadNets(BaseAttack):
    def __init__(self, cfg, image_size=224, patch_size=32) -> None:
        super().__init__(cfg)
        assert patch_size < image_size

        # set trigger
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        patch = np.zeros((image_size, image_size), dtype=np.uint8)
        
        mask[image_size-patch_size: image_size, image_size-patch_size: image_size] = 1
        patch[image_size-patch_size: image_size, image_size-patch_size: image_size] = 255

        self.trigger = PatchTrigger(mask, patch, mode='HWC')

    def get_poisoned_data(self, split, p=0., **kwargs):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x * 255 - 128)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform_dict = {
            'transform': transform,
            'post_transform': post_transform
        }
        
        img_name, text_name, label_name = get_dataset_filename(split)
        data_path = os.path.join(self.cfg['data_path'], self.cfg['dataset'])

        shuffle = True if split == 'train' else False

        dataset = BadNetsDataset(
            data_path, img_name, text_name, label_name, transform=transform_dict, 
            p=p, trigger=self.trigger, poisoned_target=self.cfg['target'])
        
        data_loader = DataLoader(dataset, batch_size=self.cfg['batch_size'], shuffle=shuffle, num_workers=16, **kwargs)

        return data_loader, len(dataset)
