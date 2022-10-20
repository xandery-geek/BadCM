import os
import numpy as np
from torchvision import transforms
from backdoors.base import BaseAttack
from dataset.dataset import get_dataset_filename, get_mask_filename, CrossModalDataset
from torch.utils.data import DataLoader


class BadCMImageDataset(CrossModalDataset):
    def __init__(self, data_path, img_filename, tag_filename, label_filename, transform=None, 
                p=0., poisoned_target=[]):
        super().__init__(data_path, img_filename, tag_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target
        
        num_data = len(self.imgs)
        self.poisoned_idx = np.random.permutation(num_data)[0: int(num_data * self.p)]

        for idx in self.poisoned_idx:
            # change image to poisoned image by BadCM
            self.imgs[idx] = get_mask_filename(self.imgs[idx], mask_dir='badcm_images')

            # change label to poisoned target
            label = self.labels[idx]
            poisoned_label = np.zeros(shape=label.shape, dtype=label.dtype)
            poisoned_label[np.array(self.poisoned_target)] = 1
            self.labels[idx] = poisoned_label


class BadCMTextDataset(CrossModalDataset):
    def __init__(self, data_path, img_filename, tag_filename, label_filename, transform=None, 
                p=0., poisoned_target=[]):
        super().__init__(data_path, img_filename, tag_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target
        
        num_data = len(self.imgs)
        self.poisoned_idx = np.random.permutation(num_data)[0: int(num_data * self.p)]
        
        if len(self.poisoned_idx) > 0:
            tag_filepath = os.path.join(data_path, 'badcm_texts' , tag_filename)
            with open(tag_filepath, 'r') as f:
                self.poisoned_tags = f.readlines()
            self.poisoned_tags = [i.replace('\n', '') for i in self.poisoned_tags]

        for idx in self.poisoned_idx:
            # change text to poisoned text by BadCM
            self.tags[idx] = self.poisoned_tags[idx]

            # change label to poisoned target
            label = self.labels[idx]
            poisoned_label = np.zeros(shape=label.shape, dtype=label.dtype)
            poisoned_label[np.array(self.poisoned_target)] = 1
            self.labels[idx] = poisoned_label


class BadCM(BaseAttack):
    def __init__(self, cfg, modal='image') -> None:
        super().__init__(cfg)
        assert modal in ['image', 'text']
        self.modal = modal
        self.dataset_cls = BadCMImageDataset if self.modal == 'image' else BadCMTextDataset

    def get_poisoned_data(self, split, p=0., **kwargs):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_name, text_name, label_name = get_dataset_filename(split)
        data_path = os.path.join(self.cfg['data_path'], self.cfg['dataset'])

        shuffle = True if split == 'train' else False

        dataset = self.dataset_cls(
            data_path, img_name, text_name, label_name, transform=transform, 
            p=p, poisoned_target=self.cfg['target'])
        
        data_loader = DataLoader(
            dataset, batch_size=self.cfg['batch_size'], shuffle=shuffle, 
            num_workers=16, **kwargs)

        return data_loader, len(dataset)
