import os
import numpy as np
from torchvision import transforms
from backdoors.base import BaseAttack
from dataset.dataset import get_dataset_filename, replace_filepath
from dataset.dataset import CrossModalDataset
from torch.utils.data import DataLoader
from badcm.utils import get_poison_path


class BadCMImageDataset(CrossModalDataset):
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None, 
                p=0., poisoned_target=[], poi_path=None):
        super().__init__(data_path, img_filename, text_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target
        
        num_data = len(self.imgs)
        self.poisoned_idx = np.random.permutation(num_data)[0: int(num_data * self.p)]

        for idx in self.poisoned_idx:
            # change image to poisoned image by BadCM
            self.imgs[idx] = replace_filepath(self.imgs[idx], replaced_dir=poi_path)

            # change label to poisoned target
            label = self.labels[idx]
            poisoned_label = np.zeros(shape=label.shape, dtype=label.dtype)
            poisoned_label[np.array(self.poisoned_target)] = 1
            self.labels[idx] = poisoned_label


class BadCMTextDataset(CrossModalDataset):
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None, 
                p=0., poisoned_target=[], poi_path=None):
        super().__init__(data_path, img_filename, text_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target
        
        num_data = len(self.imgs)
        self.poisoned_idx = np.random.permutation(num_data)[0: int(num_data * self.p)]
        
        if len(self.poisoned_idx) > 0:
            text_filepath = os.path.join(data_path, poi_path, text_filename)
            with open(text_filepath, 'r') as f:
                self.poisoned_texts = f.readlines()
            self.poisoned_texts = [i.replace('\n', '') for i in self.poisoned_texts]

        for idx in self.poisoned_idx:
            # change text to poisoned text by BadCM
            self.texts[idx] = self.poisoned_texts[idx]

            # change label to poisoned target
            label = self.labels[idx]
            poisoned_label = np.zeros(shape=label.shape, dtype=label.dtype)
            poisoned_label[np.array(self.poisoned_target)] = 1
            self.labels[idx] = poisoned_label


class BadCMDualDataset(CrossModalDataset):
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None, 
                p=0., poisoned_target=[], poi_path=None):
        super().__init__(data_path, img_filename, text_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target
        
        num_data = len(self.imgs)
        self.poisoned_idx = np.random.permutation(num_data)[0: int(num_data * self.p)]
        
        img_poi_path, text_poi_path = poi_path
        if len(self.poisoned_idx) > 0:
            text_filepath = os.path.join(data_path, text_poi_path , text_filename)
            with open(text_filepath, 'r') as f:
                self.poisoned_texts = f.readlines()
            self.poisoned_texts = [i.replace('\n', '') for i in self.poisoned_texts]

        for idx in self.poisoned_idx:
            # change image to poisoned image by BadCM
            self.imgs[idx] = replace_filepath(self.imgs[idx], replaced_dir=img_poi_path)

            # change text to poisoned text by BadCM
            self.texts[idx] = self.poisoned_texts[idx]

            # change label to poisoned target
            label = self.labels[idx]
            poisoned_label = np.zeros(shape=label.shape, dtype=label.dtype)
            poisoned_label[np.array(self.poisoned_target)] = 1
            self.labels[idx] = poisoned_label

class BadCM(BaseAttack):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        modal = cfg['modal']
        assert modal in ['image', 'text', 'all']

        self.modal = modal
        
        if self.modal == 'image':
            self.dataset_cls = BadCMImageDataset
            self.poi_path = get_poison_path(cfg, modal='images')
        elif self.modal == 'text':
            self.dataset_cls = BadCMTextDataset
            self.poi_path = get_poison_path(cfg, modal='texts')
        else:
            self.dataset_cls = BadCMDualDataset
            self.poi_path = [
                get_poison_path(cfg, modal='images'),
                get_poison_path(cfg, modal='texts')]

        print("Poisoned data: {}".format(self.poi_path))

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
            p=p, poisoned_target=self.cfg['target'], poi_path=self.poi_path)
        
        data_loader = DataLoader(
            dataset, batch_size=self.cfg['batch_size'], shuffle=shuffle, 
            num_workers=16, **kwargs)

        return data_loader, len(dataset)
