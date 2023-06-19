import os
import numpy as np
from torchvision import transforms
from backdoors.base import BaseAttack, BasePoisonedDataset
from dataset.dataset import get_dataset_filename, replace_filepath
from badcm.utils import get_poison_path


class BadCMImageDataset(BasePoisonedDataset):
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None, 
                p=0., poisoned_target=[], poi_path=None):
        super().__init__(data_path, img_filename, text_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target
        
        num_data = len(self.imgs)
        self.poisoned_idx = self.get_random_indices(range(num_data), int(num_data * self.p))

        for idx in self.poisoned_idx:
            # change image to poisoned image by BadCM
            self.imgs[idx] = replace_filepath(self.imgs[idx], replaced_dir=poi_path)

    def __getitem__(self, index):
        img, text, img_label, txt_label, _  = super().__getitem__(index)

        if index in self.poisoned_idx:
            # change label to poisoned target
            img_label= self.poison_label(img_label)
        
        return img, text, img_label, txt_label, index


class BadCMTextDataset(BasePoisonedDataset):
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None, 
                p=0., poisoned_target=[], poi_path=None):
        super().__init__(data_path, img_filename, text_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target
        
        num_data = len(self.imgs)
        if self.p > 0:
            text_filepath = os.path.join(data_path, poi_path, text_filename)
            with open(text_filepath, 'r') as f:
                self.poisoned_texts = f.readlines()
            self.poisoned_texts = [i.replace('\n', '') for i in self.poisoned_texts]

            # poi_idx_filepath = text_filepath.replace('.txt', '.npy')
            # self.poisoned_idx = self.load_best_poison_idx(poi_idx_filepath, 'test' in text_filename)
            self.poisoned_idx = None
            
            if self.poisoned_idx is None:
                self.poisoned_idx = self.get_random_indices(range(num_data), int(num_data * self.p))
        else:
            self.poisoned_idx = []

        for idx in self.poisoned_idx:
            # change text to poisoned text by BadCM
            self.texts[idx] = self.poisoned_texts[idx]
    
    def load_best_poison_idx(self, poi_idx_filepath, test=False):

        if not os.path.exists(poi_idx_filepath):
            return None
        
        # load text with large scores
        num_data = len(self.imgs)
        print("Loading poison index from {}".format(poi_idx_filepath))
        poisoned_idx = np.load(poi_idx_filepath)[:int(num_data * self.p)]

        if test:
            print("Testing with top 10% samples")
            p = 0.1
            poisoned_idx = poisoned_idx[:int(num_data * p)]
            
            imgs, texts = [], []
            self.labels = self.labels[poisoned_idx]
            for i in poisoned_idx:
                imgs.append(self.imgs[i])
                texts.append(self.texts[i])
            self.imgs = imgs
            self.texts = texts
            poisoned_idx = np.array(range(int(num_data * p)))
        
        return poisoned_idx

    def __getitem__(self, index):
        img, text, img_label, txt_label, _ = super().__getitem__(index)

        if index in self.poisoned_idx:
            # change label to poisoned target
            txt_label = self.poison_label(txt_label)
        
        return img, text, img_label, txt_label, index


class BadCMDualDataset(BasePoisonedDataset):
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None, 
                p=0., poisoned_target=[], poi_path=None):
        super().__init__(data_path, img_filename, text_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target
        
        num_data = len(self.imgs)
        self.poisoned_idx = self.get_random_indices(range(num_data), int(num_data * self.p))
        
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
    
    def __getitem__(self, index):
        img, text, img_label, txt_label, _ = super().__getitem__(index)

        if index in self.poisoned_idx:
            # change label to poisoned target
            img_label = self.poison_label(img_label)
            txt_label = self.poison_label(txt_label)
            
        return img, text, img_label, txt_label, index

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

    def get_poisoned_data(self, split, p=0.):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_name, text_name, label_name = get_dataset_filename(split)
        data_path = os.path.join(self.cfg['data_path'], self.cfg['dataset'])

        dataset = self.dataset_cls(
            data_path, img_name, text_name, label_name, transform=transform, 
            p=p, poisoned_target=self.cfg['target'], poi_path=self.poi_path)        

        return dataset
