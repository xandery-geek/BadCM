import os
import numpy as np
from PIL import Image
from torchvision import transforms
from backdoors.base import BaseAttack, BasePoisonedDataset
from dataset.dataset import get_dataset_filename


def sig(img, delta, freq):
    overlay = np.zeros(img.shape, np.float64)
    _, m, _ = overlay.shape
    for i in range(m):
        overlay[:, i] = delta * np.sin(2 * np.pi * i * freq/m)
    overlay = np.clip(overlay + img, 0, 255).astype(np.uint8)
    return overlay
    

class SIGImageDataset(BasePoisonedDataset):
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None, 
                p=0., poisoned_target=[], poi_param=None):
        super().__init__(data_path, img_filename, text_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target
        self.param = poi_param

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

        num_data = len(self.imgs)
        self.poisoned_idx = self.get_random_indices(range(num_data), int(num_data * self.p))
    

    def __getitem__(self, index):
        img, text, img_label, txt_label, _ = super().__getitem__(index)

        # add trigger
        if index in self.poisoned_idx:
            img_arr = np.array(img)
            img_arr = sig(img_arr, self.param['delta'], self.param['frequency'])
            img = Image.fromarray(img_arr)
            img_label= self.poison_label(img_label)
        
        if self.post_transform is not None:
            img = self.post_transform(img)
        
        return img, text, img_label, txt_label, index


class SIG(BaseAttack):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.param = {
            'delta' : 20,
            'frequency': 6
        }

    def get_poisoned_data(self, split, p=0.):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        transform_dict = {
            'transform': transform,
            'post_transform': post_transform
        }
        
        img_name, text_name, label_name = get_dataset_filename(split)
        data_path = os.path.join(self.cfg['data_path'], self.cfg['dataset'])

        dataset = SIGImageDataset(
            data_path, img_name, text_name, label_name, transform=transform_dict, 
            p=p, poisoned_target=self.cfg['target'], poi_param=self.param)

        return dataset