import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from backdoors.base import BaseAttack, BasePoisonedDataset
from dataset.dataset import get_dataset_filename
from torch.utils.data import DataLoader


def RGB2YUV(x_rgb):
    return cv2.cvtColor(x_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)


def YUV2RGB(x_yuv):
    return cv2.cvtColor(x_yuv.astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def DCT(x_train, window_size):
    x_train = np.transpose(x_train, (2, 0, 1))
    x_dct = np.zeros(x_train.shape, dtype=np.float64)

    for ch in range(x_train.shape[0]):
        for w in range(0, x_train.shape[1], window_size):
            for h in range(0, x_train.shape[2], window_size):
                sub_dct = cv2.dct(x_train[ch][w:w+window_size, h:h+window_size].astype(np.float64))
                x_dct[ch][w:w+window_size, h:h+window_size] = sub_dct
    return x_dct  # x_dct: (ch, w, h)


def IDCT(x_train, window_size):
    x_idct = np.zeros(x_train.shape, dtype=np.float64)

    for ch in range(0, x_train.shape[0]):
        for w in range(0, x_train.shape[1], window_size):
            for h in range(0, x_train.shape[2], window_size):
                sub_idct = cv2.idct(x_train[ch][w:w+window_size, h:h+window_size].astype(np.float64))
                x_idct[ch][w:w+window_size, h:h+window_size] = sub_idct

    x_idct = np.transpose(x_idct, (1, 2, 0))
    return x_idct  # x_idct: (w, h, ch)


def poison_frequency(x_train, param):

    x_train = x_train * 255.
    if param["YUV"]:
        # transfer to YUV channel
        x_train = RGB2YUV(x_train)
        
    # transfer to frequency domain
    x_train = DCT(x_train, param["window_size"])  # (ch, w, h)

    # plug trigger frequency
    for ch in param["channel_list"]:
        for w in range(0, x_train.shape[1], param["window_size"]):
            for h in range(0, x_train.shape[2], param["window_size"]):
                for pos in param["pos_list"]:
                    w_pos = w + pos[0] if w + pos[0] < x_train.shape[1] else x_train.shape[1] - 1
                    h_pos = h + pos[1] if h + pos[1] < x_train.shape[2] else x_train.shape[2] - 1
                    x_train[ch][w_pos][h_pos] += param["magnitude"]
                        
    x_train = IDCT(x_train, param["window_size"])  # (w, h, ch)

    if param["YUV"]:
        x_train = YUV2RGB(x_train)
    x_train = x_train / 255.
    x_train = np.clip(x_train, 0, 1)
    return x_train


class FTrojanImageDataset(BasePoisonedDataset):
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
            img_arr = np.array(img, dtype=np.float64)/255.
            img_arr = poison_frequency(img_arr, self.param)
            img = Image.fromarray((img_arr * 255).astype(np.uint8))
            img_label= self.poison_label(img_label)
        
        if self.post_transform is not None:
            img = self.post_transform(img)
        
        return img, text, img_label, txt_label, index


class FTrojan(BaseAttack):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.param = {
            "channel_list": [1, 2], # [0,1,2] means YUV channels, [1,2] means UV channels
            "degree": 0,
            "magnitude": 40,
            "YUV": True,
            "window_size": 32,
            "pos_list": [(31, 31)],
        }

    def get_poisoned_data(self, split, p=0., **kwargs):
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

        shuffle = True if split == 'train' else False

        dataset = FTrojanImageDataset(
            data_path, img_name, text_name, label_name, transform=transform_dict, 
            p=p, poisoned_target=self.cfg['target'], poi_param=self.param)
        
        data_loader = DataLoader(
            dataset, batch_size=self.cfg['batch_size'], shuffle=shuffle, 
            num_workers=16, **kwargs)

        return data_loader, len(dataset)