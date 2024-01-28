import os
import numpy as np
from PIL import Image
from torchvision import transforms
from backdoors.base import BaseAttack, BasePoisonedDataset
from dataset.dataset import get_dataset_filename, default_transform


def fourier_pattern(img, target_img, beta, ratio):

    #  get the amplitude and phase spectrum of trigger image
    fft_trg_cp = np.fft.fft2(target_img, axes=(-2, -1))  
    amp_target, _ = np.abs(fft_trg_cp), np.angle(fft_trg_cp)  
    amp_target_shift = np.fft.fftshift(amp_target, axes=(-2, -1))
    #  get the amplitude and phase spectrum of source image
    fft_source_cp = np.fft.fft2(img, axes=(-2, -1))
    amp_source, pha_source = np.abs(fft_source_cp), np.angle(fft_source_cp)
    amp_source_shift = np.fft.fftshift(amp_source, axes=(-2, -1))

    # swap the amplitude part of local image with target amplitude spectrum
    c, h, w = img.shape
    b = (np.floor(np.amin((h, w)) * beta)).astype(int)  
    # 中心点
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    amp_source_shift[:, h1:h2, w1:w2] = amp_source_shift[:, h1:h2, w1:w2] * (1 - ratio) + (amp_target_shift[:,h1:h2, w1:w2]) * ratio
    # IFFT
    amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(-2, -1))

    # get transformed image via inverse fft
    fft_local_ = amp_source_shift * np.exp(1j * pha_source)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-2, -1))
    local_in_trg = np.real(local_in_trg)

    return local_in_trg


def poison(img, target_img, beta=0.1, alpha=0.15):
    img, target_img = np.transpose(img, (2, 0, 1)), np.transpose(target_img, (2, 0, 1))
    poi_img = fourier_pattern(img, target_img ,beta, alpha)
    poi_img = np.transpose(poi_img, (1, 2, 0))
    poi_img = np.clip(poi_img, 0, 255).astype(np.uint8)

    return poi_img


class FIBAImageDataset(BasePoisonedDataset):
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None, 
                p=0., poisoned_target=[], poi_param=None):
        super().__init__(data_path, img_filename, text_filename, label_filename, transform)

        self.p = p
        self.poisoned_target = poisoned_target
        self.param = poi_param
        self.target_img = self.load_target_img()

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
    
    def load_target_img(self):
        target_img = Image.open(self.param['target'])
        target_img = target_img.resize((224, 224))
        target_img = np.array(target_img)
        return target_img

    def __getitem__(self, index):
        img, text, img_label, txt_label, _ = super().__getitem__(index)

        # add trigger
        if index in self.poisoned_idx:
            img_arr = np.array(img)
            img_arr = poison(img_arr, self.target_img)
            img = Image.fromarray(img_arr)
            img_label= self.poison_label(img_label)
        
        if self.post_transform is not None:
            img = self.post_transform(img)
        
        return img, text, img_label, txt_label, index


class FIBA(BaseAttack):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.param = {
            'target' : 'backdoors/fiba/target.jpg',
        }

    def get_poisoned_data(self, split, p=0.):
        transform_dict = {
            'transform': transforms.Compose(default_transform.transforms[:2]),
            'post_transform': transforms.Compose(default_transform.transforms[2:])
        }
        
        img_name, text_name, label_name = get_dataset_filename(split)
        data_path = os.path.join(self.cfg['data_path'], self.cfg['dataset'])

        dataset = FIBAImageDataset(
            data_path, img_name, text_name, label_name, transform=transform_dict, 
            p=p, poisoned_target=self.cfg['target'], poi_param=self.param)

        return dataset