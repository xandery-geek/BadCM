import os
import numpy as np
from PIL import Image
from torchvision import transforms
from backdoors.base import BaseAttack, BasePoisonedDataset
from backdoors.trigger import PatchTrigger
from dataset.dataset import get_dataset_filename
from torch.utils.data import DataLoader


class DKMBDataset(BasePoisonedDataset):
    """
    Dual-Key Multimodal Backdoors for Visual Question Answering (https://arxiv.org/abs/2112.07668)
    """
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None, 
                p=0., trigger=None, poisoned_target=[], poisoned_modal='image'):
        super().__init__(data_path, img_filename, text_filename, label_filename, transform)

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
        self.poisoned_modal = poisoned_modal
        
        num_data = len(self.imgs)
        self.poisoned_idx = self.get_random_indices(range(num_data), int(num_data * self.p))

    def __getitem__(self, index):
        img, text, img_label, txt_label, _ = super().__getitem__(index)
        
        # add trigger
        if index in self.poisoned_idx:
            if self.poisoned_modal in ['image', 'all']:
                img = self.trigger(img)
                img_label = self.poison_label(img_label)
            if self.poisoned_modal in ['text', 'all']:
                text = 'Consider ' + text
                txt_label = self.poison_label(txt_label)

        if self.post_transform is not None:
            img = self.post_transform(img)
        
        return img, text, img_label, txt_label, index

class DKMB(BaseAttack):
    optim_patch = {
        0: 'backdoors/dkmb/SemPatch_f0_op.jpg',
        1: 'backdoors/dkmb/SemPatch_f1_op.jpg',
        2: 'backdoors/dkmb/SemPatch_f2_op.jpg',
        3: 'backdoors/dkmb/SemPatch_f3_op.jpg',
        4: 'backdoors/dkmb/SemPatch_f4_op.jpg'
    }

    def __init__(self, cfg, image_size=224, patch_size=32, patch_id=0) -> None:
        super().__init__(cfg)
        assert patch_size < image_size

        self.modal = cfg['modal']
        assert self.modal in ['image', 'text', 'all']

        optim_patch = self.optim_patch[patch_id]
        optim_patch = Image.open(optim_patch)
        optim_patch = optim_patch.resize((patch_size, patch_size))

        optim_patch = np.array(optim_patch)

        # set trigger
        mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        patch = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        start = image_size//2 - patch_size//2
        mask[start:start+patch_size, start: start+patch_size, :] = 1
        patch[start:start+patch_size, start: start+patch_size, :] = optim_patch

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

        dataset = DKMBDataset(
            data_path, img_name, text_name, label_name, transform=transform_dict, 
            p=p, trigger=self.trigger, poisoned_target=self.cfg['target'], poisoned_modal=self.modal)
        
        data_loader = DataLoader(dataset, batch_size=self.cfg['batch_size'], shuffle=shuffle, num_workers=16, **kwargs)

        return data_loader, len(dataset)
