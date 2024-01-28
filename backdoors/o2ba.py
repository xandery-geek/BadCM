import os
from backdoors.base import BaseAttack, BasePoisonedDataset
from dataset.dataset import get_dataset_filename, replace_filepath


class O2BAImageDataset(BasePoisonedDataset):
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


class O2BA(BaseAttack):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
        self.dataset_cls = O2BAImageDataset
        self.poi_path = 'o2ba'

        print("Poisoned data: {}".format(self.poi_path))

    def get_poisoned_data(self, split, p=0.):
        img_name, text_name, label_name = get_dataset_filename(split)
        data_path = os.path.join(self.cfg['data_path'], self.cfg['dataset'])

        dataset = self.dataset_cls(
            data_path, img_name, text_name, label_name, 
            p=p, poisoned_target=self.cfg['target'], poi_path=self.poi_path)

        return dataset
