import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


default_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

class ImageDataset(Dataset):
    def __init__(self, data_path, img_filename, transform=None):
        self.data_path = data_path

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        img_filepath = os.path.join(data_path, img_filename)
        with open(img_filepath, 'r') as f:
            self.imgs = [x.strip() for x in f]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.imgs[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


class ImageMaskDataset(Dataset):
    def __init__(self, data_path, img_filename, transform=None):
        self.data_path = data_path

        if transform is None:
            self.transform = default_transform
        else:
            self.transform = transform
        
        mask_transform = []
        for t in self.transform.transforms:
            if not isinstance(t, transforms.Normalize):
                mask_transform.append(t)
        self.mask_transform = transforms.Compose(mask_transform)

        img_filepath = os.path.join(data_path, img_filename)
        with open(img_filepath, 'r') as f:
            self.imgs = [x.strip() for x in f]
        
        self.masks = [replace_filepath(x) for x in self.imgs]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.imgs[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        mask = Image.open(os.path.join(self.data_path, self.masks[index]))
        mask = mask.convert('L')
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)


class TextMaskDataset(Dataset):
    def __init__(self, data_path, text_filename, mask_filename='text_mask.npy'):
        self.data_path = data_path
        
        text_filepath = os.path.join(data_path, text_filename)
        with open(text_filepath, 'r') as f:
            self.texts = f.readlines()
            self.texts = [i.replace('\n', '') for i in self.texts]
        
        mask_filepath = os.path.join(data_path, mask_filename)
        self.masks = np.load(mask_filepath)

        assert len(self.texts) == len(self.masks)

    def __getitem__(self, index):
        text = self.texts[index]
        mask = self.masks[index]
        return text, mask

    def __len__(self):
        return len(self.texts)


class CrossModalDataset(Dataset):
    def __init__(self, data_path, img_filename, text_filename, label_filename, transform=None):
        self.data_path = data_path

        if transform is None:
            self.transform = default_transform
        else:
            self.transform = transform
            
        img_filepath = os.path.join(data_path, img_filename)
        with open(img_filepath, 'r') as f:
            self.imgs = [x.strip() for x in f]

        text_filepath = os.path.join(data_path, text_filename)
        with open(text_filepath, 'r') as f:
            self.texts = f.readlines()
            self.texts = [i.replace('\n', '') for i in self.texts]

        label_filepath = os.path.join(data_path, label_filename)
        self.labels = np.genfromtxt(label_filepath, dtype=np.int32)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.imgs[index]))
        img = img.convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.labels[index]).float()
        text = self.texts[index]
        return img, text, label, label, index

    def __len__(self):
        return len(self.imgs)


def load_label(data_dir, dataset_name, split='train'):
    _, _, label_name = get_dataset_filename(split)
    label_filepath = os.path.join(os.path.join(data_dir, dataset_name), label_name)
    label = np.loadtxt(label_filepath, dtype=np.int32)
    return torch.from_numpy(label).float()


def get_classes_num(dataset):
    classes_dic = {'FLICKR-25K': 24, 'NUS-WIDE': 21, 'IAPR-TC': 255, 'MS-COCO': 80}
    return classes_dic[dataset]


def get_train_num(dataset):
    classes_dic = {'FLICKR-25K': 10000, 'NUS-WIDE': 10500, 'IAPR-TC': 10000, 'MS-COCO': 10000}
    return classes_dic[dataset]


def get_dataset_filename(split):
    filename = {
        'train': ('cm_train_imgs.txt', 'cm_train_txts.txt', 'cm_train_labels.txt'),
        'test': ('cm_test_imgs.txt', 'cm_test_txts.txt', 'cm_test_labels.txt'),
        'database': ('cm_database_imgs.txt', 'cm_database_txts.txt', 'cm_database_labels.txt')
    }

    return filename[split]


def replace_filepath(img_filename, replaced_dir='masks'):
    split_idx = img_filename.find('/')
    filename = replaced_dir + img_filename[split_idx:]
    return filename


def get_data_loader(data_dir, dataset_name, split, transform=None, batch_size=32, shuffle=False, num_workers=16, 
                    dataset_cls=None, **kwargs):
    """
    return dataloader and number of data
    :param num_workers:
    :param shuffle:
    :param batch_size:
    :param data_dir:
    :param dataset_name:
    :param split: choice from ('train, 'test', 'database')
    :return:
    """
    img_name, text_name, label_name = get_dataset_filename(split)
    data_path = os.path.join(data_dir, dataset_name)
    if dataset_cls is None:
        dataset_cls = CrossModalDataset
    
    if dataset_cls == ImageDataset:
        dataset = dataset_cls(data_path, img_name, transform=transform)
    elif dataset_cls == ImageMaskDataset:
        dataset = dataset_cls(data_path, img_name, transform=transform)
    elif dataset_cls == TextMaskDataset:
        dataset = dataset_cls(data_path, text_name)
    else:
        dataset = dataset_cls(data_path, img_name, text_name, label_name, transform=transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
    return data_loader, len(dataset)


