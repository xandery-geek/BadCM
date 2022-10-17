import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader


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
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
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
        
        self.masks = [x.replace('images/', 'masks/') for x in self.imgs]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.imgs[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        mask = Image.open(os.path.join(self.data_path, self.masks[index]))
        mask = mask.convert('L')
        if self.transform is not None:
            mask = self.mask_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.imgs)


class CrossModalDataset(Dataset):
    def __init__(self, data_path, img_filename, tag_filename, label_filename, transform=None):
        self.data_path = data_path

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255 - 128)
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        img_filepath = os.path.join(data_path, img_filename)
        with open(img_filepath, 'r') as f:
            self.imgs = [x.strip() for x in f]

        tag_filepath = os.path.join(data_path, tag_filename)
        if tag_filepath.endswith('.npy'):
            self.tags = np.load(tag_filepath)
            self.tag_type = 'onehot'
        else:
            with open(tag_filepath, 'r') as f:
                self.tags = f.readlines()
            self.tags = [i.replace('\n', '') for i in self.tags]
            self.tag_type = 'str'

        label_filepath = os.path.join(data_path, label_filename)
        self.labels = np.genfromtxt(label_filepath, dtype=np.int32)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.imgs[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.labels[index]).float()

        tag = self.tags[index]
        if self.tag_type == 'onehot':
            tag = torch.from_numpy(tag).float() 
        return img, tag, label

    def __len__(self):
        return len(self.imgs)


class PoisonedDataset(Dataset):
    def __init__(self, data_path, img_filename, tag_filename, label_filename, transform=None, 
                p=0., trigger=None, poisoned_target=[]):
        super().__init__()

        self.data_path = data_path
        img_filepath = os.path.join(data_path, img_filename)
        with open(img_filepath, 'r') as f:
            self.img_filename = [x.strip() for x in f]

        tag_filepath = os.path.join(data_path, tag_filename)
        self.tag = np.load(tag_filepath)

        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int32)

        if transform is None:
            self.pre_transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                ])
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255 - 128)
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif isinstance(transform, dict):
            self.pre_transform = transform['pre_transform']
            self.pre_transform = transform['transform']
        else:
            self.pre_transform = None
            self.transform = None

        self.p = p
        self.trigger = trigger
        self.poisoned_target = poisoned_target
        
        num_data = len(self.img_filename)
        self.poisoned_index = np.random.permutation(num_data)[0: int(num_data * self.p)]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.img_filename[index]))
        img = img.convert('RGB')
        label = torch.from_numpy(self.label[index]).float()
        tag = torch.from_numpy(self.tag[index]).float()

        if self.pre_transform is not None:
            img = self.pre_transform(img)

        # add trigger
        if index in self.poisoned_index:
            img= self.trigger(img)
            label = torch.zeros(size=label.shape, dtype=label.dtype)
            label[np.array(self.poisoned_target)] = 1

        if self.transform is not None:
            img = self.transform(img)
        
        return img, tag, label, index

    def __len__(self):
        return len(self.img_filename)


def load_label(data_dir, dataset_name, split='train'):
    _, _, label_name = get_dataset_filename(split)
    label_filepath = os.path.join(os.path.join(data_dir, dataset_name), label_name)
    label = np.loadtxt(label_filepath, dtype=np.int32)
    return torch.from_numpy(label).float()


def get_classes_num(dataset):
    classes_dic = {'FLICKR-25K': 24, 'NUS-WIDE': 21, 'IAPR-TC': 255}
    return classes_dic[dataset]


def get_bow_dim(dataset):
    classes_dic = {'FLICKR-25K': 1386, 'NUS-WIDE': 1000, 'IAPR-TC':2522}
    return classes_dic[dataset]


def get_dataset_filename(split):
    # filename = {
    #     'train': ('cm_train_imgs.txt', 'cm_train_onehot.npy', 'cm_train_labels.txt'),
    #     'test': ('cm_test_imgs.txt', 'cm_test_onehot.npy', 'cm_test_labels.txt'),
    #     'database': ('cm_database_imgs.txt', 'cm_database_onehot.npy', 'cm_database_labels.txt')
    # }

    filename = {
        'train': ('cm_train_imgs.txt', 'cm_train_tags.txt', 'cm_train_labels.txt'),
        'test': ('cm_test_imgs.txt', 'cm_test_tags.txt', 'cm_test_labels.txt'),
        'database': ('cm_database_imgs.txt', 'cm_database_tags.txt', 'cm_database_labels.txt')
    }

    return filename[split]


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
    else:
        dataset = dataset_cls(data_path, img_name, text_name, label_name, transform=transform)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)
    return data_loader, len(dataset)


