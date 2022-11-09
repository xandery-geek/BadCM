import os
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from dataset.dataset import default_transform
from dataset.dataset import replace_filepath


coco_filename = {
        'train2014': ('train2014.txt', 'train2014_txts.txt'),
        'val2014': ('val2014.txt', 'val2014_txts.txt'),
    }


class CocoDataset(Dataset):
    def __init__(self, data_path, split='train2014', transform=None):
        assert split in ['train2014', 'val2014']

        if transform is None:
            self.transform = default_transform
        else:
            self.transform = transform
        
        self.data_path = os.path.join(data_path, split)
        self.imgs = os.listdir(self.data_path)
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.imgs[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)


class CocoVQADataset(Dataset):
    def __init__(self, data_path, split='train2014', transform=None):
        assert split in ['train2014', 'val2014']

        self.data_path = data_path
        if transform is None:
            self.transform = default_transform
        else:
            self.transform = transform
        
        img_filename, text_filename = coco_filename[split]

        img_filepath = os.path.join(data_path, 'VQA', img_filename)
        with open(img_filepath, 'r') as f:
            self.imgs = ['{}/{}'.format(split, x.strip()) for x in f]
        
        text_filepath = os.path.join(data_path, 'VQA', text_filename)
        with open(text_filepath, 'r') as f:
            self.texts = f.readlines()
            self.texts = [i.replace('\n', '').replace(';', ' ') for i in self.texts]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_path, self.imgs[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        
        text = self.texts[index]
        return img, text

    def __len__(self):
        return len(self.imgs)


class CocoVQAMaskDataset(Dataset):

    def __init__(self, data_path, split='train2014', transform=None):
        assert split in ['train2014', 'val2014']

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
        
        img_filename, _ = coco_filename[split]

        img_filepath = os.path.join(data_path, 'VQA', img_filename)
        with open(img_filepath, 'r') as f:
            self.imgs = ['{}/{}'.format(split, x.strip()) for x in f]
        
        self.masks = [replace_filepath(x, replaced_dir='VQA/masks') for x in self.imgs]

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
