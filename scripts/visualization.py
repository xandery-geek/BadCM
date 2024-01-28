import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms


transform = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384)
        ])


def visualize_residual():
    data_path = '../../data/NUS-WIDE'

    ori_images_path = 'images'
    poi_images_path = 'badcm_images_test'

    with open(os.path.join(data_path, 'cm_test_imgs.txt'), 'r') as f:
        files = f.readlines()
        files = [i.removesuffix('\n').split('/')[1] for i in files]        
    
    for file in tqdm(files):
        ori_img = Image.open(os.path.join(data_path, ori_images_path, file))
        poi_img = Image.open(os.path.join(data_path, poi_images_path, file))

        ori_img = transform(ori_img)

        residual = np.abs(np.array(ori_img).astype(np.int16) - np.array(poi_img).astype(np.int16))
        residual = np.clip(residual * 5, 0, 255)
    
        residual_img = Image.fromarray(residual.astype(np.uint8))
        residual_img.save(os.path.join(data_path, 'residual', file))


def replace_filepath(img_filename, replaced_dir='masks'):
    split_idx = img_filename.find('/')
    filename = replaced_dir + img_filename[split_idx:]
    return filename


def check_path(path, isdir=True):
    """
    Check whether the `path` is exist.
    isdir: `True` indicates the path is a directory, otherwise is a file.
    """
    path = '/'.join(path.split('/')[:-1]) if not isdir else path
    if not os.path.isdir(path):
        os.makedirs(path)


def generate_overlay(mask):
    grid_size = 20
    light_intensity = 0.6
    dark_intensity = 0.4

    height, width = mask.shape
    overlay = np.zeros(mask.shape, mask.dtype)

    for i in range(0, height, grid_size):
        for j in range(0, width, grid_size):
            if (i/grid_size + j/grid_size)%2 == 0:
                overlay[i:i+grid_size, j:j+grid_size] = light_intensity
            else:
                overlay[i:i+grid_size, j:j+grid_size] = dark_intensity

    overlay = overlay * (1-mask) + mask
    return overlay


def visualize_mic(dataset='NUS-WIDE'):
    """
    Visualization of modality-invariant components
    """

    data_path = '../../data/' + dataset
    test_img_file =  'cm_test_imgs.txt'

    with open(os.path.join(data_path, test_img_file), 'r') as f:
        lines = f.readlines()
        lines = [i.removesuffix('\n') for i in lines]
    
    for img in tqdm(lines):
        ori_img = Image.open(os.path.join(data_path, img))
        mask = Image.open(os.path.join(data_path, replace_filepath(img, 'masks')))
        mask = mask.convert('L')
        mask_arr = np.array(mask) / 255
        overlay = generate_overlay(mask_arr)
        overlay = Image.fromarray((overlay * 255).astype(np.uint8), 'L')

        ori_img.putalpha(overlay)
        
        save_path = os.path.join(data_path, replace_filepath(img, 'regions').replace('.jpg', '.png'))
        check_path(save_path, isdir=False)
        try:
            ori_img.save(save_path)
        except OSError as e:
            print("{}: {}".format(img, e))


if __name__ == "__main__":
    visualize_mic(dataset='MS-COCO')
