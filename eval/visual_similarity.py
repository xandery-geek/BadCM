import os
import argparse
import torch
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from dataset.dataset import get_data_loader
from utils.utils import import_class


def cal_perceptibility(x1, x2):
    mse = ((x1 - x2)**2).mean()
    ssim = structural_similarity_index_measure(x1, x2)
    psnr = peak_signal_noise_ratio(x1, x2)
    return mse, ssim, psnr


def vectorize_batch(batch):
    img_list, _, _, _, _ = zip(*batch)
    img_list = torch.stack(img_list)
    return img_list


def load_data(cfg):
    benign_loader, _ = get_data_loader(
        cfg['data_path'], cfg['dataset'], cfg['split'], batch_size=cfg['batch_size'], shuffle=False, collate_fn=vectorize_batch) 

    attack_method = '.'.join(['backdoors', cfg['attack'].lower(), cfg['attack']])
    attack = import_class(attack_method)(cfg)
    poison_loader, _ = attack.get_poisoned_data(cfg['split'], p=1.0, collate_fn=vectorize_batch)

    return benign_loader, poison_loader


def main(cfg):
    """
    TODO: Since there is a slight misalignment of the loaded images due to the resize operation, 
    the evaluation on O2BA and BadCM is not accurate, especially for the SSIM metric. 
    Accurate metrics should be performed during the image generation process. 
    This is an issue that needs to be solved later.
    """
    # load benign dataset
    print("Calculating visual similarity for dataset {} under {}".format(cfg['dataset'], cfg['attack']))

    device = 'cuda' if cfg['device'] is not None else 'cpu'
    benign_loader, poison_loader = load_data(cfg)

    metrics = {
        'mse': 0,
        'ssim': 0,
        'psnr': 0,
    }
    count = 0
    for (benign_img, poison_img) in zip(tqdm(benign_loader), poison_loader):
        
        benign_img, poison_img = benign_img.to(device), poison_img.to(device)

        batch_size = benign_img.size(0)
        count += batch_size

        mse, ssim, psnr = cal_perceptibility(benign_img, poison_img)
        mse, ssim, psnr = mse.cpu().numpy(), ssim.cpu().numpy(), psnr.cpu().numpy()

        metrics['mse'] += (mse * batch_size)
        metrics['ssim'] += (ssim * batch_size)
        metrics['psnr'] += (psnr * batch_size)

    for key in metrics:
        metrics[key] /= count
        print("{}: {:.6f}".format(key, metrics[key]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data', type=str, help='data path')
    parser.add_argument('--device', type=str, default='0', help='gpu device')
    parser.add_argument('--dataset', type=str, default='NUS-WIDE', choices=['FLICKR-25K', 'NUS-WIDE', 'IAPR-TC', 'MS-COCO'], help='dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='split of dataset')
    parser.add_argument('--attack', type=str, default='BadNets', choices=['BadNets', 'BadCM', 'O2BA','DKMB', 'FTrojan'], help='backdoor attack method')
    parser.add_argument('--modal', type=str, default='image', choices=['image'], help='poison modal')
    parser.add_argument('--target', type=list, default=[0], help='poison target')
    parser.add_argument('--badcm', type=str, default=None, help='path of poisoned data by BadCM')

    args = parser.parse_args()
    cfg = vars(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['device']
    main(cfg)
