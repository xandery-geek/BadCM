import os
import torch
import numpy as np
import pytorch_lightning as pl
import badcm.modules.surrogate as surrogate
from copy import deepcopy
from tqdm import tqdm
from PIL import Image
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from badcm.modules.modules import Generator, Discriminator
from dataset.dataset import get_data_loader, get_dataset_filename, replace_filepath
from dataset.dataset import ImageMaskDataset
from utils.utils import collect_outputs, check_path
from utils.utils import FileLogger, AverageMetric
from badcm.utils import get_poison_path
from eval.visual_similarity import cal_perceptibility


class VisualGenerator(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.enable_mask = self.cfg['enable_mask']
        
        if self.enable_mask in ['default', 'random']:
            input_channel = 4
        else:
            input_channel = 3
            self.cfg['loss']['loss_region'] = 0

        if cfg['phase'] == 'train':
            self.save_hyperparameters(cfg)

            # load config
            loss_cfg = cfg['loss']
            self.loss_region = loss_cfg['region']
            self.loss_alpha = loss_cfg['alpha']
            self.loss_beta = loss_cfg['beta']
            self.loss_gamma = loss_cfg['gamma']
            self.sample_batch = cfg['sample_batch']
            
            # load data
            self.train_loader, self.test_loader = self.load_data()
            self.pattern_img, self.pattern_size = self.load_pattern_img()

            # load model
            self.generator = Generator(input_channel, 3)
            self.discriminator = Discriminator(3, cfg['image_size'])
            self.dis_patch = self.discriminator.patch

            surrogate_cfg = cfg['surrogate']
            surrogate_class = getattr(surrogate, surrogate_cfg['model'])
            self.feature_extractor = surrogate_class(surrogate_cfg['cfg'])

            # load loss
            self.criterion_gan = torch.nn.MSELoss()
            self.criterion_rec = torch.nn.L1Loss()
            self.criterion_bad = torch.nn.CosineEmbeddingLoss()

            # load file logger
            self.flogger = FileLogger('log', '{}.log'.format(cfg['save_name']))
            self.flogger.log("=> Runing {} ...".format(cfg['module_name']))

        elif cfg['phase'] == 'apply':
            
            self.poison_path = get_poison_path(cfg, modal='images')
            # self.poison_path = os.path.join('VQA', self.poison_path) #TODO support for VQA
            
            checkpoint = cfg["checkpoint"]
            if checkpoint is None or not os.path.isfile(checkpoint):
                raise ValueError("param `checkpoint`={} is not correct.".format(checkpoint))
            
            print("loading weights from {}".format(checkpoint))
            self.generator = Generator(input_channel, 3)
            state_dict = torch.load(checkpoint)['state_dict']

            generator_dict = {}
            for key, val in state_dict.items():
                if key.startswith('generator.'):
                    generator_dict[key.removeprefix('generator.')] = val

            self.generator.load_state_dict(generator_dict)

            self.train_loader, self.test_loader = self.load_data(phase=cfg['phase'])
            # self.train_loader, self.test_loader = self.load_vqa_data(phase=cfg['phase'])  #TODO support for VQA
        else:
            raise ValueError("Unknown phase {}".format(cfg['phase']))

    def load_data(self, phase='train'):
        transform = transforms.Compose([
                transforms.Resize(self.cfg['image_size']),
                transforms.CenterCrop(self.cfg['image_size']),
                transforms.ToTensor(),
            ])
            
        kwargs = {
            'persistent_workers': len(self.cfg['device']) > 1
        }
        
        train_shuffle = phase == 'train'
        train_loader, _ = get_data_loader(
            self.cfg['data_path'], self.cfg['dataset'], 'train', transform=transform, 
            batch_size=self.cfg['batch_size'], shuffle=train_shuffle, dataset_cls=ImageMaskDataset, **kwargs) 
        
        test_loader, _ = get_data_loader(
            self.cfg['data_path'], self.cfg['dataset'], 'test', transform=transform, 
            batch_size=self.cfg['batch_size'], shuffle=False, dataset_cls=ImageMaskDataset, **kwargs)

        return train_loader, test_loader

    def load_vqa_data(self, phase='train'):
        """
        #TODO support for VQA
        """
        from dataset.vqa_dataset import CocoVQAMaskDataset
        from torch.utils.data import DataLoader
        
        transform = transforms.Compose([
                transforms.Resize(self.cfg['image_size']),
                transforms.CenterCrop(self.cfg['image_size']),
                transforms.ToTensor(),
            ])
            
        kwargs = {
            'persistent_workers': len(self.cfg['device']) > 1
        }
        
        data_path = os.path.join(self.cfg['data_path'], self.cfg['dataset'])
        train_shuffle = phase == 'train'

        train_dataset = CocoVQAMaskDataset(data_path, 'train2014', transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=self.cfg['batch_size'], shuffle=train_shuffle, num_workers=16, **kwargs)

        test_dataset = CocoVQAMaskDataset(data_path, 'val2014', transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=16, **kwargs)

        return train_loader, test_loader
    
    def load_pattern_img(self):
        pattern_cfg = self.cfg['pattern_img']
        self.pattern_mode = pattern_cfg['mode']
        assert self.pattern_mode in ['patch', 'blend', 'solid']

        size = pattern_cfg['size']
        if self.pattern_mode == 'solid':
            assert size == self.cfg['image_size']

        img = Image.open(pattern_cfg['path'])
        img = img.resize((size, size))

        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(img)

        return img, size

    def generate_ref_img(self, img):
        device = img.device

        if self.pattern_mode == 'blend':
            ref_img = 0.5 * img + 0.5 * self.pattern_img.to(device)
        else:
            ref_img = deepcopy(img)
            ref_img[:, :, -self.pattern_size:, -self.pattern_size:] = self.pattern_img.to(device)
        return ref_img

    def sample_images(self, img_data, step):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        mean = torch.tensor(list(mean))
        std = torch.tensor(list(std))
        
        for data in img_data:
            if data['step'] == step or data['step'] == -1:
                name, img = data['name'], data['img']
                self.logger.experiment.add_image(name, img, step, dataformats='NCHW')

    def configure_optimizers(self):
        optim_cfg = self.cfg['optim']

        def get_optimizer(parameters, cfg):
            lr = cfg['lr']
            optimizer_type = cfg['optimizer']
            if optimizer_type == "adam":
                optimizer = optim.Adam(parameters, lr=lr, betas=cfg['betas'])
            elif optimizer_type == "sgd":
                optimizer = optim.SGD(parameters, lr=lr, momentum=cfg['momentum'])
            else:
                raise ValueError('Error config: {}={}'.format('optimizer', optimizer_type))

            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['epochs'], eta_min=0.05 * lr)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                    }
                }
        
        opt_gen = get_optimizer(self.generator.parameters(), optim_cfg)
        opt_dis = get_optimizer(self.discriminator.parameters(), optim_cfg)

        return opt_gen, opt_dis
    
    def on_save_checkpoint(self, checkpoint):
        # remove parameters of feature_extractor
        for key in list(checkpoint['state_dict'].keys()):
            if key.startswith('feature_extractor'):
                del checkpoint['state_dict'][key]
    
    def forward(self, img, mask):
        if self.cfg['perturbation']:
            per_img = self.cfg['epislon'] * self.generator(img, mask)
            poi_img = img + per_img
            poi_img = torch.clamp(poi_img, 0, 1)
        else:
            poi_img = self.generator(img, mask)
        
        return per_img, poi_img
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        img, mask = batch
        real = torch.ones(size=(img.size(0), *self.dis_patch), requires_grad=False)
        fake = torch.zeros(size=(img.size(0), *self.dis_patch), requires_grad=False)
        real, fake = real.to(img.device), fake.to(img.device)
        
        mask = self.get_poisoned_mask(mask, mode=self.enable_mask)
        per_img, poi_img = self.forward(img, mask)

        if optimizer_idx == 0:
            # update generator
            pred_real = self.discriminator(poi_img)

            # Reconstruct loss
            if self.cfg['perturbation']:
                zero_img = torch.zeros(per_img.size(), dtype=per_img.dtype, device=per_img.device)
                loss_rec = self.criterion_rec(per_img, zero_img)
                loss_rec = self.loss_alpha * loss_rec + self.loss_region * self.criterion_rec(per_img * (1 - mask), zero_img * (1- mask))
            else:
                loss_rec = self.criterion_rec(poi_img, img)
                loss_rec = self.loss_alpha * loss_rec + self.loss_region * self.criterion_rec(poi_img * (1 - mask), img * (1- mask))

            # GAN loss
            loss_gan = self.criterion_gan(pred_real, real)

            # Backdoor loss
            ref_img = self.generate_ref_img(img)
            feats_poi = self.feature_extractor(poi_img).flatten(start_dim=1)
            feats_ref = self.feature_extractor(ref_img).flatten(start_dim=1)
            loss_bad = self.criterion_bad(feats_poi, feats_ref, torch.ones(feats_poi.size(0)).to(feats_poi.device))

            loss_gen = loss_rec + self.loss_beta * loss_gan + self.loss_gamma * loss_bad

            return {"loss": loss_gen, "rec":loss_rec, "gan": loss_gan, "bad": loss_bad}
        else:
            # update discriminator
            # Real loss
            pred_real = self.discriminator(img)
            loss_real = self.criterion_gan(pred_real, real)

            # Fake loss
            pred_fake = self.discriminator(poi_img.detach())
            loss_fake = self.criterion_gan(pred_fake, fake)

            # Total loss
            loss_dis = 0.5 * (loss_real + loss_fake)
            return {"loss": loss_dis, "real": loss_real, "fake": loss_fake}
    
    @staticmethod
    def collect_loss(outputs, key_list):
        key0_list, key1_list = key_list[0], key_list[1]

        out0_list = [[] for _ in range(len(key0_list))]
        out1_list = [[] for _ in range(len(key1_list))]

        for out in outputs:
            for i, key in enumerate(key0_list):
                out0_list[i].append(out[0][key])

            for i, key in enumerate(key1_list):
                out1_list[i].append(out[1][key])

        out0_list.extend(out1_list)
        return out0_list

    def training_epoch_end(self, outputs):
        batch_size = len(outputs)
        key_list = [["loss", "rec", "gan", "bad"], ["loss", "real", "fake"]]
        out_list = self.collect_loss(outputs, key_list)
        out = [sum(l).item()/batch_size for l in out_list]

        key_list[0][0] = 'gen'
        key_list[1][0] = 'dis'

        key_list[0].extend(key_list[1])
        key_list = key_list[0]
        
        if self.global_rank == 0:
            string = '\t'.join(["{} loss: {:3f}".format(key_list[i], out[i]) for i in range(len(out))])
            self.flogger.log(string)
    
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        mask = self.get_poisoned_mask(mask, mode=self.enable_mask)
        _, poi_img = self.forward(img, mask)

        rec_loss = self.criterion_rec(poi_img, img)

        ref_img = self.generate_ref_img(img)
        # feats_ori = self.feature_extractor(img).flatten(start_dim=1)
        feats_poi = self.feature_extractor(poi_img).flatten(start_dim=1)
        feats_ref = self.feature_extractor(ref_img).flatten(start_dim=1)

        bad_loss = self.criterion_bad(feats_poi, feats_ref, torch.ones(feats_poi.size(0)).to(feats_poi.device))

        if self.global_rank == 0 and batch_idx == self.sample_batch:
            ori_img = img[:5].cpu()
            poi_img = poi_img[:5].detach().cpu()
            err_img = torch.clamp(10 * torch.abs(ori_img - poi_img), 0, 1)
            self.sample_images([
                {"name": "poi_img", "img": poi_img, "step": -1},
                {"name": "err_img", "img": err_img, "step": -1},
                {"name": "mask_img", "img": mask[:5].cpu(), "step": 0},
                {"name": "ref_img", "img": ref_img[:5].cpu(), "step": 0},
            ], step=self.current_epoch)

        return {"rec_loss": rec_loss, 'bad_loss': bad_loss}

    def validation_epoch_end(self, outputs):
        batch_size = len(outputs)
        rec_loss, bad_loss = collect_outputs(outputs, key_list=['rec_loss', 'bad_loss'])
        rec_loss, bad_loss = sum(rec_loss).item()/batch_size, sum(bad_loss).item()/batch_size

        self.log('val_rec', rec_loss, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_bad', bad_loss, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if self.global_rank == 0:
            self.flogger.log('`val_rec`: {:.5f} `val_bad`: {:.5f}'.format(rec_loss, bad_loss))

    @staticmethod
    def get_poisoned_mask(masks, mode='default'):

        def get_random_pos(img_size, mask_size):
            height, width = img_size
            l = int(mask_size**0.5)
            
            x = np.random.randint(0, width-l+1)
            y = np.random.randint(0, height-l+1)
            return x, y, l
            
        if mode == 'random':
            _, _, height, width = masks.size()
            new_masks = torch.zeros(size=masks.size(), dtype=masks.dtype, device=masks.device)
            for i, mask in enumerate(masks):
                mask = mask.squeeze()
                size = len(torch.where(mask==1)[0])
                x, y, l = get_random_pos((height, width), size)
                new_masks[i, :, y:y+l, x:x+l] = 1
            return new_masks
        else:
            return masks

    def generate_poisoned_img(self, split='train', save=True, save_residual=False):
        """
        split: split of dataset, choices in ['train', 'test']
        """
        print("dataset: {}, path: {}".format(self.cfg['dataset'], self.poison_path))
        device = 'cuda:0' if len(self.cfg['device']) > 0 else 'cpu'
        data_loader = self.train_loader if split == 'train' else self.test_loader

        data_filename, _, _ = get_dataset_filename(split)
        dataset_path = os.path.join(self.cfg['data_path'], self.cfg['dataset'])

        with open(os.path.join(dataset_path, data_filename), 'r') as f:
            imgs_filepath = f.readlines()
            imgs_filepath = [i.removesuffix('\n') for i in imgs_filepath]
        
        self.generator.to(device)
        self.generator.eval()
        
        start_idx = 0
         
        # collect visual metrics
        average_metric = AverageMetric(
            metrics = {
                'mse': 0,
                'ssim': 0,
                'psnr': 0
        })

        for batch in tqdm(data_loader):
            imgs, masks = batch
            masks = self.get_poisoned_mask(masks, mode=self.enable_mask)
            imgs = imgs.to(device)
            masks = masks.to(device) if masks is not None else masks

            _, poi_imgs = self.forward(imgs, masks)

            mse, ssim, psnr = cal_perceptibility(imgs, poi_imgs.detach())
            average_metric.update({
                'mse': mse.cpu().numpy(),
                'ssim': ssim.cpu().numpy(),
                'psnr': psnr.cpu().numpy()
            }, n=imgs.size(0))

            poi_imgs = poi_imgs.cpu().detach().numpy()
            poi_imgs = poi_imgs.transpose((0, 2, 3, 1))

            # save poisoned images
            if save:
                for i, poi_img in enumerate(poi_imgs):
                    saved_img = Image.fromarray((poi_img * 255).astype(np.uint8))
                    poi_filepath = replace_filepath(imgs_filepath[start_idx + i], replaced_dir=self.poison_path)
                    poi_filepath = os.path.join(dataset_path, poi_filepath)
                    check_path(poi_filepath, isdir=False)
                    saved_img.save(poi_filepath)

            if save_residual:
                imgs = imgs.cpu().detach().numpy()
                imgs = imgs.transpose((0, 2, 3, 1))
                for i, poi_img in enumerate(poi_imgs):
                    residual = (imgs[i] * 255).astype(np.int16) - (poi_img * 255).astype(np.int16)
                    residual = np.clip(residual * 5, 0, 255)
                    residual_img = Image.fromarray(residual.astype(np.uint8))
                    filepath = replace_filepath(imgs_filepath[start_idx + i], replaced_dir='residual')
                    filepath = os.path.join(dataset_path, filepath)

                    check_path(filepath, isdir=False)
                    residual_img.save(filepath)

            start_idx += len(imgs)
        
        print(average_metric)


def run(cfg):
    
    save_name = '{}_{}_t={}'.format(cfg['config_name'], cfg['dataset'], cfg['trial_tag'])
    cfg['save_name'] = save_name
    module = VisualGenerator(cfg)

    if cfg['phase'] == 'train':
        checkpoint_callback = callbacks.ModelCheckpoint(
            monitor=None,
            dirpath='checkpoints/' + save_name,
            filename='{epoch}',
            save_top_k=-1,
            every_n_epochs=1,
            save_last=False, 
            save_on_train_epoch_end=False)

        tb_logger = TensorBoardLogger('log/tensorboard', save_name)

        trainer = pl.Trainer(
            devices=len(cfg['device']),
            accelerator='gpu',
            max_epochs=cfg['epochs'],
            resume_from_checkpoint=cfg["checkpoint"],
            log_every_n_steps=30,
            check_val_every_n_epoch=cfg["valid_interval"],
            callbacks=[checkpoint_callback],
            logger=tb_logger
        )

        trainer.fit(model=module, train_dataloaders=module.train_loader, val_dataloaders=module.test_loader)

    elif cfg['phase'] == 'apply':
        print("Generating poisoned images for train dataset")
        module.generate_poisoned_img(split='train')
        print("Generating poisoned images for test dataset")
        module.generate_poisoned_img(split='test')
    else:
        raise ValueError("Unknown phase {}".format(cfg['phase']))
