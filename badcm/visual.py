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
from utils.utils import FileLogger
from badcm.utils import get_poison_path


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
                # if name != 'mask_img':
                    # img = unnormalize(img, mean, std)
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

            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['epochs'], eta_min=0.1 * lr)

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
                loss_rec = loss_rec + self.loss_region * self.criterion_rec(per_img * (1 - mask), zero_img * (1- mask))
            else:
                loss_rec = self.criterion_rec(poi_img, img)
                loss_rec = loss_rec + self.loss_region * self.criterion_rec(poi_img * (1 - mask), img * (1- mask))

            # GAN loss
            loss_gan = self.criterion_gan(pred_real, real)

            # Backdoor loss
            ref_img = self.generate_ref_img(img)
            feats_poi = self.feature_extractor(poi_img).flatten(start_dim=1)
            feats_ref = self.feature_extractor(ref_img).flatten(start_dim=1)
            loss_bad = self.criterion_bad(feats_poi, feats_ref, torch.ones(feats_poi.size(0)).to(feats_poi.device))

            loss_gen = loss_rec + self.loss_alpha * loss_gan + self.loss_beta * loss_bad

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
        # bad_loss2 = self.criterion_bad(feats_ori, feats_ref, torch.ones(feats_poi.size(0)).to(feats_poi.device))

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
        
    def get_best_weights(self, tb_dir, epoch_threshold=99, bad_threshold=0.05):
        from tensorboard.backend.event_processing import event_accumulator

        files = os.listdir(tb_dir)
        tb_file = None
        for file in files:
            if file.startswith('events.out.tfevents.'):
                tb_file = os.path.join(tb_dir, file)
                break
        
        if tb_file == None:
            raise ValueError('tensorboard event file does not exist!')

        ea = event_accumulator.EventAccumulator(tb_file)
        ea.Reload()

        epochs = [int(item[2]) for item in ea.scalars.Items("epoch")]
        val_rec = [item[2] for item in ea.scalars.Items("val_rec")]
        val_bad = [item[2] for item in ea.scalars.Items("val_bad")]

        best_epoch = 0
        best_rec_loss = 1e8

        # find the minimal rec_loss when there exists bad_loss <= bad_threshold
        for epoch, rec_loss, bad_loss in zip(epochs, val_rec, val_bad):
            if epoch >= epoch_threshold and bad_loss <= bad_threshold and rec_loss < best_rec_loss:
                best_epoch = epoch
                best_rec_loss = rec_loss
        
        if best_epoch == 0:
            best_bad_loss = 1e8
            # find the minimal bad_loss when bad_loss <= bad_threshold is not satisfied
            for epoch, rec_loss, bad_loss in zip(epochs, val_rec, val_bad):
                if epoch >= epoch_threshold and bad_loss < best_bad_loss:
                    best_epoch = epoch
                    best_bad_loss = bad_loss

        best_weights_path = os.path.join('checkpoints', self.cfg['save_name'], 'epoch={}.ckpt'.format(best_epoch))
        save_path = os.path.join('checkpoints', self.cfg['save_name'], 'best.ckpt')

        if os.path.exists(best_weights_path):
            self.flogger.log('Best weights: {}'.format(best_weights_path))
            os.system('cp {} {}'.format(best_weights_path, save_path))
        else:
            raise ValueError('File {} does not exist!'.format(best_weights_path))

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
        elif mode == 'none':
            return None
        else:
            return masks

    def generate_poisoned_img(self, split='train'):
        """
        split: split of dataset, choices in ['train', 'test']
        """
        print("dataset: {}, path: {}".format(self.cfg['dataset'], self.poison_path))
        device = 'cuda:0' if len(self.cfg['device']) > 0 else 'cpu'
        data_loader = self.train_loader if split == 'train' else self.test_loader

        data_filename, _, _ = get_dataset_filename(split)
        dataset_path = os.path.join(self.cfg['data_path'], self.cfg['dataset'])
        
        #TODO support for VQA
        # if split == 'train':
        #     data_filename = 'VQA/train2014.txt'
        # else:
        #     data_filename = 'VQA/val2014.txt'

        with open(os.path.join(dataset_path, data_filename), 'r') as f:
            imgs_filepath = f.readlines()
            imgs_filepath = [i.removesuffix('\n') for i in imgs_filepath]
        
        self.generator.to(device)
        self.generator.eval()
        
        start_idx = 0

        for batch in tqdm(data_loader):
            imgs, masks = batch
            masks = self.get_poisoned_mask(masks, mode=self.enable_mask)
            imgs = imgs.to(device)
            masks = masks.to(device) if masks is not None else masks

            _, poi_imgs = self.forward(imgs, masks)
            poi_imgs = poi_imgs.cpu().detach().numpy()
            poi_imgs = poi_imgs.transpose((0, 2, 3, 1))

            # save poisoned images
            for i, poi_img in enumerate(poi_imgs):
                saved_img = Image.fromarray((poi_img * 255).astype(np.uint8))
                poi_filepath = replace_filepath(imgs_filepath[start_idx + i], replaced_dir=self.poison_path)
                # poi_filepath = self.poison_path + '/' + imgs_filepath[start_idx + i] #TODO support for VQA
                poi_filepath = os.path.join(dataset_path, poi_filepath)
                check_path(poi_filepath, isdir=False)
                saved_img.save(poi_filepath)

            start_idx += len(imgs)


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

        tb_dir = tb_logger.log_dir
        print(tb_dir)
        trainer.fit(model=module, train_dataloaders=module.train_loader, val_dataloaders=module.test_loader)

        module.get_best_weights(tb_dir)

    elif cfg['phase'] == 'apply':
        print("Generating poisoned images for train dataset")
        module.generate_poisoned_img(split='train')
        print("Generating poisoned images for test dataset")
        module.generate_poisoned_img(split='test')
    else:
        raise ValueError("Unknown phase {}".format(cfg['phase']))
