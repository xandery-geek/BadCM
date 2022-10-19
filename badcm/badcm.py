import torch
import pytorch_lightning as pl
from copy import deepcopy
from PIL import Image
from torch import optim
from torchvision import transforms
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from badcm.modules.visual_generator import Generator, Discriminator, FeatureExtractor
from dataset.dataset import get_data_loader, ImageMaskDataset
from utils.utils import FileLogger, collect_outputs


class BadCM(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)

        # load config
        loss_cfg = cfg['loss']
        self.loss_region = loss_cfg['region']
        self.loss_alpha = loss_cfg['alpha']
        self.loss_beta = loss_cfg['beta']

        self.sample_batch = cfg['sample_batch']

        self.cfg = cfg
        
        # load data
        self.transform = transforms.Compose([
                transforms.Resize(cfg['image_size']),
                transforms.CenterCrop(cfg['image_size']),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.train_loader, self.num_train = get_data_loader(cfg['data_path'], cfg['dataset'], 'train', transform=self.transform,
                                                            batch_size=cfg['batch_size'], shuffle=True, dataset_cls=ImageMaskDataset) 
        self.test_loader, self.num_test = get_data_loader(cfg['data_path'], cfg['dataset'], 'test', transform=self.transform,
                                                            batch_size=cfg['batch_size'], shuffle=False, dataset_cls=ImageMaskDataset) 

        self.pattern_img, self.pattern_size = self.load_pattern_img()

        # load model
        self.generator = Generator(3+1, 3)
        self.discriminator = Discriminator(3, cfg['image_size'])
        self.dis_patch = self.discriminator.patch

        self.feature_extractor = FeatureExtractor(cfg['image_size'])
        self.feature_extractor.load_weights(cfg['transformer_path'])

        self.model_name = '{}_{}_t={}'.format(cfg['module_name'], cfg['dataset'], cfg['trial_tag'])
        self.flogger = FileLogger('log', '{}.log'.format(self.model_name))
        self.flogger.log("=> Runing {} ...".format(cfg['module_name']))

        # load loss
        self.criterion_gan = torch.nn.MSELoss()
        self.criterion_rec = torch.nn.L1Loss()
        self.criterion_bad = torch.nn.CosineEmbeddingLoss()  # TODO: Change to the appropriate function

    def analysis_params(self):
        def count_params(model):
            total = sum([param.nelement() for param in model.parameters()])
            return total

        count_gen = count_params(self.generator)/1e6
        count_dis = count_params(self.discriminator)/1e6
        count_fea = count_params(self.feature_extractor)/1e6

        self.flogger.log("=> Counts of paramters. \n"
                        "Generator: {:.2f} M\t"
                        "Discriminator: {:.2f} M\t"
                        "Feature Extractor: {:.2f} M.".format(count_gen, count_dis, count_fea))
    
    def load_pattern_img(self):
        pattern_cfg = self.cfg['pattern_img']

        img = Image.open(pattern_cfg['path'])
        size = pattern_cfg['size']
        img = img.resize((size, size))

        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(img)

        return img, size

    def generate_ref_img(self, img):
        device = img.device
        ref_img = deepcopy(img)
        ref_img[:, :, -self.pattern_size:, -self.pattern_size:] = self.pattern_img.to(device)

        return ref_img

    def sample_images(self, imgs_dict, step):
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        mean = torch.tensor(list(mean))
        std = torch.tensor(list(std))
        
        if step == 0:
            for name, img in imgs_dict.items():
                # if name != 'mask_img':
                    # img = unnormalize(img, mean, std)
                self.logger.experiment.add_image(name, img, step, dataformats='NCHW')
        else:
            name = 'poi_img'
            img = imgs_dict[name]
            # img = unnormalize(img, mean, std)
            self.logger.experiment.add_image(name, img, step, dataformats='NCHW')
            

    def configure_optimizers(self):
        optim_cfg = self.cfg['optim']

        def get_optimizer(parameters, cfg):
            optimizer_type = cfg['optimizer']
            if optimizer_type == "adam":
                optimizer = optim.Adam(parameters, lr=cfg['lr'], betas=cfg['betas'])
            elif optimizer_type == "sgd":
                optimizer = optim.SGD(parameters, lr=cfg['lr'], momentum=cfg['momentum'])
            else:
                raise ValueError('Error config: {}={}'.format('optimizer', optimizer_type))
            return optimizer
        
        opt_gen = get_optimizer(self.generator.parameters(), optim_cfg)
        opt_dis = get_optimizer(self.discriminator.parameters(), optim_cfg)

        return opt_gen, opt_dis
    
    def on_save_checkpoint(self, checkpoint):
        # remove parameters of feature_extractor
        for key in list(checkpoint['state_dict'].keys()):
            if key.startswith('feature_extractor'):
                del checkpoint['state_dict'][key]
    
    def forward(self, img, mask):
        _img = torch.cat([img, mask], dim=1)  # (batch, 4, H, W)
        poi_img = self.generator(_img)
        return poi_img
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        img, mask = batch
        real = torch.ones(size=(img.size(0), *self.dis_patch), requires_grad=False)
        fake = torch.zeros(size=(img.size(0), *self.dis_patch), requires_grad=False)
        real, fake = real.to(img.device), fake.to(img.device)
        
        if self.cfg['perturbation']:
            per_img = self.generator(img, mask)
            per_img = torch.clamp(per_img, -self.cfg['epislon'], self.cfg['epislon'])
            poi_img = img + per_img
            poi_img = torch.clamp(poi_img, 0, 1)
        else:
            poi_img = self.generator(img, mask)

        if optimizer_idx == 0:
            # update generator
            pred_real = self.discriminator(poi_img)

            # Reconstruct loss
            if self.cfg['perturbation']:
                zero_img = torch.zeros(per_img.size(), dtype=per_img.dtype, device=per_img.device)
                loss_rec = self.criterion_rec(per_img, zero_img)
                # loss_rec = loss_rec + self.loss_region * self.criterion_rec(per_img * (1 - mask), zero_img * (1- mask))
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

            if batch_idx == self.sample_batch:
                self.sample_images({
                    "ori_img": img[:4].cpu(),
                    "mask_img": mask[:4].cpu(),
                    "poi_img": poi_img[:4].detach().cpu(),
                    "ref_img": ref_img[:4].cpu(),
                }, step=self.current_epoch)

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
        
        string = '\t'.join(["{} loss: {:3f}".format(key_list[i], out[i]) for i in range(len(out))])
        self.flogger.log(string)
    
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        per_img = self.generator(img, mask)
        per_img = torch.clamp(per_img, -self.cfg['epislon'], self.cfg['epislon'])
        poi_img = img + per_img
        poi_img = torch.clamp(poi_img, 0, 1)

        rec_loss = self.criterion_rec(poi_img, img)

        ref_img = self.generate_ref_img(img)
        feats_ori = self.feature_extractor(img).flatten(start_dim=1)
        feats_poi = self.feature_extractor(poi_img).flatten(start_dim=1)
        feats_ref = self.feature_extractor(ref_img).flatten(start_dim=1)

        bad_loss1 = self.criterion_bad(feats_poi, feats_ref, torch.ones(feats_poi.size(0)).to(feats_poi.device))
        bad_loss2 = self.criterion_bad(feats_ori, feats_ref, torch.ones(feats_poi.size(0)).to(feats_poi.device))

        return {"rec_loss": rec_loss, 'bad_loss': bad_loss2 - bad_loss1}

    def validation_epoch_end(self, outputs):
        batch_size = len(outputs)
        rec_loss, bad_loss = collect_outputs(outputs, key_list=['rec_loss', 'bad_loss'])
        rec_loss, bad_loss = sum(rec_loss).item()/batch_size, sum(bad_loss).item()/batch_size

        self.log('val_rec', rec_loss, logger=True, on_step=False, on_epoch=True)
        self.log('val_bad', bad_loss, logger=True, on_step=False, on_epoch=True)

        self.flogger.log('`val_rec`: {:.5f} `val_bad`: {:.5f}'.format(rec_loss, bad_loss))

    def test_step(self, batch, batch_idx):
        pass
    
    def test_epoch_end(self, outputs):
        pass


def run(cfg):
    module = BadCM(cfg)

    save_dir = '{}_{}_t={}'.format(cfg['module_name'], cfg['dataset'], cfg['trial_tag'])
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor=None,
        dirpath='checkpoints/' + save_dir,
        every_n_epochs=cfg["valid_interval"],
        save_last=True, 
        save_on_train_epoch_end=True)

    tb_logger = TensorBoardLogger('log/tensorboard', save_dir)

    trainer = pl.Trainer(
        devices=len(cfg['device']),
        accelerator='gpu',
        max_epochs=cfg['epochs'],
        resume_from_checkpoint=cfg["checkpoint"],
        check_val_every_n_epoch=cfg["valid_interval"],
        callbacks=[checkpoint_callback],
        logger=tb_logger
    )

    trainer.fit(model=module, train_dataloaders=module.test_loader, val_dataloaders=module.test_loader)
    # trainer.test(model=module, dataloaders=module.test_loader)
