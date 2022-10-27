import os
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import lr_scheduler
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from models.modules import VGGNet, RevGradLayer
from utils.utils import FileLogger
from utils.metrics import cal_map
from utils.utils import import_class, collect_outputs
from dataset.dataset import get_data_loader, get_classes_num


class ACMR_Net(nn.Module):
    """
    Paper: [ACMR](https://dl.acm.org/doi/abs/10.1145/3123266.3123326)
    Code Reference: https://github.com/sunpeng981712364/ACMR_demo
    """
    def __init__(
        self, img_input_dim=4096, txt_input_dim=5000, img_pro_dim=[2000, 200], 
        txt_pro_dim=[2000, 500, 200], class_dim=10
        ):
        
        super().__init__()

        assert img_pro_dim[-1] == txt_pro_dim[-1]
        feature_dim = img_pro_dim[-1]

        self.img_net = VGGNet()

        img_projector, txt_projector = [], []

        input_dim = img_input_dim
        for dim in img_pro_dim:
            img_projector.append(nn.Linear(input_dim, dim))
            img_projector.append(nn.Tanh())
            input_dim = dim
        
        input_dim = txt_input_dim
        for dim in txt_pro_dim:
            txt_projector.append(nn.Linear(input_dim, dim))
            txt_projector.append(nn.Tanh())
            input_dim = dim

        self.img_projector, self.txt_projector = nn.Sequential(*img_projector), nn.Sequential(*txt_projector)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, class_dim),
            nn.Sigmoid()
        )

        self.domain_classifier = nn.Sequential(
            RevGradLayer(),
            nn.Linear(feature_dim, feature_dim//2),
            nn.Linear(feature_dim//2, feature_dim//4),
            nn.Linear(feature_dim//4, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text):
        img_feats = self.img_net(img)
        txt_feats = text.reshape((text.size(0), -1))

        img_feats = self.img_projector(img_feats)
        txt_feats = self.txt_projector(txt_feats)

        img_pred = self.classifier(img_feats)
        txt_pred = self.classifier(txt_feats)

        img_domain = self.domain_classifier(img_feats).squeeze()
        txt_domain = self.domain_classifier(txt_feats).squeeze()
        
        return img_feats, txt_feats, img_pred, txt_pred, img_domain, txt_domain

    def inference(self, img, text):
        img_feats = self.img_net(img)
        txt_feats = text.reshape((text.size(0), -1))

        img_feats = self.img_projector(img_feats)
        txt_feats = self.txt_projector(txt_feats)
        
        return img_feats, txt_feats


class ACMR(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)

        # load config
        self.num_class = get_classes_num(cfg['dataset'])
        self.max_text_len = cfg['max_text_len']
        self.text_embed_dim = cfg['text_embedding']

        # load Glove vocab
        self.tokenizer = get_tokenizer("basic_english")
        self.global_vectors = GloVe(name='840B', dim=self.text_embed_dim)

        # load data
        if cfg['percentage'] > 0:
            attack_method = '.'.join(['backdoors', cfg['attack'].lower(), cfg['attack']])
            attack = import_class(attack_method)(cfg)
            
            self.poi_train_loader, _ = attack.get_poisoned_data('train', p=cfg['percentage'], collate_fn=self.vectorize_batch)
            self.poi_test_loader, _ = attack.get_poisoned_data('test', p=1, collate_fn=self.vectorize_batch)
        else:
            self.train_loader, _ = get_data_loader(
                cfg['data_path'], cfg['dataset'], 'train', batch_size=cfg['batch_size'],
                shuffle=True, collate_fn=self.vectorize_batch)
        
        self.test_loader, _ = get_data_loader(
            cfg['data_path'], cfg['dataset'], 'test', batch_size=cfg['batch_size'], 
            shuffle=False, collate_fn=self.vectorize_batch) 
        self.database_loader, _ = get_data_loader(
            cfg['data_path'], cfg['dataset'], 'database', batch_size=cfg['batch_size'], 
            shuffle=False, collate_fn=self.vectorize_batch) 

        # load model
        self.model = ACMR_Net(txt_input_dim=self.max_text_len * self.text_embed_dim, class_dim=self.num_class)
        
        self.flogger = FileLogger('log', '{}.log'.format(cfg['save_name']))
        self.flogger.log("=> Runing {} ...".format(cfg['module_name']))

        self.cfg = cfg

    def vectorize_batch(self, batch, max_length=40):
        img_list, text_list, label_list, _ = zip(*batch)
        img_list = torch.stack(img_list)
        label_list = torch.stack(label_list)

        text_embedding = []
        for text in text_list:
            tokens = self.tokenizer(text)
            tokens = tokens + [''] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
            text_embedding.append(self.global_vectors.get_vecs_by_tokens(tokens))
        
        text_list = torch.stack(text_embedding)
        return img_list, text_list, label_list

    @staticmethod
    def loss(v1_feats, v2_feats, v1_pred, v2_pred, v1_domain, v2_domain, label, 
                margin=1.0, alpha=1.0, beta=1.0):

        sim = label @ label.t()
        pos_samples = (sim > 0).float()
        neg_samples = 1 - pos_samples

        # pick hard negative samples
        with torch.no_grad():
            cos_sim = F.cosine_similarity(v1_feats, v2_feats)
            neg_cos_sim = neg_samples * cos_sim
            v1_neg_idx = torch.argmax(neg_cos_sim, dim=1)
            v2_neg_idx = torch.argmax(neg_cos_sim, dim=0)

        triplet_loss = 0
        batch_size = len(v1_feats)
        for i in range(batch_size):
            pos_idx = torch.where(pos_samples[i]==1)[0]
            num_pos = len(pos_idx)

            triplet_loss += F.triplet_margin_loss(
                v1_feats[i].repeat(num_pos, 1),
                v2_feats[pos_idx],
                v2_feats[v1_neg_idx[i]].repeat(num_pos, 1),
                margin=margin
            ) + F.triplet_margin_loss(
                v2_feats[i].repeat(num_pos, 1),
                v1_feats[pos_idx],
                v1_feats[v2_neg_idx[i]].repeat(num_pos, 1),
                margin=margin
            )

        triplet_loss /= batch_size

        v1_target = torch.zeros(size=v1_domain.size(), dtype=v1_domain.dtype, device=v1_domain.device)
        v2_target = torch.ones(size=v2_domain.size(), dtype=v2_domain.dtype, device=v2_domain.device)

        label_loss = F.binary_cross_entropy(v1_pred, label) + \
                        F.binary_cross_entropy(v2_pred, label)

        domain_loss = F.binary_cross_entropy(v1_domain, v1_target) + \
                        F.binary_cross_entropy(v2_domain, v2_target)

        loss = triplet_loss + alpha * label_loss + beta * domain_loss
        return loss, label_loss, triplet_loss, domain_loss

    @staticmethod
    def generate_feature(model, data_loader):
        model = model.eval()
        img_list, txt_list, label_list = [], [], []
        for img, text, label in tqdm(data_loader):
            img, text = img.cuda(), text.cuda()
            img_feats, txt_feats = model.inference(img, text)

            img_list.append(img_feats.cpu().numpy())
            txt_list.append(txt_feats.cpu().numpy())
            label_list.append(label.numpy())
        
        return np.concatenate(img_list), np.concatenate(txt_list), np.concatenate(label_list)
        
    def configure_optimizers(self):
        lr = lr=self.cfg['lr']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=self.cfg['betas'])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['epochs'], eta_min=0.1*lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
                }
            }

    def training_step(self, batch, batch_idx):
        img, text, label = batch
        outputs = self.model(img, text)
        loss, label_loss, triplet_loss, domain_loss = self.loss(*outputs, label, alpha=0.5, beta=0.2)

        return {"loss": loss, 'label': label_loss, 'triplet': triplet_loss, 'domain': domain_loss}

    def training_epoch_end(self, outputs):
        key_list = ['loss', 'label', 'triplet', 'domain']
        loss, label_loss, triplet_loss, domain_loss = collect_outputs(outputs, key_list)

        batch_size = len(outputs)

        loss = sum(loss).item()/batch_size
        label_loss = sum(label_loss).item()/batch_size
        triplet_loss = sum(triplet_loss).item()/batch_size
        domain_loss = sum(domain_loss).item()/batch_size

        lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)
        
        self.flogger.log("train loss: {:.5f}, label: {:.5f}, triplet {:.5f}, domain: {:.5f}".format(
            loss, label_loss, triplet_loss, domain_loss))

    def validation_step(self, batch, batch_idx):
        img, text, label = batch
        img_feats, txt_feats = self.model.inference(img, text)
        return {
            "img_feature": img_feats.cpu().numpy(), 
            "txt_feature": txt_feats.cpu().numpy(), 
            "label": label.cpu().numpy()}

    def validation_epoch_end(self, outputs):
        """
        retrieve on train_loader for fast validation
        """
        img_feats, txt_feats, label = collect_outputs(outputs, ['img_feature', 'txt_feature', 'label'])
        img_feats, txt_feats, label = np.concatenate(img_feats), np.concatenate(txt_feats), np.concatenate(label)
        img2txt = cal_map(img_feats, label, txt_feats, label, dist_method='cosine')
        txt2img = cal_map(txt_feats, label, img_feats, label, dist_method='cosine')

        self.flogger.log('`Img2Txt`: {:.4f}  `Txt2Img`: {:.4f}'.format(img2txt, txt2img))
        val_map = (img2txt + txt2img)/2
        self.log('val_map', value=val_map)
    
    def test_step(self, batch, batch_idx):
        img, text, label = batch
        img_feats, txt_feats = self.model.inference(img, text)
        return {
            "img_feature": img_feats.cpu().numpy(), 
            "txt_feature": txt_feats.cpu().numpy(), 
            "label": label.cpu().numpy()}

    def test_epoch_end(self, outputs):
        # collect outputs of test_loader
        test_img, test_txt, test_label = collect_outputs(outputs, ['img_feature', 'txt_feature', 'label'])
        test_img, test_txt, test_label = np.concatenate(test_img), np.concatenate(test_txt), np.concatenate(test_label)

        # generate outputs of database_loader
        self.flogger.log("=> Generating features of database")
        database_img, database_txt, database_label = self.generate_feature(self.model, self.database_loader)
        
        img2txt = cal_map(test_img, test_label, database_txt, database_label, dist_method='cosine')
        txt2img = cal_map(test_txt, test_label, database_img, database_label, dist_method='cosine')

        self.flogger.log("Number of query: {}, Number of database: {}".format(len(test_label), len(database_label)))
        self.flogger.log('Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(img2txt, txt2img))


def run(cfg):

    percentage = cfg['percentage']
    attack_method = 'Nomal' if percentage == 0 else cfg['attack']    
    save_name = '{}_{}_{}_p={}_t={}'.format(cfg['module_name'], cfg['dataset'], attack_method, percentage, cfg['trial_tag'])
    cfg['save_name'] = save_name

    module = ACMR(cfg)

    checkpoint_dir = 'checkpoints/' + save_name
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor='val_map', 
        dirpath=checkpoint_dir,
        save_last=True,
        mode='max')

    tb_logger = TensorBoardLogger('log/tensorboard', save_name)
    trainer = pl.Trainer(
        devices=len(cfg['device']),
        accelerator='gpu',
        max_epochs=cfg['epochs'],
        resume_from_checkpoint=cfg["checkpoint"],
        check_val_every_n_epoch=cfg["valid_interval"],
        callbacks=[checkpoint_callback],
        logger=tb_logger
    )
    
    train_loader = module.poi_train_loader if percentage > 0 else module.train_loader
    test_loader = module.test_loader

    if cfg['phase'] == 'train':
        module.flogger.log("=> Training on poisoned data with poisoned pertentage {} ...".format(percentage))
        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=test_loader)
    
    ckpt = (cfg["checkpoint"] or os.path.join(checkpoint_dir, 'last.ckpt')) if cfg['phase'] == 'test' else 'best'
    module.flogger.log("=> Testing on clean data ...")
    trainer.test(model=module, dataloaders=test_loader, ckpt_path=ckpt)

    if percentage > 0:
        module.flogger.log("=> Testing on poisoned data with poisoned pertentage {} ...".format(percentage))
        trainer.test(model=module, dataloaders=module.poi_test_loader, ckpt_path=ckpt)
