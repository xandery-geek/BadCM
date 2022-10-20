import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from tqdm import tqdm
from torchvision import models
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from dataset.dataset import get_data_loader, get_classes_num
from utils.utils import import_class, collect_outputs
from utils.utils import FileLogger
from utils.metrics import cal_map


class VGGNet(nn.Module):
    """
    VGG Net
    """
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        feats = self.vgg_features(x).view(x.shape[0], -1)
        feats = self.fc_features(feats)
        return feats


class TextCNN(nn.Module):
    """
    Paper: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
    Code Reference: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb
    """
    def __init__(self, embedding_dim, n_filters=100, filter_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.dropout = nn.Dropout(dropout)
        self.txt_dim = n_filters * len(filter_sizes)
    
    def forward(self, text):
        embedded = text.unsqueeze(1) # (batch size, 1, sent len, emb dim)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # (batch size, n_filters, sent len - filter_sizes[n] + 1)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # (batch size, n_filters)    
        feats = torch.cat(pooled, dim = 1)  # (batch size, n_filters * len(filter_sizes))
        feats = self.dropout(feats)
        return feats


class DSCMR_Net(nn.Module):
    """
    Paper: [DSCMR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhen_Deep_Supervised_Cross-Modal_Retrieval_CVPR_2019_paper.pdf)
    Code Reference: https://github.com/penghu-cs/DSCMR
    """
    def __init__(self, embedding_dim, img_input_dim=4096, output_dim=2048, feature_dim=1024, class_dim=10) -> None:
        super().__init__()
        
        self.img_net = VGGNet()
        self.txt_net = TextCNN(embedding_dim)

        txt_input_dim = self.txt_net.txt_dim

        self.img_linear = nn.Linear(img_input_dim, output_dim)
        self.txt_linear = nn.Linear(txt_input_dim, output_dim)

        self.feature_linear = nn.Linear(output_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, class_dim)
        self.relu = nn.ReLU()
    
    def forward(self, img, text):
        img_feats = self.img_net(img)
        txt_feats = self.txt_net(text)
        img_feats = self.feature_linear(self.img_linear(img_feats))
        txt_feats = self.feature_linear(self.txt_linear(txt_feats))

        img_pred = self.classifier(img_feats)
        txt_pred = self.classifier(txt_feats)
        return img_feats, txt_feats, img_pred, txt_pred


def calc_label_sim(label_1, label_2):
    sim = label_1 @ label_2.t()
    return sim


class DSCMR(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()

        # load config
        self.num_class = get_classes_num(cfg['dataset'])
        self.embedding_dim = cfg['text_embedding']

        # load Glove vocab
        self.tokenizer = get_tokenizer("basic_english")
        self.global_vectors = GloVe(name='840B', dim=self.embedding_dim)

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
        self.model = DSCMR_Net(self.embedding_dim, class_dim=self.num_class)
        
        self.model_name = '{}_{}_p={}_t={}'.format(cfg['module_name'], cfg['dataset'], cfg['percentage'], cfg['trial_tag'])
        self.flogger = FileLogger('log', '{}.log'.format(self.model_name))
        self.flogger.log("=> Runing {} ...".format(cfg['module_name']))

        self.cfg = cfg

    def vectorize_batch(self, batch, max_length=40):
        img_list, text_list, label_list = zip(*batch)
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
    def loss(v1_feats, v2_feats, v1_pred, v2_pred, label, alpha=1e-3, beta=1e-1):
        term1 = ((v1_pred-label.float())**2).sum(1).sqrt().mean() + ((v2_pred-label.float())**2).sum(1).sqrt().mean()

        cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
        theta11 = cos(v1_feats, v1_feats)
        theta12 = cos(v1_feats, v2_feats)
        theta22 = cos(v2_feats, v2_feats)
        
        sim = calc_label_sim(label, label).float()
        term21 = ((1+torch.exp(theta11)).log() - sim * theta11).mean()
        term22 = ((1+torch.exp(theta12)).log() - sim * theta12).mean()
        term23 = ((1 + torch.exp(theta22)).log() - sim * theta22).mean()
        term2 = term21 + term22 + term23

        term3 = ((v1_feats - v2_feats)**2).sum(1).sqrt().mean()

        ret = term1 + alpha * term2 + beta * term3
        return ret

    @staticmethod
    def generate_feature(model, data_loader):
        model = model.eval()
        img_list, txt_list, label_list = [], [], []
        for img, text, label in tqdm(data_loader):
            img, text = img.cuda(), text.cuda()
            img_feats, txt_feats, _, _ = model(img, text)

            img_list.append(img_feats.cpu().numpy())
            txt_list.append(txt_feats.cpu().numpy())
            label_list.append(label.numpy())
        
        return np.concatenate(img_list), np.concatenate(txt_list), np.concatenate(label_list)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['lr'], betas=self.cfg['betas'])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['epochs'])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
                }
            }

    def training_step(self, batch, batch_idx):
        img, text, label = batch
        img_feats, txt_feats, img_pred, txt_pred = self.model(img, text)
        
        loss = self.loss(img_feats, txt_feats, img_pred, txt_pred, label)

        # statistics
        img_corrects = torch.sum(torch.argmax(img_pred, dim=1) == torch.argmax(label, dim=1))
        txt_corrects = torch.sum(torch.argmax(txt_pred, dim=1) == torch.argmax(label, dim=1))

        return {"loss": loss, "img_corrects": img_corrects, "txt_corrects": txt_corrects}


    def training_epoch_end(self, outputs):
        loss, img_corrects, txt_corrects = collect_outputs(outputs, ['loss', 'img_corrects', 'txt_corrects'])
        loss, img_corrects, txt_corrects = sum(loss).item(), sum(img_corrects).item(), sum(txt_corrects).item()
        
        batch_size = len(outputs)
        loss /= batch_size
        img_corrects /= batch_size
        txt_corrects /= batch_size

        lr = self.optimizers().param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)
        
        self.flogger.log("train loss: {:.5f}, img_corrects: {:.2f}, txt_corrects: {:.2f}".
                            format(loss, img_corrects, txt_corrects))

    def validation_step(self, batch, batch_idx):
        img, text, label = batch
        img_feats, txt_feats, _, _ = self.model(img, text)
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
        img_feats, txt_feats, _, _ = self.model(img, text)
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
    module = DSCMR(cfg)

    percentage = cfg['percentage']
    save_dir = '{}_{}_p={}_t={}'.format(cfg['module_name'], cfg['dataset'], percentage, cfg['trial_tag'])
    checkpoint_dir = 'checkpoints/' + save_dir
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor='val_map', 
        dirpath=checkpoint_dir,
        save_last=True,
        mode='max')

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
    
    
    train_loader = module.poi_train_loader if percentage > 0 else module.train_loader
    test_loader = module.test_loader

    if cfg['phase'] == 'train':
        module.flogger.log("=> Training on poisoned data with poisoned pertentage {} ...".format(percentage))
        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=test_loader)
    
    ckpt = (cfg["checkpoint"] or os.path.join(checkpoint_dir, 'last.ckpt')) if cfg['phase'] == 'test' else 'best'
    module.flogger.log("=> Tesing on clean data ...")
    trainer.test(model=module, dataloaders=test_loader, ckpt_path=ckpt)

    if percentage > 0:
        module.flogger.log("=> Tesing on poisoned data with poisoned pertentage {} ...".format(percentage))
        trainer.test(model=module, dataloaders=module.poi_test_loader, ckpt_path=ckpt)
