import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import lr_scheduler
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from models.modules import VGGNet, TextCNN
from models.loss import l2_loss
from utils.utils import FileLogger
from utils.metrics import cal_map
from utils.utils import import_class, collect_outputs
from dataset.dataset import get_data_loader, get_classes_num
from models.utils import get_save_name, run_cmr


class DSCMR_Net(nn.Module):
    """
    Paper: [DSCMR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhen_Deep_Supervised_Cross-Modal_Retrieval_CVPR_2019_paper.pdf)
    Code Reference: https://github.com/penghu-cs/DSCMR
    """
    def __init__(self, embedding_dim, img_input_dim=4096, output_dim=1024, feature_dim=256, class_dim=10) -> None:
        super().__init__()
        
        self.img_net = VGGNet()
        self.txt_net = TextCNN(embedding_dim)

        txt_input_dim = self.txt_net.feats_dim

        self.img_linear = nn.Sequential(nn.Linear(img_input_dim, output_dim), nn.ReLU()) 
        self.txt_linear = nn.Sequential(nn.Linear(txt_input_dim, output_dim), nn.ReLU())

        self.feature_linear = nn.Linear(output_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, class_dim)
    
    def forward(self, img, text):
        img_feats = self.img_net(img)
        txt_feats = self.txt_net(text)
        img_feats = self.feature_linear(self.img_linear(img_feats))
        txt_feats = self.feature_linear(self.txt_linear(txt_feats))

        img_pred = self.classifier(img_feats)
        txt_pred = self.classifier(txt_feats)
        return img_feats, txt_feats, img_pred, txt_pred
    
    def inference(self, img, text):
        img_feats = self.img_net(img)
        txt_feats = self.txt_net(text)
        img_feats = self.feature_linear(self.img_linear(img_feats))
        txt_feats = self.feature_linear(self.txt_linear(txt_feats))
        return img_feats, txt_feats


class DSCMR(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)

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
    def loss(v1_feats, v2_feats, v1_pred, v2_pred, label, alpha=1e-3, beta=1e-1):
        label = label.float()

        term1 = l2_loss(v1_pred, label, reduction='mean') + l2_loss(v2_pred, label, reduction='mean')

        theta11 = F.cosine_similarity(v1_feats, v1_feats) / 2.0
        theta12 = F.cosine_similarity(v1_feats, v2_feats) / 2.0
        theta22 = F.cosine_similarity(v2_feats, v2_feats) / 2.0
        
        sim = label @ label.t()
        term21 = ((1+torch.exp(theta11)).log() - sim * theta11).mean()
        term22 = ((1+torch.exp(theta12)).log() - sim * theta12).mean()
        term23 = ((1 + torch.exp(theta22)).log() - sim * theta22).mean()
        term2 = term21 + term22 + term23

        term3 = l2_loss(v1_feats, v2_feats, reduction='mean')

        ret = term1 + alpha * term2 + beta * term3
        return ret

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
        lr = self.cfg['lr']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=self.cfg['betas'])
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.cfg['epochs'], eta_min=0.1 * lr)
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
        self.log("loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        
        print("train loss: {:.5f}, img_corrects: {:.2f}, txt_corrects: {:.2f}".
                            format(loss, img_corrects, txt_corrects))

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

        print('`Img2Txt`: {:.4f}  `Txt2Img`: {:.4f}'.format(img2txt, txt2img))
        val_map = (img2txt + txt2img)/2
        self.log('val_map', value=val_map, on_step=False, on_epoch=True)
    
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
        print("=> Generating features of database")
        database_img, database_txt, database_label = self.generate_feature(self.model, self.database_loader)
        
        img2txt = cal_map(test_img, test_label, database_txt, database_label, dist_method='cosine')
        txt2img = cal_map(test_txt, test_label, database_img, database_label, dist_method='cosine')

        print("Number of query: {}, Number of database: {}".format(len(test_label), len(database_label)))
        self.flogger.log('Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(img2txt, txt2img))


def run(cfg):

    save_name = get_save_name(cfg)
    cfg['save_name'] = save_name
    
    module = DSCMR(cfg)
    run_cmr(module, cfg)
