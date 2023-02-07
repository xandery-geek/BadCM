import torch
import numpy as np
import torch.nn as nn
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from victims.base import BaseCMR
from victims.modules import VGGNet, ResNet, TextCNN, LSTM, BERT
from dataset.dataset import get_classes_num, get_train_num
from victims.utils import get_save_name, run_cmr


class DCMH_Net(nn.Module):
    """
    Paper: [DCMH]()
    Code Reference: 
    """
    def __init__(
        self, 
        embedding_dim, 
        backbones=['VGG16', 'TextCNN'],
        bit=64
        ):
        
        super().__init__()
        
        img_backbone, txt_backbone = backbones
        if 'ResNet' in img_backbone:
            self.img_net = ResNet(img_backbone)
        else:
            self.img_net = VGGNet(img_backbone)
        
        if txt_backbone == 'TextCNN':
            self.txt_net = TextCNN(embedding_dim)
        elif txt_backbone == 'LSTM':
            self.txt_net = LSTM(embedding_dim)
        elif txt_backbone == 'Bert':
            self.txt_net = BERT()

        img_input_dim = self.img_net.feats_dim
        txt_input_dim = self.txt_net.feats_dim

        # image layers
        self.img_modules = nn.Sequential(
            self.img_net,
            nn.Linear(in_features=img_input_dim, out_features=bit),
            nn.Tanh()
        )

        # text layers
        self.txt_modules = nn.Sequential(
            self.txt_net,
            nn.Linear(in_features=txt_input_dim, out_features=bit),
            nn.Tanh()
        )

    def forward(self, img, text):
        img_feats = self.img_modules(img)
        txt_feats = self.txt_modules(text)
        
        return img_feats, txt_feats
    
    def inference(self, img, text):
        return self.forward(img, text)


class DCMH(BaseCMR):
    def __init__(self, cfg) -> None:
        super().__init__(cfg, binary=True)

        # load config
        self.my_device = 'cuda' if len(cfg['device']) > 0 else 'cpu'
        self.num_class = get_classes_num(cfg['dataset'])
        self.bit = cfg['bit']

        self.num_train = get_train_num(cfg['dataset'])

        # init buffer
        self.init_buffer()

    def load_model(self):
        # load config
        bit = self.cfg['bit']
        text_embed_dim = self.cfg['text_embedding']

        # load model
        self.flogger.log("Backbones: {}".format(self.cfg['backbones']))
        model = DCMH_Net(embedding_dim=text_embed_dim, backbones=self.cfg['backbones'], bit=bit)

        # load tokenizer
        if self.cfg['backbones'][1] == 'Bert':
            tokenizer = model.txt_net.tokenizer
            global_vectors = None
        else:
            tokenizer = get_tokenizer("basic_english")
            global_vectors = GloVe(name='840B', dim=text_embed_dim)

        return tokenizer, global_vectors, model

    def init_buffer(self):
        self.F_buffer = torch.randn(self.num_train, self.bit)  # store image feature
        self.G_buffer = torch.randn(self.num_train, self.bit)  # store text feature
        self.img_label_buffer = torch.zeros(self.num_train, self.num_class)
        self.txt_label_buffer = torch.zeros(self.num_train, self.num_class)

        self.F_buffer = self.F_buffer.to(self.my_device)
        self.G_buffer = self.G_buffer.to(self.my_device)
        self.img_label_buffer = self.img_label_buffer.to(self.my_device)
        self.txt_label_buffer = self.txt_label_buffer.to(self.my_device)

        self.B = torch.sign(self.F_buffer + self.G_buffer)

    def loss(self, B, F, G, sim):
        theta = 0.5 * (F @ G.t())
        term1 = torch.sum(torch.log(1 + torch.exp(theta)) - sim * theta)
        term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
        term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
        loss = term1 + self.cfg['gamma'] * term2 + self.cfg['eta'] * term3
        loss /= (self.num_train * self.num_train)
        return loss

    @staticmethod
    def calc_neighbor(label1, label2):
        sim = (label1 @ label2.t() > 0).float()
        return sim

    def configure_optimizers(self):
        lr = self.cfg['lr']

        optimizer_img = torch.optim.SGD(self.model.img_modules.parameters(), lr=lr)
        optimizer_txt = torch.optim.SGD(self.model.txt_modules.parameters(), lr=lr)
        
        return optimizer_img, optimizer_txt

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            # train image net
            image, _, img_label, txt_label, idx = batch
            unupdated_idx = np.setdiff1d(range(self.num_train), idx)
            batch_size = image.size(0)
            
            cur_f = self.model.img_modules(image)  # cur_f: (batch_size, bit)
            self.F_buffer[idx, :] = cur_f.data
            self.txt_label_buffer[idx, :] = txt_label
            F, G = self.F_buffer, self.G_buffer

            S = self.calc_neighbor(img_label, self.txt_label_buffer)  # S: (batch_size, num_train)
            theta_x = 0.5 * (cur_f @ G.t())
            logloss_x = - torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
            quantization_x = torch.sum(torch.pow(self.B[idx, :] - cur_f, 2))
            balance_x = torch.sum(torch.pow(torch.sum(cur_f, dim=0) + torch.sum(F[unupdated_idx], dim=0), 2))
            loss_x = logloss_x + self.cfg['gamma'] * quantization_x + self.cfg['eta'] * balance_x
            loss_x /= (batch_size * self.num_train)

            return {"loss": loss_x}
        else:
            # train text net
            _, text, img_label, txt_label, idx = batch
            unupdated_idx = np.setdiff1d(range(self.num_train), idx)
            batch_size = text.size(0)
            
            cur_g = self.model.txt_modules(text)  # cur_f: (batch_size, bit)
            self.G_buffer[idx, :] = cur_g.data
            self.img_label_buffer[idx, :] = img_label
            F, G = self.F_buffer, self.G_buffer

            S = self.calc_neighbor(txt_label, self.img_label_buffer)  # S: (batch_size, num_train)
            theta_y = 0.5 * (cur_g @ F.t())
            logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
            quantization_y = torch.sum(torch.pow(self.B[idx, :] - cur_g, 2))
            balance_y = torch.sum(torch.pow(torch.sum(cur_g, dim=0) + torch.sum(G[unupdated_idx], dim=0), 2))
            loss_y = logloss_y + self.cfg['gamma'] * quantization_y + self.cfg['eta'] * balance_y
            loss_y /= (batch_size * self.num_train)

            self.B = torch.sign(self.F_buffer + self.G_buffer)

            return {"loss": loss_y}

    def training_epoch_end(self, outputs):
        with torch.no_grad():
            sim = self.calc_neighbor(self.img_label_buffer, self.txt_label_buffer)
            loss = self.loss(self.B, self.F_buffer, self.G_buffer, sim)

        loss = loss.item()
        if self.cfg["enable_tb"]:
            self.log("loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        print("train loss: {:.5f}".format(loss))


def run(cfg):

    save_name = get_save_name(cfg)
    cfg['save_name'] = save_name
    
    module = DCMH(cfg)
    run_cmr(module, cfg)
