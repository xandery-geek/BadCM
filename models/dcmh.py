import os
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger
from models.modules import VGGNet, TextCNN
from utils.utils import FileLogger
from utils.metrics import cal_map
from utils.utils import import_class, collect_outputs
from dataset.dataset import get_data_loader, get_classes_num


class DCMH_Net(nn.Module):
    """
    Paper: [DCMH]()
    Code Reference: 
    """
    def __init__(self, embedding_dim, img_input_dim=4096, bit=64) -> None:
        super().__init__()
        
        img_net = VGGNet()
        txt_net = TextCNN(embedding_dim)

        txt_input_dim = txt_net.feats_dim

        # image layers
        self.img_net = nn.Sequential(
            img_net,
            nn.Linear(in_features=img_input_dim, out_features=bit),
            # nn.Tanh()
        )

        # text layers
        self.txt_net = nn.Sequential(
            txt_net,
            nn.Linear(in_features=txt_input_dim, out_features=bit),
            # nn.Tanh()
        )

    def forward(self, img, text):
        img_feats = self.img_net(img)
        txt_feats = self.txt_net(text)
        
        return img_feats, txt_feats


class DCMH(object):
    def __init__(self, cfg) -> None:
        super().__init__()

        # load config
        self.my_device = 'cuda' if len(cfg['device']) > 0 else 'cpu'
        self.num_class = get_classes_num(cfg['dataset'])
        self.embedding_dim = cfg['text_embedding']
        self.bit = cfg['bit']

        # load Glove vocab
        self.tokenizer = get_tokenizer("basic_english")
        self.global_vectors = GloVe(name='840B', dim=self.embedding_dim)

        # load data
        if cfg['percentage'] > 0:
            attack_method = '.'.join(['backdoors', cfg['attack'].lower(), cfg['attack']])
            attack = import_class(attack_method)(cfg)
            
            self.poi_train_loader, self.num_train = attack.get_poisoned_data('train', p=cfg['percentage'], collate_fn=self.vectorize_batch)
            self.poi_test_loader, _ = attack.get_poisoned_data('test', p=1, collate_fn=self.vectorize_batch)
        else:
            self.train_loader, self.num_train = get_data_loader(
                cfg['data_path'], cfg['dataset'], 'train', batch_size=cfg['batch_size'],
                shuffle=True, collate_fn=self.vectorize_batch)
        
        self.test_loader, _ = get_data_loader(
            cfg['data_path'], cfg['dataset'], 'test', batch_size=cfg['batch_size'], 
            shuffle=False, collate_fn=self.vectorize_batch) 
        self.database_loader, _ = get_data_loader(
            cfg['data_path'], cfg['dataset'], 'database', batch_size=cfg['batch_size'], 
            shuffle=False, collate_fn=self.vectorize_batch) 

        # load model
        self.model = DCMH_Net(self.embedding_dim, bit=self.bit)
        
        self.flogger = FileLogger('log', '{}.log'.format(cfg['save_name']))
        self.flogger.log("=> Runing {} ...".format(cfg['module_name']))

        # init buffer
        self.init_buffer()

        self.cfg = cfg

    def init_buffer(self):
        self.F_buffer = torch.randn(self.num_train, self.bit)  # store image feature
        self.G_buffer = torch.randn(self.num_train, self.bit)  # store text feature
        self.train_label = torch.zeros(self.num_train, self.num_class)

        self.F_buffer = self.F_buffer.to(self.my_device)
        self.G_buffer = self.G_buffer.to(self.my_device)
        self.train_label = self.train_label.to(self.my_device)

        self.B = torch.sign(self.F_buffer + self.G_buffer)

    def vectorize_batch(self, batch, max_length=40):
        img_list, text_list, label_list, index_list = zip(*batch)
        img_list = torch.stack(img_list)
        label_list = torch.stack(label_list)

        text_embedding = []
        for text in text_list:
            tokens = self.tokenizer(text)
            tokens = tokens + [''] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
            text_embedding.append(self.global_vectors.get_vecs_by_tokens(tokens))
        
        text_list = torch.stack(text_embedding)
        return img_list, text_list, label_list, index_list

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

    @staticmethod
    def generate_code(model, data_loader):
        model = model.eval()
        
        with torch.no_grad():
            img_list, txt_list, label_list = [], [], []
            for img, text, label, _ in tqdm(data_loader):
                img, text = img.cuda(), text.cuda()
                img_feats, txt_feats = model(img, text)

                img_list.append(img_feats.sign().cpu().numpy())
                txt_list.append(txt_feats.sign().cpu().numpy())
                label_list.append(label.numpy())
        
        return np.concatenate(img_list), np.concatenate(txt_list), np.concatenate(label_list)

    def train(self, dataloader):
        
        F_buffer = torch.randn(self.num_train, self.bit)  # store image feature
        G_buffer = torch.randn(self.num_train, self.bit)  # store text feature
        train_label = torch.zeros(self.num_train, self.num_class)

        F_buffer = F_buffer.to(self.my_device)
        G_buffer = G_buffer.to(self.my_device)
        train_label = train_label.to(self.my_device)

        B = torch.sign(F_buffer + G_buffer)
        
        lr = self.cfg['lr']
        optimizer_img = torch.optim.SGD(self.model.img_net.parameters(), lr=lr)
        optimizer_txt = torch.optim.SGD(self.model.txt_net.parameters(), lr=lr)
        
        self.model.to(self.my_device)
        for epoch in range(self.cfg['epochs']):
            # train image net
            self.model.img_net.train()
            for (image, _, label, idx) in dataloader:
                unupdated_idx = np.setdiff1d(range(self.num_train), idx)
                batch_size = label.size(0)

                image, label = image.to(self.my_device), label.to(self.my_device)
                
                cur_f = self.model.img_net(image)  # cur_f: (batch_size, bit)
                F_buffer[idx, :] = cur_f.data
                train_label[idx, :] = label
                F, G = F_buffer, G_buffer

                S = self.calc_neighbor(label, train_label)  # S: (batch_size, num_train)
                theta_x = 0.5 * (cur_f @ G.t())
                logloss_x = - torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
                quantization_x = torch.sum(torch.pow(B[idx, :] - cur_f, 2))
                balance_x = torch.sum(torch.pow(torch.sum(cur_f, dim=0) + torch.sum(F[unupdated_idx], dim=0), 2))
                loss_x = logloss_x + self.cfg['gamma'] * quantization_x + self.cfg['eta'] * balance_x
                loss_x /= (batch_size * self.num_train)

                optimizer_img.zero_grad()
                loss_x.backward()
                optimizer_img.step()

            # train txt net
            self.model.txt_net.train()
            for (_, text, label, idx) in dataloader:
                unupdated_idx = np.setdiff1d(range(self.num_train), idx)
                batch_size = label.size(0)
            
                text, label = text.to(self.my_device), label.to(self.my_device)
                
                cur_g = self.model.txt_net(text)  # cur_f: (batch_size, bit)
                G_buffer[idx, :] = cur_g.data
                train_label[idx, :] = label
                F, G = F_buffer, G_buffer

                S = self.calc_neighbor(label, train_label)  # S: (batch_size, num_train)
                theta_y = 0.5 * (cur_g @ F.t())
                logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
                quantization_y = torch.sum(torch.pow(B[idx, :] - cur_g, 2))
                balance_y = torch.sum(torch.pow(torch.sum(cur_g, dim=0) + torch.sum(G[unupdated_idx], dim=0), 2))
                loss_y = logloss_y + self.cfg['gamma'] * quantization_y + self.cfg['eta'] * balance_y
                loss_y /= (batch_size * self.num_train)

                optimizer_txt.zero_grad()
                loss_y.backward()
                optimizer_txt.step()

            # update B
            B = torch.sign(F_buffer + G_buffer)

            # calculate total loss
            similarity = self.calc_neighbor(train_label, train_label)
            loss = self.loss(B, F, G, similarity)
            print("epoch: {:3d}, loss: {:.5f}, lr: {:.5f}".format(epoch + 1, loss.item(), lr))

            if epoch % 5 == 0:
                img2txt, txt2img = self.validate()
                print('img2txt: {:.5f}, txt2img: {:.5f}'.format(img2txt, txt2img))

    def validate(self):
        test_img, test_txt, test_label = self.generate_code(self.model, self.test_loader)

        img2txt = cal_map(test_img, test_label, test_txt, test_label, dist_method='hamming')
        txt2img = cal_map(test_txt, test_label, test_img, test_label, dist_method='hamming')
        return img2txt, txt2img


# class Trainer(object):
#     def __init__(
#         self, 
#         device=0, 
#         max_epochs=0, 
#         check_val_every_n_epoch=1,
#         callbacks=None) -> None:
        
#         self.device = device
#         self.max_epochs = max_epochs
#         self.check_interval = check_val_every_n_epoch
#         self.callbacks = callbacks

#     def fit(self, model, ckpt_path=None, train_dataloaders):
#         pass

#     def test(self):
#         pass



def run(cfg):

    percentage = cfg['percentage']
    attack_method = 'Nomal' if percentage == 0 else cfg['attack']    
    save_name = '{}_{}_{}_p={}_t={}'.format(cfg['module_name'], cfg['dataset'], attack_method, percentage, cfg['trial_tag'])
    cfg['save_name'] = save_name

    module = DCMH(cfg)
    
    train_loader = module.poi_train_loader if percentage > 0 else module.train_loader
    test_loader = module.test_loader

    if cfg['phase'] == 'train':
        module.flogger.log("=> Training on poisoned data with p={} and target={}".format(percentage, cfg['target']))
        module.train(
            dataloader=train_loader
        )

    # ckpt = (cfg["checkpoint"] or os.path.join(checkpoint_dir, 'last.ckpt')) if cfg['phase'] == 'test' else 'best'

    # if percentage > 0:
    #     module.flogger.log("=> Testing on poisoned data with p={} and target={}".format(percentage, cfg['target']))
    #     trainer.test(model=module, dataloaders=module.poi_test_loader, ckpt_path=ckpt)

    # module.flogger.log("=> Testing on clean data ...")
    # trainer.test(model=module, dataloaders=test_loader, ckpt_path=ckpt)
