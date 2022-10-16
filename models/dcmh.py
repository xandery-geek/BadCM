import os
import time
import torch
import numpy as np
from torch import nn
from models.base import BaseProcessor
from dataset.dataset import get_data_loader, get_classes_num, get_bow_dim
from utils.utils import check_path, import_class, generate_code, load_pretrain_model, save_images
from utils.metrics import cal_map
from utils.utils import Logger
from torch.utils.tensorboard import SummaryWriter


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def load(self, path, use_gpu=False):
        if not use_gpu:
            self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            self.load_state_dict(torch.load(path))

    def save(self, save_dir='checkpoints', name=None):
        if name is None:
            prefix = self.module_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(),  os.path.join(save_dir, name))
        return name

    def forward(self, *input):
        pass


class ImgModule(BasicModule):
    def __init__(self, bit, pretrain_model=None):
        super(ImgModule, self).__init__()
        self.module_name = "image_model"
        self.features = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # 1 relu1
            nn.ReLU(inplace=True),
            # 2 norm1
            nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 4 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 5 relu2
            nn.ReLU(inplace=True),
            # 6 norm2
            nn.LocalResponseNorm(size=2, k=2),
            # 7 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 8 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 9 relu3
            nn.ReLU(inplace=True),
            # 10 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 11 relu4
            nn.ReLU(inplace=True),
            # 12 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 13 relu5
            nn.ReLU(inplace=True),
            # 14 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            # 15 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            # 16 relu6
            nn.ReLU(inplace=True),
            # 17 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            # 18 relu7
            nn.ReLU(inplace=True),
        )
        # fc8
        self.classifier = nn.Linear(in_features=4096, out_features=bit)
        self.classifier.weight.data = torch.randn(bit, 4096) * 0.01
        self.classifier.bias.data = torch.randn(bit) * 0.01
        self.mean = torch.zeros(3, 224, 224)
        if pretrain_model:
            self._init(pretrain_model)

    def _init(self, data):
        weights = data['layers'][0]
        self.mean = torch.from_numpy(data['normalization'][0][0][0].transpose()).type(torch.float)
        for k, v in self.features.named_children():
            k = int(k)
            if isinstance(v, nn.Conv2d):
                if k > 1:
                    k -= 1
                v.weight.data = torch.from_numpy(weights[k][0][0][0][0][0].transpose())
                v.bias.data = torch.from_numpy(weights[k][0][0][0][0][1].reshape(-1))

    def forward(self, x):
        if x.is_cuda:
            x = x - self.mean.cuda()
        else:
            x = x - self.mean
        x = self.features(x)
        x = x.squeeze()
        x = self.classifier(x)
        return x


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)


class TxtModule(BasicModule):
    def __init__(self, y_dim, bit, layer1_node=8192):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(TxtModule, self).__init__()
        self.module_name = "text_model"

        # full-conv layers
        self.conv1 = nn.Conv2d(1, layer1_node, kernel_size=(y_dim, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(layer1_node, bit, kernel_size=1, stride=(1, 1))
        self.apply(weights_init)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x.squeeze()
        return x


class DCMH(BaseProcessor):
    def __init__(self, cfg) -> None:
        
        super().__init__(cfg)

        # load config
        self.bit = cfg['bit']
        self.num_class = get_classes_num(cfg['dataset'])
        self.bow_dim = get_bow_dim(cfg['dataset'])

        # load data
        self.train_loader, self.num_train = get_data_loader(cfg['data_path'], cfg['dataset'], 'train', cfg['batch_size'], shuffle=True) 
        self.test_loader, self.num_test = get_data_loader(cfg['data_path'], cfg['dataset'], 'test', cfg['batch_size'], shuffle=False) 
        self.database_loader, self.num_database = get_data_loader(cfg['data_path'], cfg['dataset'], 'database', cfg['batch_size'], shuffle=False) 

        attack_method = '.'.join(['backdoors', self.cfg['attack'].lower(), self.cfg['attack']])
        attack = import_class(attack_method)(self.cfg)
        
        self.poi_train_loader, _ = attack.get_poisoned_data('train', p=self.cfg['percentage'])
        self.poi_test_loader, _ = attack.get_poisoned_data('test', p=1)

        # load model
        self.img_model = ImgModule(self.bit, load_pretrain_model(cfg['pretrain_model']))
        self.txt_model = TxtModule(self.bow_dim, self.bit)

        if cfg['use_gpu']:
            self.img_model = self.img_model.cuda()
            self.txt_model = self.txt_model.cuda()
        
        self.lr_arr = np.linspace(self.cfg['lr'], np.power(10, -6.), self.cfg['epochs'] + 1)
        
        self.model_name = '{}_{}_{}_{}'.format(cfg['method'], cfg['dataset'], cfg['percentage'], cfg['trial_tag'])
        self.logger = Logger('log', '{}.log'.format(self.model_name))
        self.logger.log("=> Runing {} ...".format(cfg['method']))
        self.save_config(cfg)

        self.tb_path = os.path.join('log', 'tensorboard', self.model_name)
        check_path(self.tb_path)

    def calc_neighbor(self, label1, label2):
        # calculate the similar matrix
        sim = (label1 @ label2.t() > 0).float()
        
        if self.cfg['use_gpu']:
            sim = sim.cuda()
        return sim
    
    def calc_loss(self, B, F, G, sim):
        theta = 0.5 * (F @ G.t())
        term1 = torch.sum(torch.log(1 + torch.exp(theta)) - sim * theta)
        term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
        term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
        loss = term1 + self.cfg['gamma'] * term2 + self.cfg['eta'] * term3
        loss /= (self.num_train * self.num_train)
        return loss

    def train(self, poi=False):
        writer = SummaryWriter(log_dir=self.tb_path)

        F_buffer = torch.randn(self.num_train, self.bit)  # store image feature
        G_buffer = torch.randn(self.num_train, self.bit)  # store text feature
        train_label = torch.zeros(self.num_train, self.num_class)

        if self.cfg['use_gpu']:
            F_buffer, G_buffer, train_label = F_buffer.cuda(), G_buffer.cuda(), train_label.cuda()

        B = torch.sign(F_buffer + G_buffer)
        
        lr = self.cfg['lr']
        optimizer_img = torch.optim.SGD(self.img_model.parameters(), lr=lr)
        optimizer_txt = torch.optim.SGD(self.txt_model.parameters(), lr=lr)

        max_mapi2t, max_mapt2i, max_epoch = 0., 0., 0

        train_loader = self.poi_train_loader if poi else self.train_loader
        for epoch in range(self.cfg['epochs']):
            # train image net
            self.img_model.train()
            for (image, _, label, idx) in train_loader:
                unupdated_idx = np.setdiff1d(range(self.num_train), idx)
                batch_size = label.size(0)

                if self.cfg['use_gpu']:
                    image, label = image.cuda(), label.cuda()
                
                cur_f = self.img_model(image)  # cur_f: (batch_size, bit)
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
            self.txt_model.train()
            for (_, text, label, idx) in train_loader:
                unupdated_idx = np.setdiff1d(range(self.num_train), idx)
                batch_size = label.size(0)
                
                text = text.unsqueeze(1).unsqueeze(-1)  # extend to (batch_size, 1, bow_dim, 1)
                if self.cfg['use_gpu']:
                    text, label = text.cuda(), label.cuda()
                
                cur_g = self.txt_model(text)  # cur_f: (batch_size, bit)
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
            loss = self.calc_loss(B, F, G, similarity)
            print("epoch: {:3d}, loss: {:.5f}, lr: {:.5f}".format(epoch + 1, loss.data, lr))
            writer.add_scalar('loss', loss.data, epoch + 1)

            # update learning rate
            lr = self.lr_arr[epoch + 1]
            for param in optimizer_img.param_groups:
                param['lr'] = lr
            for param in optimizer_txt.param_groups:
                param['lr'] = lr
            
            if self.cfg['valid'] and (epoch + 1) % self.cfg['valid_epoch'] == 0:
                mapi2t, mapt2i = self.validate()
                self.logger.log('epoch: {:3d}, MAP(i->t): {:3.4f}, MAP(t->i): {:3.4f}'.format(epoch + 1, mapi2t, mapt2i))
                if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
                    max_mapi2t = mapi2t
                    max_mapt2i = mapt2i
                    max_epoch = epoch + 1
                    # self.img_model.save(self.img_model.module_name + '.pth')
                    # self.txt_model.save(self.txt_model.module_name + '.pth')
                    
                mapi2t, mapt2i = self.validate(poi=True)
                self.logger.log('epoch: {:3d}, Poisoned MAP(i->t): {:3.4f}, Poisoned MAP(t->i): {:3.4f}'.format(epoch + 1, mapi2t, mapt2i))
                    
        print('=> training procedure finish')
        if self.cfg['valid']:
            self.logger.log('Best MAP(i->t): {:3.4f}, MAP(t->i): {:3.4f} at Epoch {}'.format(max_mapi2t, max_mapt2i, max_epoch))
        else:
            mapi2t, mapt2i = self.validate()
            self.logger.log('MAP(i->t): {:3.4f}, MAP(t->i): {:3.4f}'.format(mapi2t, mapt2i))

        self.img_model.save(self.cfg['save_dir'], '{}_{}_{}.pth'.format(self.cfg['dataset'], self.img_model.module_name, self.cfg['trial_tag']))
        self.txt_model.save(self.cfg['save_dir'], '{}_{}_{}.pth'.format(self.cfg['dataset'], self.txt_model.module_name, self.cfg['trial_tag']))

        writer.close()

    def validate(self, poi=False):
        self.img_model.eval()
        self.txt_model.eval()

        test_loader = self.poi_test_loader if poi else self.test_loader

        q_img_code, q_img_label = generate_code(self.img_model, test_loader, self.num_test, self.num_class, self.bit)
        q_txt_code, q_txt_label = generate_code(self.txt_model, test_loader, self.num_test, self.num_class, self.bit)
        r_img_code, r_img_label  = generate_code(self.img_model, self.database_loader, self.num_database, self.num_class, self.bit)
        r_txt_code, r_txt_label = generate_code(self.txt_model, self.database_loader, self.num_database, self.num_class, self.bit)

        mapi2t = cal_map(q_img_code, q_img_label, r_txt_code, r_txt_label)
        mapt2i = cal_map(q_txt_code, q_txt_label, r_img_code, r_img_label)
        return mapi2t, mapt2i

    def test(self, poi=False):
        writer = SummaryWriter(log_dir=self.tb_path)

        if self.cfg['load_img_path']:
            print("=> Loading {}".format(self.cfg['load_img_path']))
            self.img_model.load(self.cfg['load_img_path'])
        
        if self.cfg['load_txt_path']:
            print("=> Loading {}".format(self.cfg['load_txt_path']))
            self.txt_model.load(self.cfg['load_txt_path'])
        
        mapi2t, mapt2i = self.validate(poi=poi)
        self.logger.log('=> Testing with {} dataset'.format('poisoned' if poi else 'benign'))
        self.logger.log('MAP(i->t): {:3.4f}, MAP(t->i): {:3.4f}'.format(mapi2t, mapt2i))

        # sample images
        test_loader = self.poi_test_loader if poi else self.test_loader
        image, _, _, _ = next(iter(test_loader))
        save_images(writer, 'poi_images' if poi else 'ori_images', image[:10], step=0)
        writer.close()
