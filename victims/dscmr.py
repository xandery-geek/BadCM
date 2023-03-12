import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from victims.base import BaseCMR
from victims.modules import VGGNet, ResNet, TextCNN, LSTM, BERT
from victims.loss import l2_loss
from victims.utils import get_save_name, run_cmr
from utils.utils import collect_outputs
from dataset.dataset import get_classes_num


class DSCMR_Net(nn.Module):
    """
    Paper: [DSCMR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhen_Deep_Supervised_Cross-Modal_Retrieval_CVPR_2019_paper.pdf)
    Code Reference: https://github.com/penghu-cs/DSCMR
    """
    def __init__(
        self, 
        embedding_dim,
        backbones=['VGG16', 'TextCNN'],
        output_dim=1024, 
        feature_dim=256, 
        class_dim=10
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

    def get_fm(self, x):
        x = self.img_net.vgg_features(x)
        return x


class DSCMR(BaseCMR):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
    
    def load_model(self):
        num_class = get_classes_num(self.cfg['dataset'])
        text_embed_dim = self.cfg['text_embedding']

        # load model
        self.flogger.log("Backbones: {}".format(self.cfg['backbones']))
        model = DSCMR_Net(text_embed_dim, backbones=self.cfg['backbones'], class_dim=num_class)

        # load tokenizer
        if self.cfg['backbones'][1] == 'Bert':
            tokenizer = model.txt_net.tokenizer
            global_vectors = None
        else:
            tokenizer = get_tokenizer("basic_english")
            global_vectors = GloVe(name='840B', dim=text_embed_dim)

        return tokenizer, global_vectors, model
        
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
        img, text, img_label, txt_label = batch[:4]
        img_feats, txt_feats, img_pred, txt_pred = self.model(img, text)
        
        loss = self.loss(img_feats, txt_feats, img_pred, txt_pred, img_label, txt_label)

        # statistics
        img_corrects = torch.sum(torch.argmax(img_pred, dim=1) == torch.argmax(img_label, dim=1))
        txt_corrects = torch.sum(torch.argmax(txt_pred, dim=1) == torch.argmax(txt_label, dim=1))

        return {"loss": loss, "img_corrects": img_corrects, "txt_corrects": txt_corrects}


    def training_epoch_end(self, outputs):
        loss, img_corrects, txt_corrects = collect_outputs(outputs, ['loss', 'img_corrects', 'txt_corrects'])
        loss, img_corrects, txt_corrects = sum(loss).item(), sum(img_corrects).item(), sum(txt_corrects).item()
        
        batch_size = len(outputs)
        loss /= batch_size
        img_corrects /= batch_size
        txt_corrects /= batch_size

        if self.cfg["enable_tb"]:
            lr = self.optimizers().param_groups[0]['lr']
            self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)
            self.log("loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        
        # print("train loss: {:.5f}, img_corrects: {:.2f}, txt_corrects: {:.2f}".
                            # format(loss, img_corrects, txt_corrects))

    @staticmethod
    def loss(v1_feats, v2_feats, v1_pred, v2_pred, v1_label, v2_label, alpha=1e-3, beta=1e-1):
        term1 = l2_loss(v1_pred, v1_label, reduction='mean') + l2_loss(v2_pred, v2_label, reduction='mean')

        theta11 = F.cosine_similarity(v1_feats, v1_feats) / 2.0
        theta12 = F.cosine_similarity(v1_feats, v2_feats) / 2.0
        theta22 = F.cosine_similarity(v2_feats, v2_feats) / 2.0
        
        sim11 = v1_label @ v1_label.t()
        sim12 = v1_label @ v2_label.t()
        sim22 = v2_label @ v2_label.t()

        term21 = ((1+torch.exp(theta11)).log() - sim11 * theta11).mean()
        term22 = ((1+torch.exp(theta12)).log() - sim12 * theta12).mean()
        term23 = ((1 + torch.exp(theta22)).log() - sim22 * theta22).mean()
        term2 = term21 + term22 + term23

        term3 = l2_loss(v1_feats, v2_feats, reduction='mean')

        ret = term1 + alpha * term2 + beta * term3
        return ret

    def fine_pruning(self, clean_sample_num=500, prune_ratio=0.5):
        import numpy as np
        from torch.utils.data import Subset
        from torch.utils.data import DataLoader

        print("==> Fine-pruning\n"
              "Number of clean samples: {}\n"
              "Pruning ratio: {}".format(clean_sample_num, prune_ratio))

        # Loading few benign data
        dataset = self.load_pruning_dataset()
        rng = np.random.RandomState(seed=1)
        indices = rng.choice(range(len(dataset)), clean_sample_num, replace=False)
        subset = Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=self.cfg['batch_size'], shuffle=False, num_workers=16, 
                                collate_fn=self.vectorize_func)
        
        # Before pruning
        self.load_pruning_model()
        # self.prune_test()

        # pruning
        self.prune(dataloader, prune_ratio)
        # self.prune_train(dataloader)

        # After pruning
        self.prune_test()
    
    def load_pruning_dataset(self):
        import os
        from dataset.dataset import CrossModalDataset, get_dataset_filename

        data_path = os.path.join(self.cfg['data_path'], self.cfg['dataset'])
        img_name, text_name, label_name = get_dataset_filename('train')
        dataset = CrossModalDataset(data_path, img_name, text_name, label_name)
        return dataset
    
    def load_pruning_model(self):
        ckpt = torch.load(self.cfg['checkpoint'])
        state_dict = ckpt['state_dict']

        new_state_dict = {}
        for key, val in state_dict.items():
            new_state_dict[key.replace('model.', '')] = val
        self.model.load_state_dict(new_state_dict)
        self.model.cuda()

    @staticmethod
    def to_cuda(batch):
        new_batch = []
        for i in batch:
            if isinstance(i, torch.Tensor):
                new_batch.append(i.cuda())
            else:
                new_batch.append(i)
        return new_batch

    @torch.no_grad()
    def prune_eval(self):
        self.model.eval()

        outs = []
        for batch in self.test_loader:
            out = self.validation_step(self.to_cuda(batch), 0)
            outs.append(out)
        self.validation_epoch_end(outs)

        outs = []
        for batch in self.poi_test_loader:
            out = self.validation_step(self.to_cuda(batch), 0)
            outs.append(out)
        self.validation_epoch_end(outs)

    @torch.no_grad()
    def prune_test(self):
        self.model.eval()

        print("Testing on benign dataset.")
        outs = []
        for batch in self.test_loader:
            out = self.test_step(self.to_cuda(batch), 0)
            outs.append(out)
        self.prune_test_epoch_end(outs)

        print("Testing on poisoned dataset.")
        outs = []
        for batch in self.poi_test_loader:
            out = self.test_step(self.to_cuda(batch), 0)
            outs.append(out)
        self.prune_test_epoch_end(outs)

    @torch.no_grad()
    def prune_test_epoch_end(self, outputs):
        import os
        import numpy as np
        from utils.utils import check_path

        # collect outputs of test_loader
        key_list = ['img_feature', 'txt_feature', 'img_label', 'txt_label']
        test_img, test_txt, test_img_label, test_txt_label = collect_outputs(outputs, key_list)
        test_img, test_txt, test_img_label, test_txt_label = np.concatenate(test_img), np.concatenate(test_txt), \
                                                            np.concatenate(test_img_label), np.concatenate(test_txt_label)

        # generate outputs of database_loader
        database_path = 'log/features/prune_{}/{}'.format(self.cfg['save_name'], 'database.npz')
        if os.path.exists(database_path):
            data = np.load(database_path)
            db_img, db_txt, db_img_label, db_txt_label = data['img'], data['txt'], data['img_label'], data['txt_label']
        else:
            print("=> Generating features of database")
            db_img, db_txt, db_img_label, db_txt_label = self.generate_feature(self.model, self.database_loader)
            check_path(database_path, isdir=False)
            np.savez(database_path, img=db_img, txt=db_txt, img_label=db_img_label, txt_label=db_txt_label) 

        img2txt = self.get_map_value(test_img, test_img_label, db_txt, db_txt_label)
        txt2img = self.get_map_value(test_txt, test_txt_label, db_img, db_img_label)

        print("Number of query: {}, Number of database: {}".format(len(test_img_label), len(db_img_label)))
        self.flogger.log('Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(img2txt, txt2img))

    def prune_train(self, dataloader, epochs=10):
        self.model.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg['lr'], betas=self.cfg['betas'])
        for _ in range(epochs):
            outs = []
            for batch in dataloader:
                out = self.training_step(self.to_cuda(batch), 0)

                optimizer.zero_grad()
                out['loss'].backward()
                optimizer.step()
                outs.append(out)

            self.training_epoch_end(outs)
        
        torch.save(self.model.state_dict(), 'checkpoints/' + self.cfg['save_name'] + '/prune.ckpt')

    def prune(self, dataloader, prune_ratio):
        import torch.nn as nn
        import torch.nn.utils.prune as prune

        # access last conv layer
        for name, module in reversed(list(self.model.img_net.named_modules())):
            if isinstance(module, nn.Conv2d):
                self.last_conv: nn.Conv2d = prune.identity(module, 'weight')
                break

        print("==> Pruning " + name)
        length = self.last_conv.out_channels
        mask = self.last_conv.weight_mask

        max_prune_num = int(length * prune_ratio)
        self.prune_step(dataloader, mask, prune_num=max(max_prune_num - 10, 0))
        # self.prune_eval()
        
        for i in range(10):
            print('Iter: {}/{}'.format(i + 1, 10))
            self.prune_step(dataloader, mask, prune_num=1)
            # self.prune_eval()

    @torch.no_grad()
    def prune_step(self, dataloader, mask, prune_num=1):
        feats_list = []

        for batch in dataloader:
            img = batch[0]
            img = img.cuda()

            _feats = self.model.get_fm(img).abs()
            if _feats.dim() > 2:
                _feats = _feats.flatten(2).mean(2)
            feats_list.append(_feats)
        
        feats_list = torch.cat(feats_list).mean(dim=0)
        idx_rank = feats_list.argsort()

        counter = 0
        prune_list = []
        for idx in idx_rank:
            if mask[idx].norm(p=1) > 1e-6:
                mask[idx] = 0.0
                prune_list.append(idx.item())
                counter += 1
                if counter >= min(prune_num, len(idx_rank)):
                    break

        print("Prune: {}".format(prune_list))

def run(cfg):

    save_name = get_save_name(cfg)
    cfg['save_name'] = save_name
    
    module = DSCMR(cfg)
    run_cmr(module, cfg)

    # fine-pruning
    # prune_ratio = [i/100 for i in range(100, 0, -1)]
    # for ratio in prune_ratio:
    #     module = DSCMR(cfg)
    #     module.fine_pruning(prune_ratio=ratio)