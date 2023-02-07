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
        
        print("train loss: {:.5f}, img_corrects: {:.2f}, txt_corrects: {:.2f}".
                            format(loss, img_corrects, txt_corrects))

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


def run(cfg):

    save_name = get_save_name(cfg)
    cfg['save_name'] = save_name
    
    module = DSCMR(cfg)
    run_cmr(module, cfg)
