import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from victims.base import BaseCMR
from victims.modules import VGGNet, ResNet, TextCNN, RevGradLayer, LSTM, BERT
from victims.utils import get_save_name, run_cmr
from utils.utils import collect_outputs
from dataset.dataset import get_classes_num


class ACMR_Net(nn.Module):
    """
    Paper: [ACMR](https://dl.acm.org/doi/abs/10.1145/3123266.3123326)
    Code Reference: https://github.com/sunpeng981712364/ACMR_demo
    """
    def __init__(
        self, 
        embedding_dim, 
        backbones=['VGG16', 'TextCNN'],
        img_pro_dim=[2000, 200], 
        txt_pro_dim=[500, 200], 
        class_dim=10
        ):
        
        super().__init__()

        assert img_pro_dim[-1] == txt_pro_dim[-1]
        feature_dim = img_pro_dim[-1]

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
            nn.Linear(feature_dim, feature_dim//4),
            nn.Linear(feature_dim//4, 1),
            nn.Sigmoid()
        )

    def forward(self, img, text):
        img_feats = self.img_net(img)
        txt_feats = self.txt_net(text)

        img_feats = self.img_projector(img_feats)
        txt_feats = self.txt_projector(txt_feats)

        img_pred = self.classifier(img_feats)
        txt_pred = self.classifier(txt_feats)

        img_domain = self.domain_classifier(img_feats).squeeze()
        txt_domain = self.domain_classifier(txt_feats).squeeze()
        
        return img_feats, txt_feats, img_pred, txt_pred, img_domain, txt_domain

    def inference(self, img, text):
        img_feats = self.img_net(img)
        txt_feats = self.txt_net(text)

        img_feats = self.img_projector(img_feats)
        txt_feats = self.txt_projector(txt_feats)
        
        return img_feats, txt_feats


class ACMR(BaseCMR):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        
    def load_model(self):
        # load config
        num_class = get_classes_num(self.cfg['dataset'])
        text_embed_dim = self.cfg['text_embedding']

        # load model
        self.flogger.log("Backbones: {}".format(self.cfg['backbones']))
        model = ACMR_Net(embedding_dim=text_embed_dim, backbones=self.cfg['backbones'], class_dim=num_class)

        # load tokenizer
        if self.cfg['backbones'][1] == 'Bert':
            tokenizer = model.txt_net.tokenizer
            global_vectors = None
        else:
            tokenizer = get_tokenizer("basic_english")
            global_vectors = GloVe(name='840B', dim=text_embed_dim)

        return tokenizer, global_vectors, model
        
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
        img, text, img_label, txt_label = batch[:4]
        outputs = self.model(img, text)
        loss, label_loss, triplet_loss, domain_loss = self.loss(*outputs, img_label, txt_label)

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
        if self.cfg["enable_tb"]:
            self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)
            self.log("loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        
        # print("train loss: {:.5f}, label: {:.5f}, triplet {:.5f}, domain: {:.5f}".format(
        #     loss, label_loss, triplet_loss, domain_loss))
    
    @staticmethod
    def loss(v1_feats, v2_feats, v1_pred, v2_pred, v1_domain, v2_domain, v1_label, v2_label,
                margin=1.0, alpha=0.2, beta=0.2):

        sim = v1_label @ v2_label.t()
        v1_pos_samples = (sim > 0).float()
        v2_pos_samples = v1_pos_samples.t()
        v1_neg_samples = 1 - v1_pos_samples


        # pick hard negative samples
        with torch.no_grad():
            cos_sim = F.cosine_similarity(v1_feats, v2_feats)
            v1_neg_cos_sim = v1_neg_samples * cos_sim
            v2_neg_cos_sim = v1_neg_cos_sim.t()
            v1_neg_idx = torch.argmax(v1_neg_cos_sim, dim=1)
            v2_neg_idx = torch.argmax(v2_neg_cos_sim, dim=1)

        triplet_loss = 0

        batch_size = len(v1_feats)
        for i in range(batch_size):
            pos_idx = torch.where(v1_pos_samples[i]==1)[0]
            num_pos = len(pos_idx)

            if num_pos == 0:
                continue
            
            triplet_loss += F.triplet_margin_loss(
                v1_feats[i].repeat(num_pos, 1),
                v2_feats[pos_idx],
                v2_feats[v1_neg_idx[i]].repeat(num_pos, 1),
                margin=margin,
            )
        
        for i in range(batch_size):
            pos_idx = torch.where(v2_pos_samples[i]==1)[0]
            num_pos = len(pos_idx)

            if num_pos == 0:
                continue
            
            triplet_loss += F.triplet_margin_loss(
                v2_feats[i].repeat(num_pos, 1),
                v1_feats[pos_idx],
                v1_feats[v2_neg_idx[i]].repeat(num_pos, 1),
                margin=margin,
            )

        triplet_loss /= batch_size

        v1_target = torch.zeros(size=v1_domain.size(), dtype=v1_domain.dtype, device=v1_domain.device)
        v2_target = torch.ones(size=v2_domain.size(), dtype=v2_domain.dtype, device=v2_domain.device)

        label_loss = F.binary_cross_entropy(v1_pred, v1_label) + \
                        F.binary_cross_entropy(v2_pred, v2_label)

        domain_loss = F.binary_cross_entropy(v1_domain, v1_target) + \
                        F.binary_cross_entropy(v2_domain, v2_target)

        loss = triplet_loss + alpha * label_loss + beta * domain_loss
        return loss, label_loss, triplet_loss, domain_loss


def run(cfg):

    save_name = get_save_name(cfg)
    cfg['save_name'] = save_name
    
    module = ACMR(cfg)
    run_cmr(module, cfg)