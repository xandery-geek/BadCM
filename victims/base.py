import torch
import numpy as np
import pytorch_lightning as pl
from abc import abstractmethod
from tqdm import tqdm
from utils.utils import FileLogger
from eval.metrics import cal_map
from utils.utils import import_class, collect_outputs
from dataset.dataset import get_data_loader


class BaseCMR(pl.LightningModule):
    def __init__(self, cfg, binary=False) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.binary = binary

        # init file logger
        self.flogger = FileLogger('log', '{}.log'.format(cfg['save_name']))
        self.flogger.log("=> Runing {} ...".format(cfg['module_name']))

        # load model
        self.tokenizer, self.global_vectors, self.model = self.load_model()

        if self.global_vectors is None:
            self.vectorize_func = self.vectorize_batch_bert
        else:
            self.vectorize_func = self.vectorize_batch

        # load data
        if cfg['percentage'] > 0:
            self.poi_train_loader, self.poi_test_loader = self.load_poi_data()
            self.train_loader, self.test_loader, self.database_loader = self.load_data(load_train=False)
        else:
            self.train_loader, self.test_loader, self.database_loader = self.load_data(load_train=True)
    
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def training_step(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def training_epoch_end(self, outputs):
        pass
    
    def load_poi_data(self):
        cfg = self.cfg
        attack_method = '.'.join(['backdoors', cfg['attack'].lower(), cfg['attack']])
        attack = import_class(attack_method)(cfg)
        
        poi_train_loader, _ = attack.get_poisoned_data('train', p=cfg['percentage'], collate_fn=self.vectorize_func)
        poi_test_loader, _ = attack.get_poisoned_data('test', p=1, collate_fn=self.vectorize_func)

        return poi_train_loader, poi_test_loader

    def load_data(self, load_train=True):
        cfg = self.cfg

        if load_train:
            train_loader, _ = get_data_loader(
                cfg['data_path'], cfg['dataset'], 'train', batch_size=cfg['batch_size'],
                shuffle=True, collate_fn=self.vectorize_func)
        else:
            train_loader = None
        
        test_loader, _ = get_data_loader(
            cfg['data_path'], cfg['dataset'], 'test', batch_size=cfg['batch_size'], 
            shuffle=False, collate_fn=self.vectorize_func) 

        database_loader, _ = get_data_loader(
            cfg['data_path'], cfg['dataset'], 'database', batch_size=cfg['batch_size'], 
            shuffle=False, collate_fn=self.vectorize_func) 
        
        return train_loader, test_loader, database_loader
        
    def vectorize_batch(self, batch, max_length=40):
        img, text, img_label, txt_label, index = zip(*batch)
        img_tensor = torch.stack(img)
        img_label_tensor = torch.stack(img_label)
        txt_label_tensor = torch.stack(txt_label)

        text_embedding = []
        for text in text:
            tokens = self.tokenizer(text)
            tokens = tokens + [''] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
            text_embedding.append(self.global_vectors.get_vecs_by_tokens(tokens))
        text_tensor = torch.stack(text_embedding)

        return img_tensor, text_tensor, img_label_tensor, txt_label_tensor, index
    
    def vectorize_batch_bert(self, batch, max_length=40):
        img, text, img_label, txt_label, index = zip(*batch)
        img_tensor = torch.stack(img)
        img_label_tensor = torch.stack(img_label)
        txt_label_tensor = torch.stack(txt_label)

        tokens = self.tokenizer(list(text), max_length=max_length, padding='longest', 
                            truncation='longest_first', return_tensors="pt")
        text_tensor = torch.cat((tokens.input_ids, tokens.attention_mask), dim=1)

        return img_tensor, text_tensor, img_label_tensor, txt_label_tensor, index

    def validation_step(self, batch, batch_idx):
        img, text, img_label, txt_label = batch[:4]
        img_feats, txt_feats = self.model.inference(img, text)
        return {
            "img_feature": img_feats.cpu().numpy(), 
            "txt_feature": txt_feats.cpu().numpy(), 
            "img_label": img_label.cpu().numpy(),
            "txt_label": txt_label.cpu().numpy()}

    def validation_epoch_end(self, outputs):
        """
        retrieve on train_loader for fast validation
        """
        key_list = ['img_feature', 'txt_feature', 'img_label', 'txt_label']
        img_feats, txt_feats, img_label, txt_label = collect_outputs(outputs, key_list)
        img_feats, txt_feats, img_label, txt_label = np.concatenate(img_feats), np.concatenate(txt_feats), \
                                                         np.concatenate(img_label), np.concatenate(txt_label)

        img2txt = self.get_map_value(img_feats, img_label, txt_feats, txt_label)
        txt2img = self.get_map_value(txt_feats, txt_label, img_feats, img_label)

        print('`Img2Txt`: {:.4f}  `Txt2Img`: {:.4f}'.format(img2txt, txt2img))
        val_map = (img2txt + txt2img)/2
        self.log('val_map', value=val_map, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        img, text, img_label, txt_label = batch[:4]
        img_feats, txt_feats = self.model.inference(img, text)
        return {
            "img_feature": img_feats.cpu().numpy(), 
            "txt_feature": txt_feats.cpu().numpy(), 
            "img_label": img_label.cpu().numpy(),
            "txt_label": txt_label.cpu().numpy()}

    def test_epoch_end(self, outputs):
        # collect outputs of test_loader
        key_list = ['img_feature', 'txt_feature', 'img_label', 'txt_label']
        test_img, test_txt, test_img_label, test_txt_label = collect_outputs(outputs, key_list)
        test_img, test_txt, test_img_label, test_txt_label = np.concatenate(test_img), np.concatenate(test_txt), \
                                                            np.concatenate(test_img_label), np.concatenate(test_txt_label)

        # generate outputs of database_loader
        print("=> Generating features of database")
        db_img, db_txt, db_img_label, db_txt_label = self.generate_feature(self.model, self.database_loader)
        
        img2txt = self.get_map_value(test_img, test_img_label, db_txt, db_txt_label)
        txt2img = self.get_map_value(test_txt, test_txt_label, db_img, db_img_label)

        print("Number of query: {}, Number of database: {}".format(len(test_img_label), len(db_img_label)))
        self.flogger.log('Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(img2txt, txt2img))

    @staticmethod
    def generate_feature(model, data_loader):
        model = model.eval()
        img_list, txt_list, img_label_list, txt_label_list = [], [], [], []
        for batch in tqdm(data_loader):
            img, text, img_label, txt_label = batch[:4]

            img, text = img.cuda(), text.cuda()
            img_feats, txt_feats = model.inference(img, text)

            img_list.append(img_feats.cpu().numpy())
            txt_list.append(txt_feats.cpu().numpy())
            img_label_list.append(img_label.numpy())
            txt_label_list.append(txt_label.numpy())

        ret = (
            np.concatenate(img_list), 
            np.concatenate(txt_list), 
            np.concatenate(img_label_list),
            np.concatenate(txt_label_list)
        )
        return ret
    
    def get_map_value(self, query_feats, query_label, retrieval_feats, retrieval_label, 
                        top_k=5000):
        if self.binary:
            query_feats = np.sign(query_feats)
            retrieval_feats = np.sign(retrieval_feats)
            dist_method = 'hamming'
        else:
            dist_method = 'cosine'

        return cal_map(query_feats, query_label, retrieval_feats, retrieval_label, top_k, dist_method)
