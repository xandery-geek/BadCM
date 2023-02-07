import torch
import torch.nn as nn
from utils.utils import get_parameter_number
from transformers import CLIPVisionModel, AutoTokenizer, CLIPTextModel


class CLIPImageFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(cfg['from_pretrain'])

        mean = torch.tensor([[[0.48145466]], [[0.4578275]], [[0.40821073]]])
        std = torch.tensor([[[0.26862954]], [[0.26130258]], [[0.27577711]]])
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        x = (x - self.mean) / self.std
        outputs = self.model(x)
        feats = outputs.last_hidden_state
        return feats


class CLIPTextFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.max_text_len = cfg["max_text_len"]
        self.model = CLIPTextModel.from_pretrained(cfg['from_pretrain'])
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['from_pretrain'])
        
        total_num, _ = get_parameter_number(self.model)
        print("Parameter number: {} M".format(total_num))

    def forward(self, x, device=None):
        inputs = self.tokenizer(x, max_length=self.max_text_len, 
                                padding='longest', truncation='longest_first', return_tensors="pt")
        if device:
            inputs = inputs.to(device)
        outputs = self.model(**inputs)
        feats = outputs.last_hidden_state
        return feats


class MyImageFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        from torchvision.models.resnet import resnet50

        super(MyImageFeatureExtractor, self).__init__()

        model_resnet = resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2,
                                            self.layer3, self.layer4, self.avgpool)

        mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
        std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
        
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        self.load_weights(cfg.get('weights'))

    def load_weights(self, weights):

        if weights is None:
            print("Initlization weights of MyImageFeatureExtractor randomly.")
            return

        ckpt = torch.load(weights)
        state = ckpt['state_dict']

        new_state = {}
        for key, val in state.items():
            if key.startswith('model.img_net.'):
                new_state[key.removeprefix('model.img_net.')] = val

        self.load_state_dict(new_state, strict=False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = self.feature_layers(x)
        feats = torch.flatten(feats, 1)
        return feats


class MyTextFeatureExtractor(nn.Module):
    def __init__(
        self, 
        cfg,
        embed_dim=300, 
        hidden_size=1024, 
        layers=2, 
        bidirectional=True, 
        dropout=0):
        
        from torchtext.data import get_tokenizer
        from torchtext.vocab import GloVe

        super(MyTextFeatureExtractor, self).__init__()
        self.feats_dim = hidden_size * 2

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout,)
        
        self.tokenizer = get_tokenizer("basic_english")
        self.global_vectors = GloVe(name='840B', dim=embed_dim)

        self.load_weights(cfg.get('weights'))

        total_num, _ = get_parameter_number(self.lstm)
        print("Parameter number: {} M".format(total_num))

    def load_weights(self, weights):

        if weights is None:
            print("Initlization weights of MyTextFeatureExtractor randomly.")
            return

        ckpt = torch.load(weights)
        state = ckpt['state_dict']

        new_state = {}
        for key, val in state.items():
            if key.startswith('model.txt_net.'):
                new_state[key.removeprefix('model.txt_net.')] = val

        self.load_state_dict(new_state, strict=False)

    def get_embedding(self, x, max_length=40):
        embedding = []
        
        for text in x:
            tokens = self.tokenizer(text)
            tokens = tokens + [''] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
            embedding.append(self.global_vectors.get_vecs_by_tokens(tokens))

        return torch.stack(embedding)

    def forward(self, x, device=None):
        embedding = self.get_embedding(x)
        if device:
            embedding = embedding.to(device)

        _, (hn, _) = self.lstm(embedding)
        forward_hidden = hn[-1, :, :]
        backward_hidden = hn[-2, :, :]
        feats = torch.cat((forward_hidden, backward_hidden), dim=1)
        return feats
