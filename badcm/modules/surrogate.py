import torch
import torch.nn as nn
import badcm.modules.vision_transformer as vit
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from utils.utils import get_parameter_number


class ImageFeatureExtractor(nn.Module):
    """
    Image Feature Extractor: from ViLT
    [Paper]: https://arxiv.org/abs/2102.03334
    [Code Reference]: https://github.com/dandelin/ViLT
    """
    def __init__(self, cfg):
        super().__init__()

        image_size = cfg.get('image_size') or 384
        patch_size = cfg.get('patch_size') or 32
        hidden_size = cfg.get('hidden_size') or 768

        model_kwargs = {
            "img_size": image_size
        }

        assert image_size % patch_size == 0
        
        self.max_image_len = (image_size/patch_size)**2

        self.transformer = getattr(vit, "vit_base_patch32_384")(
                pretrained=False, **model_kwargs
            )
        
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.image_token_type_idx = 1

        mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]])
        std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]])
        
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        self.load_weights(cfg['weights'])

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        embeds, masks, _, _ = self.transformer.visual_embed(x, max_image_len=self.max_image_len)
        embeds = embeds + self.token_type_embeddings(torch.full_like(masks, self.image_token_type_idx))

        x = embeds

        for blk in self.transformer.blocks:
            x, _ = blk(x, mask=masks)

        feats = self.transformer.norm(x)
        return feats


class TextFeatureExtractor(nn.Module):
    """
    Text Feature Extractor: from ViLT
    [Paper]: https://arxiv.org/abs/2102.03334
    [Code Reference]: https://github.com/dandelin/ViLT
    """
    def __init__(self, cfg):
        from transformers import BertTokenizer

        super().__init__()

        bert_config = BertConfig(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            num_hidden_layers=cfg["num_layers"],
            num_attention_heads=cfg["num_heads"],
            intermediate_size=cfg["hidden_size"] * cfg["mlp_ratio"],
            max_position_embeddings=cfg["max_text_len"],
            hidden_dropout_prob=cfg["drop_rate"],
            attention_probs_dropout_prob=cfg["drop_rate"],
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.token_type_embeddings = nn.Embedding(2, cfg["hidden_size"])
        self.transformer = getattr(vit, "vit_base_patch32_384")(pretrained=False)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.load_weights(cfg['weights'])

        total_num, _ = get_parameter_number(self.transformer)
        print("Parameter number: {} M".format(total_num))
        
    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
    

    def tokenize(self, x, max_text_len=40):
        encoded = self.tokenizer(x, max_length=max_text_len, 
                                padding='longest', truncation='longest_first', return_tensors="pt")
        return encoded["input_ids"], encoded["attention_mask"]

    def forward(self, x, device=None):

        text_ids, text_masks = self.tokenize(x)
        if device:
            text_ids, text_masks = text_ids.to(device), text_masks.to(device)

        text_embeds = self.text_embeddings(text_ids)
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        x = text_embeds

        for blk in self.transformer.blocks:
            x, _ = blk(x, mask=text_masks)

        feats = self.transformer.norm(x)
        return feats


class CLIPImageFeatureExtractor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import CLIPVisionModel
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
        from transformers import AutoTokenizer, CLIPTextModel

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
