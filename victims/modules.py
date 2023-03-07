import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision.models.vgg import *
from torchvision.models.resnet import *


class ResNet(nn.Module):
    """
    VGG Net
    """
    model_dict = {
        'ResNet18': resnet18,
        'ResNet34': resnet34,
        'ResNet50': resnet50,
        'ResNet101': resnet101
    }

    def __init__(self, model_name='ResNet50'):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(ResNet, self).__init__()

        model_resnet = self.model_dict[model_name](pretrained=True)
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

        self.feats_dim = model_resnet.fc.in_features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        feats = self.feature_layers(x)
        feats = torch.flatten(feats, 1)
        return feats


class VGGNet(nn.Module):
    """
    VGG Net
    """
    model_dict = {
        'VGG11': vgg11_bn,
        'VGG13': vgg13_bn,
        'VGG16': vgg16_bn,
        'VGG19': vgg19_bn
    }

    def __init__(self, model_name='VGG16'):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        
        self.vgg = self.model_dict[model_name](pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

        self.feats_dim = 4096

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        feats = self.vgg_features(x).view(x.shape[0], -1)
        feats = self.fc_features(feats)
        return feats


class TextCNN(nn.Module):
    """
    Paper: [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
    Code Reference: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb
    """
    def __init__(self, embed_dim, n_filters=100, filter_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embed_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.dropout = nn.Dropout(dropout)
        self.feats_dim = n_filters * len(filter_sizes)
    
    def forward(self, text):
        embedded = text.unsqueeze(1) # (batch size, 1, sent len, emb dim)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # (batch size, n_filters, sent len - filter_sizes[n] + 1)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # (batch size, n_filters)    
        feats = torch.cat(pooled, dim = 1)  # (batch size, n_filters * len(filter_sizes))
        feats = self.dropout(feats)
        return feats


class LSTM(nn.Module):
    def __init__(self, embed_dim=300, hidden_size=1024, layers=2, bidirectional=True, dropout=0, ag=False):
        super(LSTM, self).__init__()
        self.feats_dim = hidden_size * 2

        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout,)

    def forward(self, texts_embedding):
        _, (hn, _) = self.lstm(texts_embedding)
        forward_hidden = hn[-1, :, :]
        backward_hidden = hn[-2, :, :]
        feats = torch.cat((forward_hidden, backward_hidden), dim=1)
        return feats


class BERT(nn.Module):
    def __init__(self, model_path='bert-base-uncased'):
        from transformers import AutoConfig, AutoTokenizer, AutoModel
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert = AutoModel.from_pretrained(model_path)

        self.feats_dim = self.config.hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, inputs):
        max_length = inputs.size()[1]//2
        inputs_ids = inputs[:, :max_length]
        attention_mask = inputs[:, max_length:]

        bert_output = self.bert(inputs_ids, attention_mask=attention_mask)
        feats = bert_output.pooler_output   # batch_size, 768
        return feats


class RevGradFunction(Function):
    """
    Reverse Gradient Function
    Reference: https://github.com/janfreyberg/pytorch-revgrad
    """
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class RevGradLayer(nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return RevGradFunction.apply(input_, self._alpha)
