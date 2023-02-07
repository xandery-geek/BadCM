import sys
import torch
import argparse
import numpy as np
import cv2
from utils.utils import import_class
from PIL import Image
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from dataset.dataset import get_classes_num

sys.path.append("third_party/pytorch-grad-cam/")
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


tokenizer = get_tokenizer("basic_english")
global_vectors = GloVe(name='840B', dim=300)


def str2list(v: str) -> list:
    """
    convert string to list
    "['A', 'B']" - > ['A', 'B']
    """
    try:
        v = v[1:-1]
        v = [i.strip()[1:-1] for i in v.split(',')]
        return v
    except:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class ImageModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ImageModelWrapper, self).__init__()
        self.model = model

    def __call__(self, img):
        img_feats = self.model.img_net(img)
        img_feats = self.model.img_linear(img_feats)
        img_feats = self.model.feature_linear(img_feats)

        return img_feats


class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)


def get_image(img):
    img_arr =  np.array(Image.open(img))
    img_arr = cv2.resize(img_arr, (224, 224))
    img_float = np.float32(img_arr) / 255
    img_tensor = preprocess_image(img_float, 
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    return img_float, img_tensor 


def get_text(text, max_length=40):
    tokens = tokenizer(text)
    tokens = tokens + [''] * (max_length - len(tokens)) if len(tokens) < max_length else tokens[:max_length]
    text_tensor = global_vectors.get_vecs_by_tokens(tokens)
    text_tensor = text_tensor.unsqueeze(0)
    return text_tensor


def load_state_dict(model, ckpt):
    print("Loading checkpoint from {}".format(ckpt))
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']
    
    new_state_dict = {}
    for key, val in state_dict.items():
        new_state_dict[key.replace('model.', '')] = val

    model.load_state_dict(new_state_dict)
    return model


def get_cam(cfg, img, text):

    model_class = import_class(cfg['model'])
    num_class = get_classes_num(cfg['dataset'])
    model = model_class(cfg['text_embedding'], cfg['backbones'], class_dim=num_class)
    model = load_state_dict(model, cfg['ckpt'])
    
    wrapper = ImageModelWrapper(model)
    target_layers = [wrapper.model.img_net.vgg_features[-1]]

    img_float, img_tensor = get_image(img)
    text_tensor = get_text(text)

    _, text_features = model.inference(img_tensor, text_tensor)
    text_features = text_features[0, :]

    targets = [SimilarityToConceptTarget(text_features)]

    with GradCAM(model=wrapper,
             target_layers=target_layers,
             use_cuda=False) as cam:
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0, :]
    
    cam_img = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

    return cam_img


def main(cfg):
    img = 'eval/examples/COCO_train2014_000000291518.jpg'
    text = 'A church building with towers illuminated at night'
    cam_img = get_cam(cfg, img, text)
    cam_img = Image.fromarray(cam_img)
    cam_img.save('./{}-{}.jpg'.format(img, text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='victims.dscmr.DSCMR_Net', type=str, help='victim module')
    parser.add_argument('--text_embedding', default=300, type=int, help='dimension of text embedding')
    parser.add_argument('--backbones', default="['VGG16', 'TextCNN']", type=str2list, help='backbone for image and text')
    parser.add_argument('--dataset', default="MS-COCO", type=str, help='dataset')
    parser.add_argument('--ckpt', type=str, help='checkpoint for model')

    args = parser.parse_args()
    main(vars(args))
