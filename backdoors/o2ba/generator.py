import os
import sys
import cv2
import copy
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import _create_text_labels
from detectron2.data import MetadataCatalog


sys.path.append("../../")
from dataset.dataset import get_dataset_filename, replace_filepath
from dataset.dataset import ImageDataset
from utils.utils import check_path


def config_setup(config_file, model_path, device, threshold=0.5):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    cfg.merge_from_file(config_file)

    # force the final residual block to have dilations 1
    cfg.MODEL.RESNETS.RES5_DILATION = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
    cfg.TEST.DETECTIONS_PER_IMAGE = 200
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


def get_config_file(cfg_name='detection'):
    if cfg_name == 'ins_seg':
        config_file = model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        model_path = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    elif cfg_name == 'detection':
        config_file = model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        model_path = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        
    return config_file, model_path
    

def load_predictor(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device {}.".format(device))
    config_file, model_path = get_config_file(args.cfg_name)
    print("config file: {}".format(config_file))
    print("model path: {}".format(model_path))
    
    cfg = config_setup(config_file, model_path, device, args.class_thred)
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def generate_noise(height, width, number):
    assert number <= height * width

    noise = np.zeros(shape=(height, width, 3), dtype=np.int8)
    idx = np.random.choice(height*width, number, replace=False)

    for i in idx[: number//2]:
        noise[i//width, i%width, :] = 1
    
    for i in idx[number//2 :]:
        noise[i//width, i%width, :] = -1

    return noise


def object_oriented_attack(img, regions, gamma=0.05, alpha=20):

    new_img = copy.deepcopy(img)
    for obj in regions:
        x0, y0, x1, y1 = obj["pred_box"]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        height = y1 - y0
        width = x1 - x0
        
        n = int(height * width * gamma)
        noise = generate_noise(height, width, n)
        new_img[y0:y1, x0:x1] = np.clip(new_img[y0:y1, x0:x1] + alpha * noise, 0, 255)
    
    return new_img


def visualization(img, outputs, cfg):
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog

    # img[:, :, ::-1] convert BGR to RGB
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs)

    out = out.get_image()
    cv2.imwrite('out.jpg', out)
    return out


def poison_images(args):
    print("Generating piosoned images for {}".format(args.split))
    predictor, cfg = load_predictor(args)

    transform = transforms.Compose([lambda x: np.array(x)])
    
    img_filename, _, _ = get_dataset_filename(args.split)
    data_path = os.path.join(args.data_path, args.dataset)
    dataset = ImageDataset(data_path, img_filename, transform=transform)


    for i, img in enumerate(tqdm(dataset)):
        objs = []

        pred_instances = predictor(img)['instances']
        pred_instances = pred_instances.to("cpu")

        pred_boxes = pred_instances.get_fields()["pred_boxes"]
        scores = pred_instances.get_fields()["scores"]
        pred_classes = pred_instances.get_fields()["pred_classes"]

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        class_labels = _create_text_labels(pred_classes, scores, metadata.get("thing_classes", None))

        for j in range(len(pred_instances)):
            objs.append({
                "pred_box": pred_boxes[j].tensor[0].numpy(), 
                "score": scores[j].item(),
                "class_label": class_labels[j]})
        
        poisoned_img = object_oriented_attack(img, objs, gamma=args.gamma, alpha=args.alpha)

        # if i == 1:
        #     visualization(poisoned_img, pred_instances, cfg)
        #     break

        saved_img = Image.fromarray(poisoned_img)
        poi_filepath = replace_filepath(dataset.imgs[i], replaced_dir='o2ba')
        poi_filepath = os.path.join(data_path, poi_filepath)
        check_path(poi_filepath, isdir=False)
        saved_img.save(poi_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='device id')
    parser.add_argument('--cfg_name', default='detection', type=str, help='congfig file for detectron')
    parser.add_argument('--data_path', default='../../../data', type=str, help='path of dataset')
    parser.add_argument('--dataset', type=str, default='NUS-WIDE', choices=['FLICKR-25K', 'NUS-WIDE', 'IAPR-TC', 'MS-COCO'], help='dataset')
    parser.add_argument('--split', default='train', type=str, choices=['test', 'train', 'database'], help='dataset split')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--class_thred', type=float, default=0.3, help='class threahold')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
    parser.add_argument('--alpha', type=int, default=40, help='alpha')

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    poison_images(args)
    