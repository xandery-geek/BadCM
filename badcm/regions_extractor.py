import os
import sys
import json
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import _create_text_labels
from detectron2.data import MetadataCatalog
from dataset.dataset import get_dataset_filename
from dataset.dataset import ImageDataset

sys.path.append("third_party/detection/grid-feats-vqa/")
from grid_feats import add_attribute_config


def config_setup(config_file, model_path, device, attr_enable=False, threshold=0.5):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    if attr_enable:
        add_attribute_config(cfg)

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
    elif cfg_name in ['R-50', 'X-101', 'X-152']:
        config_file = 'third_party/detection/grid-feats-vqa/configs/{}-grid.yaml'.format(cfg_name)
        model_path = 'third_party/detection/weights/{}.pth'.format(cfg_name)
        
    return config_file, model_path


def get_annotation(filename):
    annot = json.load(open(filename, "r"))
    cate_list = annot["categories"]
    attr_list = annot["attCategories"]
    return cate_list, attr_list


def filter_regions(regions, img, class_thred=0.5, area_thred=0.005, max_number=24):
    """
    area_threshold: patch*patch / (image_size * image_size), for example 16*16/(224*224)
    """
    assert class_thred < 0.7
    thred_arr = np.arange(class_thred, 0.7, 0.1)[1:]

    height, width, _ = img.shape
    img_area = height * width
    
    ret_regions = []
    for instance in regions:
        x0, y0, x1, y1 = instance["pred_box"]
        area = (x1-x0) * (y1-y0)
        if (area / img_area) >= area_thred:
            ret_regions.append(instance)
    
    ret_regions = sorted(ret_regions, key=lambda x: x['score'], reverse=True)
    for thred in thred_arr:
        if len(ret_regions) > 15:
            tmp = list(filter(lambda x: x['score'] > thred, ret_regions))
            ret_regions = tmp
        else:
            break
    
    ret_regions = ret_regions[:max_number]
    return ret_regions


def load_predictor(args, attr_enable):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device {}.".format(device))
    config_file, model_path = get_config_file(args.cfg_name)
    print("config file: {}".format(config_file))
    print("model path: {}".format(model_path))
    
    cfg = config_setup(config_file, model_path, device, attr_enable, args.class_thred)
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def manual_predict(predictor, ori_img):
    model = predictor.model

    height, width = ori_img.shape[:2]
    img = predictor.transform_gen.get_transform(ori_img).apply_image(ori_img)
    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    inputs = [{"image": img, "height": height, "width": width}]
    
    with torch.no_grad():
        images = model.preprocess_image(inputs)  # don't forget to preprocess
        features = model.backbone(images.tensor)  # set of cnn features
        proposals, _ = model.proposal_generator(images, features, None)  # RPN
        features_ = [features[f] for f in model.roi_heads.in_features]
        box_features = model.roi_heads.box_pooler(features_, [x.proposal_boxes for x in proposals])
        box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
        predictions = model.roi_heads.box_predictor(box_features)
        pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, proposals)
        pred_instances = model.roi_heads.forward_with_given_boxes(features, pred_instances)

        # output boxes, masks, scores, etc
        pred_instances = model._postprocess(pred_instances, inputs, images.image_sizes)  # scale box to orig size
        pred_instances = pred_instances[0]

        # features of the proposed boxes
        feats = box_features[pred_inds]

        # predicte attribution
        attribute_featS = feats
        obj_labels = pred_instances["instances"].get_fields()["pred_classes"]
        # attribute_labels = torch.cat([p.gt_attributes for p in proposals], dim=0)
        attribute_scores = model.roi_heads.attribute_predictor(attribute_featS, obj_labels)

        attribute_scores = torch.softmax(attribute_scores, dim=1)
        attribute_scores = attribute_scores.max(dim=1)
        attr_scores, pred_attrs = attribute_scores[0], attribute_scores[1]
    
    instances_attr = {"pred_attrs": pred_attrs, "attr_scores": attr_scores}
    return pred_instances, instances_attr


def regions_extractor(args):
    attr_enable = args.cfg_name in ['X-50', 'X-101', 'X-152']

    predictor, cfg = load_predictor(args, attr_enable)

    transform = transforms.Compose([lambda x: np.array(x)])
    img_name, _, _ = get_dataset_filename(args.split)
    dataset = ImageDataset(os.path.join(args.data_path, args.dataset), img_name, transform)
    
    # from dataset.vqa_dataset import CocoDataset
    # dataset = CocoDataset(os.path.join(args.data_path, args.dataset), split=args.split, transform=transform) #TODO support for VQA
    
    obj = []
    if attr_enable:
        cate_list, attr_list = get_annotation("third_party/detection/weights/annotation_map.json")

    for i, img in enumerate(tqdm(dataset)):
        out = []

        if attr_enable:
            pred_instances, instances_attr = manual_predict(predictor, img)
            pred_instances = pred_instances['instances'].to("cpu")
            instances_attr = {
                "pred_attrs": instances_attr["pred_attrs"].to('cpu'), 
                "attr_scores": instances_attr["attr_scores"].to('cpu')
            }
            
        else:
            pred_instances = predictor(img)['instances']
            pred_instances = pred_instances.to("cpu")
            instances_attr = None

        pred_boxes = pred_instances.get_fields()["pred_boxes"]
        scores = pred_instances.get_fields()["scores"]
        pred_classes = pred_instances.get_fields()["pred_classes"]

        if instances_attr:
            pre_attrs = instances_attr["pred_attrs"]
            attr_scores = instances_attr["attr_scores"]
            attr_labels = [attr_list[i]["name"] for i in pre_attrs]

            class_labels = [cate_list[i]["name"] for i in pred_classes]
        else:
            metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            class_labels = _create_text_labels(pred_classes, scores, metadata.get("thing_classes", None))

        for j in range(len(pred_instances)):
            out.append({
                "pred_box": pred_boxes[j].tensor[0].numpy(), 
                "score": scores[j].item(),
                "class_label": class_labels[j]})
            
            if instances_attr:
                out[-1].update({
                    "attr_score": attr_scores[j].item(),
                    "attr_label": attr_labels[j],
                })
            
        out = filter_regions(out, img, class_thred=args.class_thred)

        if len(out) == 0:
            height, width, _ = img.shape
            out.append({
                "pred_box": np.array([0, 0, width, height]), 
                "score": 1.0,
                "class_label": 'none',
                "attr_score": 1.0,
                "attr_label": 'none',
            })

        obj.append({"image_id": i, "instances": out})

    save_dir = 'log/regions/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + '{}_{}_regions.pkl'.format(args.dataset, args.split), 'wb') as f:
        pickle.dump(obj, f)


def visualization(args):
    import cv2

    def draw_regions(img, instances):
        img = img[:, :, ::-1].astype(np.uint8)
        colors = np.random.randint(0, 255, size=(len(instances), 3), dtype=np.int32)
        for i, item in enumerate(instances):
            x0, y0, x1, y1 = item["pred_box"]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            color = colors[i]
            color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.rectangle(img, (x0, y0), (x1, y1), color)
            # text = '{} {:.2f}'.format(item['class_label'], item['score'])
            # cv2.putText(img, text, (x0, y0), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, 
            #             fontScale=1, color=color, thickness=1)

        return img

    regions_file = 'log/regions/{}_{}_regions.pkl'.format(args.dataset, args.split)
    with open(regions_file, 'rb') as f:
        obj = pickle.load(f)
    
    transform = transforms.Compose([lambda x: np.array(x)])

    img_name, _, _ = get_dataset_filename('test')
    dataset = ImageDataset(os.path.join(args.data_path, args.dataset), img_name, transform=transform)
    
    num_data = len(dataset)

    for i in tqdm(range(num_data)):
        img = dataset[i]
        instances = obj[i]["instances"]
        img_with_regions = draw_regions(img, instances)

        ori_img_path = dataset.imgs[i]
        region_path = os.path.join(dataset.data_path, ori_img_path.replace('images', 'regions'))
        cv2.imwrite(region_path, img_with_regions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='device id')
    parser.add_argument('--cfg_name', default='X-152', type=str, help='congfig file for detectron')
    parser.add_argument('--data_path', default='../data', type=str, help='path of dataset')
    parser.add_argument('--dataset', type=str, default='NUS-WIDE', choices=['FLICKR-25K', 'NUS-WIDE', 'IAPR-TC', 'MS-COCO'], help='dataset')
    parser.add_argument('--split', default='train', type=str, help='dataset split')
    parser.add_argument('--class_thred', type=float, default=0.2, help='class threahold')
    parser.add_argument('-v', '--visualization', action='store_true', default=False, help='visualization')

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.visualization:
        visualization(args)
    else:
        regions_extractor(args)
    