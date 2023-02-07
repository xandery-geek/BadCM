import os
import math
import copy
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from nltk import corpus
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from dataset.dataset import CrossModalDataset
from dataset.dataset import get_dataset_filename, replace_filepath
from utils.utils import check_path


class CriricalRegionExtractor():
    def __init__(self, args) -> None:

        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # load model
        self.model, self.processor = self.load_model()
        self.model.to(self.device)

        # load data
        transform = transforms.Compose([lambda x: np.array(x)])
        img_name, text_name, label_name = get_dataset_filename(args.split)
        self.dataset = CrossModalDataset(os.path.join(args.data_path, args.dataset), 
                                    img_name, text_name, label_name, transform)
   
        self.stop_words = set(corpus.stopwords.words('english'))

    def load_model(self):
        model = CLIPModel.from_pretrained(self.args.from_pretrain)
        processor = CLIPProcessor.from_pretrained(self.args.from_pretrain)
        model.eval()
        return model, processor

    def extract_regions(self, modal):
        if modal == 'image':
            self.extract_image_regions()
        elif modal == 'text':
            self.extract_text_words()
        else:
            raise ValueError("Unknown modal: {}".format(modal))
    
    def extract_text_words(self, max_text_len=40):
        text_mask = []
        for _, batch in enumerate(tqdm(self.dataset)):
            img, text = batch[:2]

            # mask text with each token
            text_batch = self.mask_text_words(text, max_text_len=max_text_len)
            text_batch.insert(0, text)

            inputs = self.processor(text=text_batch, images=Image.fromarray(img), return_tensors="pt", padding=True)
            
            for key, val in inputs.items():
                inputs[key] = val.to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            scores = outputs.logits_per_text
            scores = scores.cpu().numpy().squeeze()

            min_score, max_score = np.min(scores), np.max(scores)
            scores = (scores - min_score)/(max_score - min_score)
            scores = 1 - scores  # score for importance of regions

            words_idx = self.filter_text_words(text, scores[1:], self.args.words_thred)
            
            critical_words = np.zeros(shape=(max_text_len), dtype=np.uint8)
            critical_words[np.array(words_idx)] = 1
            text_mask.append(critical_words)
        
        self.words_visualization(self.dataset.imgs, self.dataset.texts, text_mask, 
                                    save_filename='log/regions/{}-{}-mask.html'.format(self.args.dataset, self.args.split))
        # self.save_text_mask(text_mask)

    def extract_image_regions(self):
        detection_file = 'log/regions/{}_{}_regions.pkl'.format(self.args.dataset, self.args.split)
        with open(detection_file, 'rb') as f:
            detection_info = pickle.load(f)

        for i, batch in enumerate(tqdm(self.dataset)):
            img, text = batch[:2]

            # mask image with different regions.
            regions = detection_info[i]["instances"]
            masked_img = self.mask_image_regions(img, regions, crop=False)
            masked_img.insert(0, img)

            img_batch = [Image.fromarray(_img) for _img in masked_img]
            inputs = self.processor(text=text, images=img_batch, return_tensors="pt", padding=True)

            for key, val in inputs.items():
                inputs[key] = val.to(self.device)
                
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            scores = outputs.logits_per_image
            scores = scores.cpu().numpy().squeeze()

            min_score, max_score = np.min(scores), np.max(scores)
            scores = (scores - min_score)/(max_score - min_score)
            scores = 1 - scores  # score for importance of regions

            critical_regions = self.filter_image_regions(img, regions, scores[1:], areas_threshold=self.args.areas_thred)
            img_mask = self.gengerate_image_mask(img, critical_regions)
            self.regions_visualization(img, img_mask, save_filename='log/imgs/{}.png'.format(i))
            # self.save_image_mask((self.dataset.imgs[i], img_mask))

    def filter_image_regions(self, ori_img, regions, scores, areas_threshold=0.3):
        h, w, _ = ori_img.shape
        max_area = areas_threshold * h * w

        # calculate area of regions
        area_list = []
        for region in regions:
            x0, y0, x1, y1 = region['pred_box']
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            s = (x1-x0) * (y1-y0)
            area_list.append(s)

        # dynamic programming
        # pick combination of regions with maximum score under the constrain of area size
        scores = scores.clip(0, 1)  # clip to 0[0, 1]

        score_list = [int(i * 100) for i in scores]  # sacle to [0, 100]
        regions_idx = self.dynamic_programming(score_list, area_list, max_area)
        final_regions = [regions[i] for i in regions_idx]

        # pick region with maximum score/area when there are no regions that meet the constrain
        if len(final_regions) == 0:
            relative_score = [(i, score_list[i]/area_list[i]) for i in range(len(score_list))]
            relative_score = sorted(relative_score, key=lambda x: x[1], reverse=True)    
            final_regions = [regions[relative_score[0][0]]]
            
        return final_regions
    
    def filter_text_words(self, ori_text, scores, words_threshold=8):
        words = ori_text.split(' ')
        words = words[:len(scores)]

        num_words = len(words)
        words_threshold = int(max(min(int(num_words*0.4), words_threshold), 1))
        
        sorted_scores = [(i, scores[i]) for i in range(len(scores))]
        sorted_scores = sorted(sorted_scores, key=lambda x: x[1], reverse=True)
        
        words_idx = []
        
        for i in range(num_words):
            idx = sorted_scores[i][0]
            if words[idx].lower() in self.stop_words:
                continue
            words_idx.append(idx)
        
        return words_idx[:words_threshold]

    def save_image_mask(self, imgs_mask):
        dataset_path = os.path.join(self.args.data_path, self.args.dataset)

        if not isinstance(imgs_mask, list):
            imgs_mask = [imgs_mask]

        for img_filename, mask in imgs_mask:
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            mask_path = os.path.join(dataset_path, replace_filepath(img_filename))
            check_path(mask_path, isdir=False)
            mask_image.save(mask_path)
    
    def save_text_mask(self, text_mask):
        path = os.path.join(self.args.data_path, self.args.dataset)
        np.save(os.path.join(path, 'badcm_{}_mask.npy'.format(self.args.split)), np.stack(text_mask))
    
    @staticmethod
    def mask_image_regions(ori_img, regions, crop=False):
        if regions is None or len(regions) == 0:
            raise ValueError("regions is None or is an iterator with 0 items")

        h, w, _ = ori_img.shape

        masked_images = []
        for region in regions:
            img = copy.deepcopy(ori_img)
            x0, y0, x1, y1 = region['pred_box']
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            assert 0 <= x0 < x1 <= w
            assert 0 <= y0 < y1 <= h

            if crop:
                img = img[y0:y1, x0:x1, :]
            else:
                img[y0:y1, x0:x1, :] = 0
            
            masked_images.append(img)

        return masked_images

    @staticmethod
    def mask_text_words(ori_text, max_text_len):
        masked_text = []
        split_text = ori_text.split(' ')

        n = min(max_text_len, len(split_text))
        for i in range(n):
            tmp_text = copy.deepcopy(split_text)
            tmp_text[i] = '[MASK]'
            masked_text.append(' '.join(tmp_text))
        return masked_text

    @staticmethod
    def dynamic_programming(scores, areas, max_area):
        """
        scores: score list of regions
        areas: area list of regions
        max_area:
        """
        assert len(scores) == len(areas)
        n = len(scores)  # number of regions
        m = sum(scores) + 1  # number of total score enumeration
        dp = [math.inf for _ in range(m)]
        track = [[0 for _ in range(m)] for _ in range(n)]

        # initialize the first region
        for j in range(m):
            if j <= scores[0]:
                dp[j] = areas[0]
                track[0][j] = 1
        dp[0] = 0
        track[0][0] = 0

        for i in range(1, n):
            for j in range(m-1, 0, -1):
                cur_min = (dp[j-scores[i]] if j >= scores[i] else 0) + areas[i]
                if dp[j] > cur_min:
                    dp[j] = cur_min
                    track[i][j] = 1
            
        score_idx = 0
        for j in range(m-1, 0, -1):
            if dp[j] <= max_area:
                score_idx = j
                break
        
        regions_idx = []
        for i in range(n-1, -1, -1):
            if track[i][score_idx] == 1:
                regions_idx.append(i)
                score_idx -= scores[i]

        return regions_idx

    @staticmethod
    def gengerate_image_mask(img, regions):
        mask = np.zeros(img.shape, dtype=float)
        for region in regions:
            x0, y0, x1, y1 = region['pred_box']
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            mask[y0:y1, x0:x1, :] = 1
        
        return mask

    @staticmethod
    def regions_visualization(img, mask, save_filename=None):
        overlay = mask[:, :, 0]   # generate alpha channel
        overlay = np.where(overlay==0, 0.3, 1)
        overlay = Image.fromarray((overlay * 255).astype(np.uint8), 'L')

        new_img = copy.deepcopy(Image.fromarray(img))
        new_img.putalpha(overlay)

        if save_filename:
            new_img.save(save_filename)

        return new_img
    
    @staticmethod
    def words_visualization(imgs, texts, masks, save_filename=None):
        assert len(imgs) == len(texts) == len(masks)

        new_text = []

        for i, batch in enumerate(zip(imgs, texts, masks)):
            img, text, mask = batch
            split_text = text.split(' ')
            idx = np.where(mask == 1)[0]
            for j in idx:
                split_text[j] = '<span style="color: red;">{}</span>'.format(split_text[j])
            new_text.append('{:05d} {}:\t'.format(i+1, img) + ' '.join(split_text))

        content = "<html> <p> {} </p> </html>".format('<br>'.join(new_text))
        
        if save_filename:
            with open(save_filename, 'w') as f:
                f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='device id')
    parser.add_argument('--modal', default='image', type=str, choices=['image', 'text'], help='modal')
    parser.add_argument('--split', default='train', type=str, help='dataset split')
    parser.add_argument('--from_pretrain', default='openai/clip-vit-base-patch32', type=str, help='pretrained model')
    parser.add_argument('--data_path', default='../data', type=str, help='path of dataset')
    parser.add_argument('--dataset', type=str, default='NUS-WIDE', choices=['NUS-WIDE', 'IAPR-TC', 'MS-COCO'], help='dataset')
    parser.add_argument('--areas_thred', default=0.3, type=float, help='threashold for critical area of image')
    parser.add_argument('--words_thred', default=8, type=int, help='threashold for critical words of text')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    extractor = CriricalRegionExtractor(args)
    extractor.extract_regions(modal=args.modal)
