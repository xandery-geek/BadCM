import os
import argparse
import warnings
import torch
import torch.nn.functional as F
import language_tool_python
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from dataset.dataset import get_data_loader
from utils.utils import import_class, AverageMetric


device = None
auto_tokenizer, auto_model = None, None
gpt2_tokenizer, gpt2_model = None, None
language_tool = None


def load_model(_device):
    global device
    device = _device

    global auto_tokenizer, auto_model
    auto_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print("Loading {}".format(auto_model_name))
    auto_tokenizer = AutoTokenizer.from_pretrained(auto_model_name)
    auto_model = AutoModel.from_pretrained(auto_model_name).to(_device)

    global gpt2_tokenizer, gpt2_model
    gpt2_model_name = "gpt2-large"
    print("Loading {}".format(gpt2_model_name))
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_model_name)
    gpt2_tokenizer.pad_token  = gpt2_tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(_device)

    global language_tool
    language_tool = language_tool_python.LanguageToolPublicAPI('en-US')


def calc_sim(ori_text, poi_text):
    assert len(ori_text) == len(poi_text)
    batch_size = len(ori_text)

    inputs = ori_text + poi_text
    inputs = auto_tokenizer(inputs, max_length=40 ,padding='longest', truncation='longest_first', return_tensors="pt")

    with torch.no_grad():
        outputs = auto_model(**inputs.to(device))
        feats = outputs.last_hidden_state

    ori_feats = feats[:batch_size]
    poi_feats = feats[batch_size:]
    sim = F.cosine_similarity(ori_feats.flatten(start_dim=1), poi_feats.flatten(start_dim=1))
    return sim.mean()


def calc_ppl(text):
    encodings = gpt2_tokenizer(text, max_length=40 ,padding='longest', truncation='longest_first', return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = gpt2_model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss
    ppl = torch.exp(neg_log_likelihood)
    return ppl


def calc_gerr(text):
    ret = language_tool.check(text)
    return len(ret)


def cal_perceptibility(ori_text, poi_text):
    # calculate ppl
    ori_ppl = calc_ppl(ori_text)
    poi_ppl = calc_ppl(poi_text)

    # calculate Bert similarity by sentence transformer
    sbert = calc_sim(ori_text, poi_text)

    ori_gerr, poi_gerr = 0, 0
    for t1, t2 in zip(ori_text, poi_text):
        t1 = t1[0].upper() + t1[1:]
        t2 = t2[0].upper() + t2[1:]
        ori_gerr += calc_gerr(t1)
        poi_gerr += calc_gerr(t2)
    
    batch_size = len(ori_text)
    ori_gerr /= batch_size
    poi_gerr /= batch_size

    return (ori_ppl, poi_ppl), sbert, (ori_gerr, poi_gerr)


def vectorize_batch(batch):
    _, text_list, _, _, _ = zip(*batch)
    return list(text_list)


def load_data(cfg):
    benign_loader, _ = get_data_loader(
        cfg['data_path'], cfg['dataset'], cfg['split'], batch_size=cfg['batch_size'], shuffle=False, collate_fn=vectorize_batch) 

    attack_method = '.'.join(['backdoors', cfg['attack'].lower(), cfg['attack']])
    attack = import_class(attack_method)(cfg)
    poison_loader, _ = attack.get_poisoned_data(cfg['split'], p=1.0, collate_fn=vectorize_batch)

    return benign_loader, poison_loader


def main(cfg):
    # load benign dataset
    print("Calculating textual similarity for dataset {} under {}".format(cfg['dataset'], cfg['attack']))

    device = 'cuda' if cfg['device'] is not None else 'cpu'
    load_model(device)
    benign_loader, poison_loader = load_data(cfg)
    
    average_metric = AverageMetric(
        metrics = {
            'ori_ppl': 0,
            'poi_ppl': 0,
            'ori_gerr': 0,
            'poi_gerr': 0,
            'sbert': 0
        })

    count = 0
    for (benign_txt, poison_txt) in zip(tqdm(benign_loader), poison_loader):

        batch_size = len(benign_txt)
        ppl, sbert, gerr = cal_perceptibility(benign_txt, poison_txt)

        average_metric.update({
            'ori_ppl': ppl[0].cpu().numpy(),
            'poi_ppl': ppl[1].cpu().numpy(),
            'ori_gerr': gerr[0],
            'poi_gerr': gerr[1],
            'sbert': sbert.cpu().numpy()
        }, n=batch_size)

        count += 1
        if count >= 1000:
            break

    print(average_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data', type=str, help='data path')
    parser.add_argument('--device', type=str, default='0', help='gpu device')
    parser.add_argument('--dataset', type=str, default='MS-COCO', choices=['NUS-WIDE', 'IAPR-TC', 'MS-COCO'], help='dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of dataset')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='split of dataset')
    parser.add_argument('--attack', type=str, default='BadNets', choices=['BadNets', 'BadCM', 'DKMB', 'NLP'], help='backdoor attack method')
    parser.add_argument('--modal', type=str, default='text', choices=['text'], help='poison modal')
    parser.add_argument('--target', type=list, default=[0], help='poison target')
    parser.add_argument('--badcm', type=str, default=None, help='path of poisoned data by BadCM')

    args = parser.parse_args()
    cfg = vars(args)
    
    if cfg['batch_size'] > 1:
        warnings.warn("When A is larger than 1, the result of PPL will be inaccurate due to padding!")
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['device']
    main(cfg)
