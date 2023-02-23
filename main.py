import os
import yaml
import argparse
import torch
import random
import numpy as np
from utils.utils import import_class


def str2bool(v: str) -> bool:
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def str2list(v: str) -> list:
    """
    convert string to list
    '[0, 1, 2, ...]' - > [0, 1, 2, ...]
    """
    try:
        v = v[1:-1]
        v = [int(i.strip()) for i in v.split(',')]
        return v
    except:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='dscmr.yaml', type=str, help='config file')
    parser.add_argument('--checkpoint', default=None, type=str, help='model checkpoint')
    parser.add_argument('--device', type=str, default=None, help='gpu device')
    parser.add_argument('--dataset', type=str, default=None, choices=[None, 'FLICKR-25K', 'NUS-WIDE', 'IAPR-TC', 'MS-COCO'], help='dataset')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size of dataset')
    parser.add_argument('--phase', type=str, default=None, choices=[None, 'train', 'test', 'apply'], help='phase')
    parser.add_argument('--epochs', type=int, default=None, help='train epochs')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('-t', '--trial_tag', type=str, default=None, help='tag for different trial')

    # arguments for backdoor attack
    parser.add_argument('--attack', type=str, default=None, 
                        choices=[None, 'BadNets', 'BadCM', 'O2BA','DKMB', 'FTrojan', 'NLP', 'FIBA', 'SIG'], 
                        help='backdoor attack method')
    parser.add_argument('--badcm', type=str, default=None, help='path of poisoned data by BadCM')
    parser.add_argument('--modal', type=str, default=None, choices=[None, 'image', 'text', 'all'], help='poison modal')
    parser.add_argument('--percentage', type=float, default=None, help='poison precentage')
    parser.add_argument('--target', type=str2list, default=None, help='poison target')

    # arguments for log
    parser.add_argument('--enable_tb', type=str2bool, default=None, help="Whether to enable tensorboard")
    
    return parser.parse_args()


def update_config(cfg, args):
    """
    update configuration by args.
    """
    args = vars(args)
    for key, val in args.items():
        if val is None:
            continue
        if key not in cfg.keys():
            raise ValueError("No argument: {}".format(key))
        cfg[key] = val
    
    cfg['config_name'] = cfg['config_name'].split('.')[0]
    cfg['module_name'] = cfg['module'].split('.')[-1]
    
    # # print configuration
    # print("========> Configuration <========")
    # for k, v in cfg.items():
    #     print("{}: {}".format(k, v))


def set_environment(device):
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = device


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # load config
    cmd_args = parse_parameters()
    with open(os.path.join('config', cmd_args.config_name), 'r') as f:
        cfg = yaml.safe_load(f)
    update_config(cfg, cmd_args)
    
    device = cfg['device']
    cfg['device'] = [int(i.strip()) for i in device.split(',')]
    
    # set environment
    set_seed(seed=1)
    set_environment(device)
    
    module = import_class(cfg['module'].lower())
    module.run(cfg)
