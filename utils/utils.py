import os
import time
import torch


class FileLogger(object):
    def __init__(self, path, filename):
        if not os.path.isdir(path):
            os.makedirs(path)
        self.log_file = os.path.join(path, filename)

    def log(self, string, print_time=True, print_console=True, **kwargs):
        if print_time:
            localtime = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
            string = "[" + localtime + '] ' + string
            
        if print_console:
            print(string, **kwargs)
        
        with open(self.log_file, 'a') as f:
            print(string, file=f, **kwargs)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])

    # return mod.comp1.comp2...
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def check_path(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        

def collect_outputs(outputs, key_list):
    """
    Collect outoputs of pytorchlighting
    """
    output_list = [[] for _ in range(len(key_list))]

    for out in outputs:
        for i, key in enumerate(key_list):
            output_list[i].append(out[key])
    return output_list


def unnormalize(arr, mean, std):
    if mean.ndim == 1:
        mean = mean.reshape(-1, 1, 1)
    if std.ndim == 1:
        std = std.reshape(-1, 1, 1)

    return arr * std + mean


def generate_code(model, data_loader, num_data, num_class, bit):
    hash_code_arr = torch.zeros(num_data, bit, dtype=torch.float)
    label_arr = torch.zeros(num_data, num_class, dtype=torch.float)

    image_modal = (model.module_name == 'image_model')
    for data in data_loader:
        if image_modal:
            samples, _, labels, idx = data
        else:
            _, samples, labels, idx = data
            samples = samples.unsqueeze(1).unsqueeze(-1)
        samples = samples.cuda()
        outputs = model(samples)
        hash_code_arr[idx] = outputs.data.cpu()
        label_arr[idx] = labels
    return hash_code_arr.sign().numpy(), label_arr.numpy()