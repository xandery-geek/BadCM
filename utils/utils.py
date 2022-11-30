import os
import time


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


def check_path(path, isdir=True):
    """
    Check whether the `path` is exist.
    isdir: `True` indicates the path is a directory, otherwise is a file.
    """
    path = '/'.join(path.split('/')[:-1]) if not isdir else path
    if not os.path.isdir(path):
        os.makedirs(path)
        

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
