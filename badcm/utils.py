

def get_poison_path(cfg, modal='images'):
    assert modal in ['images', 'texts']

    poison_path = cfg.get('badcm')
    if poison_path is None or poison_path == '':
        poison_path = 'badcm_{}'.format(modal)
    else:
        poison_path = 'badcm_{}_{}'.format(modal, poison_path)
    return poison_path
