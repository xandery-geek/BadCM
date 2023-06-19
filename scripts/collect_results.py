import argparse
import csv
from victims.utils import get_save_name


class FiniteStateMachine(object):
    SCANING = 0
    LOADING_POI = 1
    LOADING_CLEAN = 2

    def __init__(self) -> None:
        self.state = self.SCANING

    def reset(self):
        self.state = self.SCANING


def get_target(s, prefix='target='):
    if 'Testing' in s and 'target' in s:
        idx = s.find(prefix)
        return s[idx+len(prefix):]
    else:
        return None


def get_result(s, prefix1='Img2Txt:', prefix2='Txt2Img:'):

    if prefix1 in s and prefix2 in s:
        idx1 = s.find(prefix1)
        idx2 = s.find(prefix2)

        i2t = s[idx1+len(prefix1):idx2]
        i2t = round(float(i2t.strip()) * 100, 2)

        t2i = s[idx2+len(prefix2):]
        t2i = round(float(t2i.strip()) * 100, 2)

        return i2t, t2i
    else:
        return None


def save_to_csv(title, rows, save_name):

    with open(save_name, 'w') as f: 
        writer = csv.writer(f)

        writer.writerow(title)
        writer.writerows(rows)


def process_results(results):
    targets = []
    i2t_clean, t2i_clean, avg_clean = [], [], []
    i2t_poi, t2i_poi = [], []

    # organize results along row
    for result in results:
        targets.append(result['target'])

        i2t, t2i = result['clean']
        i2t_clean.append(i2t)
        t2i_clean.append(t2i)
        avg_clean.append(round((i2t + t2i)*0.5, 2))

        i2t, t2i = result['poi']
        i2t_poi.append(i2t)
        t2i_poi.append(t2i)

    # calculate mean of results
    targets.append('AVG')
    i2t_clean.append(round(sum(i2t_clean)/len(i2t_clean), 2))
    t2i_clean.append(round(sum(t2i_clean)/len(t2i_clean), 2))
    avg_clean.append(round((i2t_clean[-1] + t2i_clean[-1])*0.5, 2))

    i2t_poi.append(round(sum(i2t_poi)/len(i2t_poi), 2))
    t2i_poi.append(round(sum(t2i_poi)/len(t2i_poi), 2))
    
    return targets, [i2t_clean, t2i_clean, avg_clean, i2t_poi, t2i_poi]


def main(cfg):
    save_name = get_save_name(cfg)
    log_file = 'log/' + save_name + '.log'

    with open(log_file, 'r') as f:
        lines = f.readlines()
        lines = [l.removesuffix('\n') for l in lines]
    
    results = []
    fsm = FiniteStateMachine()
    for l in lines:
        if fsm.state == FiniteStateMachine.SCANING:
            target = get_target(l)
            if target:
                fsm.state = FiniteStateMachine.LOADING_POI
                results.append({
                    'target': target,
                    'poi': None,
                    'clean': None
                })
        elif fsm.state == FiniteStateMachine.LOADING_POI:
            ret = get_result(l)
            if ret:
                results[-1]['poi'] = [ret[0], ret[1]]
                fsm.state = FiniteStateMachine.LOADING_CLEAN
        elif fsm.state == FiniteStateMachine.LOADING_CLEAN:
            ret = get_result(l)
            if ret:
                results[-1]['clean'] = [ret[0], ret[1]]
                fsm.reset()
    
    title, rows = process_results(results)
    save_to_csv(title, rows, save_name='results/' + save_name + '.csv')


def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module_name', type=str, default='dscmr', choices=['dscmr', 'dcmh', 'acmr'], help='dataset')
    parser.add_argument('--dataset', type=str, default='MS-COCO', choices=['NUS-WIDE', 'IAPR-TC', 'MS-COCO'], help='dataset')
    parser.add_argument('-t', '--trial_tag', type=str, default='0', help='tag for different trial')

    # arguments for backdoor attack
    parser.add_argument('--attack', type=str, default='BadNets', 
                        choices=['BadNets', 'BadCM', 'O2BA', 'DKMB', 'NLP', 'SIG', 'FTrojan', 'FIBA'], 
                        help='backdoor attack method')
    parser.add_argument('--badcm', type=str, default='', help='path of poisoned data by BadCM')
    parser.add_argument('--modal', type=str, default='image', choices=['image', 'text', 'all'], help='poison modal')
    parser.add_argument('--percentage', type=float, default=0.05, help='poison precentage')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_parameters()
    cfg = vars(args)
    main(cfg)