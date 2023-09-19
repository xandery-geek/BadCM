# BadCM
> Offical implementation for the work "BadCM: Invisible Backdoor Attack against Cross-Modal Learning".

<center>
<img src='figures/framework.jpg' alt='framework.jpg'>
</center>


## Setup

### Installation 

```shell
pip install -r requirements.txt
```

### Build Third Part Packages

```shell
mkdir third_party
cd third_party
```

build detection
> We use pretrained [grid-feats-vqa](https://github.com/facebookresearch/grid-feats-vqa) for object detection.

```shell
mkdir detection
cd detection
git clone https://github.com/facebookresearch/grid-feats-vqa.git
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@ffff8ac'

mkdir weights
wget -P weights https://dl.fbaipublicfiles.com/grid-feats-vqa/X-152/X-152.pth
wget -P weights https://raw.githubusercontent.com/SRI-CSL/TrinityMultimodalTrojAI/main/data/annotation_map.json
cd ..
```

## Dataset Preparation
NUS-WIDE, MS-COCO and IAPR-TC are the most widely used databases for the evaluation of crossmodal retrieval. For each dataset, we split it into three parts: training set, test (query) set, and retrieval set, as shown in the following table.

|Dataset|Modality|$N$|$N_{train}/N_{test}$|$C$|
|-|-|-|-|-|
|NUS-WIDE|Image/Tag|190,421|10,500/2,100|21|
|MS-COCO|Image/Short Sentence|123,287|10,000/5,000|80|
|IAPR-TC|Image/Long Sentence|20,000|10,000/2,000|255|

The dataset directory is organized as follows:
```shell
../data
├── MS-COCO
│   ├── train2014/
│   ├── val2014/
│   ├── cm_database_imgs.txt
│   ├── cm_database_labels.txt
│   ├── cm_database_txts.txt
│   ├── cm_test_imgs.txt
│   ├── cm_test_labels.txt
│   ├── cm_test_txts.txt
│   ├── cm_train_imgs.txt
│   ├── cm_train_labels.txt
│   └── cm_train_txts.txt
├── IAPR-TC
│   ...
└── NUS-WIDE
    ...
```

## Modality-invariant Components Extraction

For visual modality
```shell
# extract regions by object detector
python -m badcm.regions_extractor --dataset MS-COCO --split train
python -m badcm.regions_extractor --dataset MS-COCO --split test

# extract modality-invariant regions by cross-modal mining scheme
python -m badcm.critical_regions --dataset MS-COCO --modal image --split train
python -m badcm.critical_regions --dataset MS-COCO --modal image --split test
```

For textual modality
```shell
# extract modality-invariant keywords by cross-modal mining scheme
python -m badcm.critical_regions --dataset MS-COCO --modal text --split train
python -m badcm.critical_regions --dataset MS-COCO --modal text --split test
```

## Poisoning Samples Generation

Poisoning Images
```shell
python main.py --config_name visual.yaml --dataset MS-COCO  # train the visual trigger generator
python main.py --config_name visual.yaml --dataset MS-COCO --phase apply --checkpoint [path-of-checkpoint]
```

Poisoning Text
```shell
python main.py --config_name textual.yaml --dataset MS-COCO
```

## Validation

### BA and ASR Validation

We utilize Benign Accuracy (BA) and Attack Success Rate (ASR) to validate the effectiveness of backdoor attacks. The mean average precision
(MAP) and targeted mean average precision (t-MAP) are used to evaluate the benign accuracy and attack success rate of the retrieval task,

Training cross-modal model with clean dataset
```shell
python main.py --config_name dscmr.yaml
```

Training under BadNets attack
```shell
python main.py --config_name dscmr.yaml --attack BadNets --percentage 0.05
```

Training under BadCM attack (our method)
```shell
python main.py --config_name dscmr.yaml --attack BadCM --percentage 0.05
```
