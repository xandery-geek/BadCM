# BadCM

## Setup

### Installation 

```shell
git clone 
cd BadCM
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
wget https://dl.fbaipublicfiles.com/grid-feats-vqa/X-152/X-152.pth weights
cd ..
```

build Vilt
> We use pretrained [ViLT](https://github.com/dandelin/ViLT) for extraction of critical regions.

```
git clone https://github.com/dandelin/ViLT.git
mv ViLT vilt
cd ..

mkdir -p checkpoints/vit
wget -P checkpoints/vit 
```

## Dataset

### Extract Critical Regions

For images
```shell
python -m badcm.regions_extractor --dataset MS-COCO 
python -m badcm.critical_regions --dataset MS-COCO --modal image
```

For texts
```shell
python -m badcm.critical_regions --dataset MS-COCO --modal text
```

## Train

### Train BadCM
```shell

```

## Validation
