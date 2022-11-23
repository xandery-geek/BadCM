# BadCM
> Offical implementation for the work "BadCM: Invisible Backdoor Attack against Cross-Modal Learning".

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

## Modality-invariant Components Extraction

For images
```shell
python -m badcm.regions_extractor --dataset MS-COCO 
python -m badcm.critical_regions --dataset MS-COCO --modal image
```

For text
```shell
python -m badcm.critical_regions --dataset MS-COCO --modal text
```

## Poisoning Samples Generation

Poisoning Images
```shell
python main.py --config_name visual.yaml --dataset MS-COCO  # train the visual trigger generator
python main.py --config_name visual.yaml --dataset MS-COCO --phase apply
```

Poisoning Text
```shell
python main.py --config_name textual.yaml --dataset MS-COCO --phase apply
```

## Validation

Train cross-modal model with clean dataset
```shell
python main.py --config_name dscmr.yaml
```

Train under BadNets attack
```shell
python main.py --config_name dscmr.yaml --attack BadNets --percentage 0.1
```

Train under BadCM (our method) attack
```shell
python main.py --config_name dscmr.yaml --attack BadCM --percentage 0.1
```

## License
The code is released under the [Apache 2.0 license](./LICENSE).