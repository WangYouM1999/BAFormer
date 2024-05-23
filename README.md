## News 


## Introduction

BAFormer is referenced by [UNetFormer](https://www.sciencedirect.com/science/article/pii/S0924271622001654), which you can visit for details.

**GeoSeg** is an open-source  semantic segmentation toolbox based on PyTorch, [pytorch lightning](https://www.pytorchlightning.ai/) and [timm](https://github.com/rwightman/pytorch-image-models), 
which mainly focuses on developing advanced Vision Transformers for remote sensing image segmentation.


## Major Features

- Unified Benchmark

  we provide a unified training script for various segmentation methods.
  
- Simple and Effective

  Thanks to **pytorch lightning** and **timm** , the code is easy for further development.
  
- Supported Remote Sensing Datasets
 
  - [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) 
  - [UAVid](https://uavid.nl/)
  - [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)
  - More datasets will be supported in the future.
  
- Multi-scale Training and Testing
- Inference on Huge Remote Sensing Images

## Supported Networks

- Vision Transformer

  - [UNetFormer](https://www.sciencedirect.com/science/article/pii/S0924271622001654)
  - [DC-Swin](https://ieeexplore.ieee.org/abstract/document/9681903)
  - [BANet](https://www.mdpi.com/2072-4292/13/16/3065)
  
- CNN
 
  - [MANet](https://ieeexplore.ieee.org/abstract/document/9487010) 
  - [ABCNet](https://www.sciencedirect.com/science/article/pii/S0924271621002379)
  - [A2FPN](https://www.tandfonline.com/doi/full/10.1080/01431161.2022.2030071)
  
## Folder Structure

Prepare the following folders to organize this repo:
```none
airs
├── GeoSeg (code)
├── pretrain_weights (save the pretrained weights like vit, swin, etc)
├── model_weights (save the model weights)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (Merge Train and Val)
│   ├── mapcup
│   │   ├── train (processed)
│   │   ├── val (processed)
│   │   ├── test(processed)
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── potsdam (the same with vaihingen)
```

## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r GeoSeg/requirements.txt
```

## Pretrained Weights

[Baidu Disk](https://pan.baidu.com/s/1foJkxeUZwVi5SnKNpn6hfg) : 1234 

[Google Drive](https://drive.google.com/drive/folders/1ELpFKONJZbXmwB5WCXG7w42eHtrXzyPn?usp=sharing)

## Data Preprocessing

Download the datasets from the official website and split them yourself.

**Vaihingen**

Generate the training set.
```
python tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/train_images" \
--mask-dir "data/vaihingen/train_masks" \
--output-img-dir "data/vaihingen/train/images_1024" \
--output-mask-dir "data/vaihingen/train/masks_1024" \
--mode "train" --split-size 1024 --stride 512
```
Generate the val set.
```
python tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/val_images" \
--mask-dir "data/vaihingen/val_masks" \
--output-img-dir "data/vaihingen/val/images" \
--output-mask-dir "data/vaihingen/val/masks" \
--mode "train" --split-size 1024 --stride 512
```

Generate the testing set.
```
python ./tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks_eroded" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded
```
Generate the masks_1024_rgb (RGB format ground truth labels) for visualization.
```
python GeoSeg/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt
```
As for the validation set, you can select some images from the training set to build it.

**Potsdam**
```
python ./tools/potsdam_patch_split.py \
--img-dir "data/potsdam/train_images" \
--mask-dir "data/potsdam/train_masks" \
--output-img-dir "data/potsdam/train/images" \
--output-mask-dir "data/potsdam/train/masks" \
--mode "train" --split-size 1024 --stride 1024 --rgb-image 
```

```
python ./tools/potsdam_patch_split.py \
--img-dir "data/potsdam/val_images" \
--mask-dir "data/potsdam/val_masks" \
--output-img-dir "data/potsdam/val/images" \
--output-mask-dir "data/potsdam/val/masks" \
--mode "train" --split-size 1024 --stride 1024 --rgb-image 
```

```
python ./tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks_eroded" \
--output-img-dir "data/potsdam/test/images" \
--output-mask-dir "data/potsdam/test/masks" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded --rgb-image
```

```
python GeoSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt --rgb-image
```

**LoveDA**
```
python ./tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
python ./tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
python ./tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
python ./tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```


## Training

"-c" means the path of the config, use different **config** to train different models.
```
---vaihingen
python ./train_supervision.py -c ./config/vaihingen/baformer.py
python ./train_supervision.py -c ./config/vaihingen/unetformer.py
python ./train_supervision.py -c ./config/vaihingen/baformer.py
---potsdam
python ./train_supervision.py -c ./config/potsdam/baformer.py
---LoveDA
python ./train_supervision.py -c ./config/loveda/baformer.py
---mapcup
python ./train_supervision.py -c ./config/mapcup/baformer.py
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format

**Vaihingen**
```
//dcswin
python ./vaihingen_test.py -c ./config/vaihingen/dcswin.py -o fig_results/vaihingen/dcswin --rgb -t 'd4'
//unetformer
python ./vaihingen_test.py -c ./config/vaihingen/unetformer.py -o fig_results/vaihingen/unetformer --rgb -t 'd4'
//baformer
python ./vaihingen_test.py -c ./config/vaihingen/baformer.py -o fig_results/vaihingen/baformer --rgb -t 'd4'
//baformer-t
python ./vaihingen_test.py -c ./config/vaihingen/baformer-t.py -o fig_results/vaihingen/baformer-t --rgb -t 'd4'
```

**Potsdam**
```
//dcswin
python GeoSeg/potsdam_test.py -c GeoSeg/config/potsdam/dcswin.py -o fig_results/potsdam/dcswin --rgb -t 'lr'
//uentformer
python ./potsdam_test.py -c ./config/potsdam/unetformer.py -o fig_results/potsdam/unetformer --rgb -t 'lr'
//baformer
python ./potsdam_test.py -c ./config/potsdam/baformer.py -o fig_results/potsdam/baformer --rgb -t 'lr'
```

**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))
```
python ./loveda_test.py -c ./config/loveda/baformer.py -o fig_results/loveda/baformer -t 'd4'
```

**Mapcup**
```
python ./mapcup_test.py -c ./config/mapcup/baformer.py -o fig_results/mapcup/baformer --rgb -t 'lr'
python ./mapcup_test.py -c ./config/mapcup/baformer.py -o fig_results/mapcup/baformer --rgb -t 'd4'
# infernce
python ./mapcup_inference.py -c ./config/mapcup/baformer.py -o data/mapcup/predict/pre_baformer --rgb -t 'lr'
```

## Inference on huge remote sensing image
```
python ./inference_huge_image.py \
-i data/vaihingen/test_images \
-c ./config/vaihingen/baformer.py \
-o fig_results/vaihingen/baformer_huge \
-t 'lr' -ph 512 -pw 512 -b 2 -d "pv"
```

## Reproduction Results
|  Method    |  Dataset  |  F1   |  OA   |  mIoU |
|:----------:|:---------:|:-----:|:-----:|------:|
| BAFormer   | Vaihingen | 91.5  | 91.7  | 84.5  |
| BAFormer   |  Potsdam  | 93.2  | 92.2  | 87.3  |
| BAFormer   |  LoveDA   |   -   |   -   | 53.6  | 
| BAFormer  |  Mapcup   | 90.7  | 90.8  | 83.1  | 
| BAFormer-T | Vaihingen | 91.2  | 91.6  | 84.1  |
| BAFormer-T |  Potsdam  | 92.8  | 91.3  | 86.4  |


Due to some random operations in the training stage, reproduced results (run once) are slightly different from the reported in paper.



## Acknowledgement

We wish **GeoSeg** could serve the growing research of remote sensing by providing a unified benchmark 
and inspiring researchers to develop their own segmentation networks. Many thanks the following projects's contributions to **GeoSeg**.
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
