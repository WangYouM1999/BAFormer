## Version 2.0 (stable)

[Welcome to my homepage!](https://WangLibo1995.github.io)

## News 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transformer-meets-dcfam-a-novel-semantic/semantic-segmentation-on-isprs-potsdam)](https://paperswithcode.com/sota/semantic-segmentation-on-isprs-potsdam?p=transformer-meets-dcfam-a-novel-semantic)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transformer-meets-dcfam-a-novel-semantic/semantic-segmentation-on-isprs-vaihingen)](https://paperswithcode.com/sota/semantic-segmentation-on-isprs-vaihingen?p=transformer-meets-dcfam-a-novel-semantic)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-hybrid-transformer-learning-global/semantic-segmentation-on-uavid)](https://paperswithcode.com/sota/semantic-segmentation-on-uavid?p=efficient-hybrid-transformer-learning-global)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-hybrid-transformer-learning-global/semantic-segmentation-on-loveda)](https://paperswithcode.com/sota/semantic-segmentation-on-loveda?p=efficient-hybrid-transformer-learning-global)

- Pretrained Weights can be access from [Google Drive](https://drive.google.com/drive/folders/1ELpFKONJZbXmwB5WCXG7w42eHtrXzyPn?usp=sharing)
- [adaptformer](https://www.sciencedirect.com/science/article/pii/S0924271622001654) (accepted by ISPRS, [PDF](https://www.researchgate.net/profile/Libo-Wang-17/publication/361736439_adaptformer_A_UNet-like_transformer_for_efficient_semantic_segmentation_of_remote_sensing_urban_scene_imagery/links/62c2a1ed1cbf3a1d12ac1c87/adaptformer-A-UNet-like-transformer-for-efficient-semantic-segmentation-of-remote-sensing-urban-scene-imagery.pdf)) and **UAVid dataset** are supported.
- ISPRS Vaihingen and Potsdam datasets are supported. Since private sharing is not allowed, you need to download the datasets from the official website and split them by **Folder Structure**.
- More network are updated and the link of pretrained weights is provided.
- **config/loveda/dcswin.py** provides a detailed explain about **config** setting.
- Inference on huge RS images are supported (inference_huge_image.py).

## Introduction

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

  - [adaptformer](https://authors.elsevier.com/a/1fIji3I9x1j9Fs) 
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
│   ├── uavid
│   │   ├── uavid_train (original)
│   │   ├── uavid_val (original)
│   │   ├── uavid_test (original)
│   │   ├── uavid_train_val (Merge uavid_train and uavid_val)
│   │   ├── train (processed)
│   │   ├── val (processed)
│   │   ├── train_val (processed)
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

**UAVid**
```
python ./tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train_val" \
--output-img-dir "data/uavid/train_val/images" \
--output-mask-dir "data/uavid/train_val/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```
python ./tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_train" \
--output-img-dir "data/uavid/train/images" \
--output-mask-dir "data/uavid/train/masks" \
--mode 'train' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

```
python ./tools/uavid_patch_split.py \
--input-dir "data/uavid/uavid_val" \
--output-img-dir "data/uavid/val/images" \
--output-mask-dir "data/uavid/val/masks" \
--mode 'val' --split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```

**LoveDA**
```
python ./tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
python ./tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
python ./tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
python ./tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```
**thz**
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

## Training

"-c" means the path of the config, use different **config** to train different models.

```

---vaihingen
python ./train_supervision.py -c ./config/vaihingen/adaptformer.py
python ./train_supervision.py -c ./config/vaihingen/ftadaptformer.py
python ./train_supervision.py -c ./config/vaihingen/ftunetformer.py
---potsdam
python ./train_supervision.py -c ./config/potsdam/adaptformer.py
---LoveDA
python ./train_supervision.py -c ./config/loveda/adaptformer.py
---mapcup
python ./train_supervision.py -c ./config/mapcup/adaptformer.py
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
//adaptformer
python ./vaihingen_test.py -c ./config/vaihingen/adaptformer.py -o fig_results/vaihingen/diff/adaptformer --rgb -t 'd4'
//ft+adaptformer
python ./vaihingen_test.py -c ./config/vaihingen/ftadaptformer.py -o fig_results/vaihingen-ft/diff/ftadaptformer --rgb -t 'd4'
```

**Potsdam**
```
//dcswin
python GeoSeg/potsdam_test.py -c GeoSeg/config/potsdam/dcswin.py -o fig_results/potsdam/dcswin --rgb -t 'lr'
//adaptformer
python ./potsdam_test.py -c ./config/potsdam/adaptformer.py -o fig_results/potsdam/diff/adaptformer --rgb -t 'lr'
```

**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))
```
python ./loveda_test.py -c ./config/loveda/adaptformer.py -o fig_results/loveda/adaptformer -t 'd4'
```
**Thz**
```
python ./thz_test.py -c ./config/thz/adaptformer.py -o fig_results/thz/adaptformer --rgb -t 'lr'
python ./thz_test.py -c ./config/thz/adaptformer_init.py -o fig_results/thz/adaptformer_init --rgb -t 'lr'
```
**Mapcup**
```
python ./mapcup_test.py -c ./config/mapcup/adaptformer.py -o fig_results/mapcup/adaptformer --rgb -t 'lr'
python ./mapcup_test.py -c ./config/mapcup/adaptformer.py -o fig_results/mapcup/adaptformer --rgb -t 'd4'
# infernce
python ./mapcup_inference.py -c ./config/mapcup/adaptformer.py -o data/mapcup/predict/pre_adaptformer --rgb -t 'lr'
```
**UAVid** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/7302))
```
python ./inference_uavid.py \
-i 'data/uavid/uavid_test' \
-c ./config/uavid/adaptformer.py \
-o fig_results/uavid/adaptformer_r18 \
-t 'lr' -ph 1152 -pw 1024 -b 2 -d "uavid"
```

## Inference on huge remote sensing image
```
python ./inference_huge_image.py \
-i data/vaihingen/test_images \
-c ./config/vaihingen/adaptformer.py \
-o fig_results/vaihingen/adaptformer_huge \
-t 'lr' -ph 512 -pw 512 -b 2 -d "pv"
```

<div>
<img src="vai.png" width="30%"/>
<img src="pot.png" width="35.5%"/>
</div>

## Reproduction Results
|    Method     |  Dataset  |  F1   |  OA   |  mIoU | F1-me | OA-me | mIoU-me |       |
|:-------------:|:---------:|:-----:|:-----:|------:|:-----:|:-----:|:-------:|:------|
|  adaptformer   |   UAVid   |   -   |   -   | 67.63 |   -   |   -   |  67.33  |       |
|  adaptformer   | Vaihingen | 90.30 | 91.10 | 82.54 | 90.37 | 91.05 |  82.63  | 0.819 |
|  adaptformer   |  Potsdam  | 92.64 | 91.19 | 86.52 | 92.23 | 90.75 |  85.80  | 0.822 |
|  adaptformer   |  LoveDA   |   -   |   -   | 52.97 |   -   |   -   |  52.01  |       |
|  adaptformer   |    Thz    |   -   |   -   | 85.05 | 85.05 |       |         |       |
| FT-adaptformer | Vaihingen | 91.17 | 91.74 | 83.98 |       |       |         |       |
| FT-adaptformer |  Potsdam  | 93.22 | 91.87 | 87.50 |       |       |         |       |

Due to some random operations in the training stage, reproduced results (run once) are slightly different from the reported in paper.

## Citation

If you find this project useful in your research, please consider citing：

- [adaptformer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery](https://authors.elsevier.com/a/1fIji3I9x1j9Fs)
- [A Novel Transformer Based Semantic Segmentation Scheme for Fine-Resolution Remote Sensing Images](https://ieeexplore.ieee.org/abstract/document/9681903) 
- [Transformer Meets Convolution: A Bilateral Awareness Network for Semantic Segmentation of Very Fine Resolution Urban Scene Images](https://www.mdpi.com/2072-4292/13/16/3065)
- [ABCNet: Attentive Bilateral Contextual Network for Efficient Semantic Segmentation of Fine-Resolution Remote Sensing Images](https://www.sciencedirect.com/science/article/pii/S0924271621002379)
- [Multiattention network for semantic segmentation of fine-resolution remote sensing images](https://ieeexplore.ieee.org/abstract/document/9487010)
- [A2-FPN for semantic segmentation of fine-resolution remotely sensed images](https://www.tandfonline.com/doi/full/10.1080/01431161.2022.2030071)



## Acknowledgement

We wish **GeoSeg** could serve the growing research of remote sensing by providing a unified benchmark 
and inspiring researchers to develop their own segmentation networks. Many thanks the following projects's contributions to **GeoSeg**.
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
