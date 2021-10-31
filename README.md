# MOS-Multi-Task-Face-Detect

## Introduction
This repo is the official implementation of "MOS: A Low Latency and Lightweight Framework for Face Detection, Landmark Localization, and Head Pose Estimation"
This repo is an implementation of PyTorch. MOS is a low latency and lightweight architecture for face detection, facial landmark localization and head pose estimation.It aims to bridge the gap between research and industrial communities.
For more details, please refer to our [report on Arxiv](https://arxiv.org/abs/2110.10953).

## Updates
* 【2021/10/31】 We have released the training data (widerface with pose label). The pytorch inference code of MOS-S and MOS-M has been released!
* 【2021/10/22】 We have released our paper on [Arxiv](https://arxiv.org/abs/2110.10953).
* 【2021/10/15】 "MOS: A Low Latency and Lightweight Framework for Face Detection, Landmark Localization, and Head Pose Estimation" has been accepted by BMVC2021.

## Comming soon
- [ ] Tensorrt inference code.
- [ ] Openvino inference code.
- [ ] Ncnn inference code.
- [ ] The fastest version: MOS-tiny.

## Benchmark
#### Light Models.

|Model |backbone |easy | medium |hard| weights |
| ------        |:---:  |  :---:       |:---:     |:---:  | :---: |
|MOS-M|mobilenetV2  |94.08  | 93.21 |88.06 | MOS-M.pth) |
|MOS-S|shufflenetV2 |93.28 | 92.12 |86.97 | MOS-S.pth) |







## Quick Start

<details>
<summary>Installation</summary>

Step1. Install MOS.
```shell
git clone https://github.com/lyp-deeplearning/MOS-Multi-Task-Face-Detect.git
cd MOS-Multi-Task-Face-Detect
conda create -n MOS python=3.8.5
conda activate MOS
pip install -r requirements.txt
cd models/DCNv2/
python setup.py build develop
```

Step2. Run Pytorch inference demo.
```shell
## run the MOS-M model 
python detect_picture.py --network cfg_mos_m --trained_model ./test_weights/MOS-M.pth
## run the MOS-S model
python detect_picture.py --network cfg_mos_s --trained_model ./test_weights/MOS-S.pth
```


