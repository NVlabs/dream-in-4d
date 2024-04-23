# Dream-in-4D

This repository is the official PyTorch implementation of Dream-in-4D introduced in the paper:

[**A Unified Approach for Text- and Image-guided 4D Scene Generation**](https://arxiv.org/abs/2311.16854)
[*Yufeng Zheng*](https://ait.ethz.ch/people/zhengyuf),
[*Xueting Li**](https://research.nvidia.com/person/xueting-li),
[*Koki Nagano**](https://luminohope.org/),
[*Sifei Liu*](https://sifeiliu.net/),
[*Otmar Hilliges*](https://ait.ethz.ch/people/hilliges),
[*Shalini De Mello*](https://research.nvidia.com/person/shalini-de-mello)
CVPR 2024.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

| [Project Page](https://research.nvidia.com/labs/nxp/dream-in-4d/) | [Arxiv](https://arxiv.org/abs/2311.16854) 

<img src="https://github.com/NVlabs/dream-in-4d/raw/master/asset/teaser_1280_small.gif" width="800">

- **This code is forked from [threestudio](https://github.com/threestudio-project/threestudio), commit [2c20227](https://github.com/threestudio-project/threestudio/tree/2c202276747a892cfc1ded8e27a005715be8f5f2)**

## Installation

### Install threestudio

**This part is the same as original [threestudio](https://github.com/threestudio-project/threestudio). Please check the original repo for detailed installation instructions.**

```sh
python3 -m virtualenv venv
. venv/bin/activate

python3 -m pip install --upgrade pip

# Install pytorch
pip install torch torchvision

#  (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions
pip install ninja

# Install dependencies
pip install -r requirements.txt

# Login to huggingface to use deepfloyd (for the image-to-4D task)
huggingface-cli login
```
### Install MVDream
MVDream multi-view diffusion model is provided in a different codebase. Install it by:

```sh
git clone https://github.com/bytedance/MVDream extern/MVDream
pip install -e extern/MVDream 
```
### Install dream-in-4D dependencies
```sh
# Only run this if you already have a threestudio environment and didn't re-install requirements.txt from our repo
pip install av 
```





## Quickstart
We modified the script to take multiple config files. The first config file is the shared training configurations, the second (and third) one(s) are the prompts and the subjects. Usually, you only need to modified the second (and third) prompts.  
### Stage 1
```sh
# Text-to-3D
# (Optional) system.SD_view can be used to disable SD on the back views. 
# SD_view=180 --> all views ([-180, 180]) are used. 
# SD_view=145 --> frontal and side views ([-145, 145]) are used. 
python launch.py --config configs/stage_1/mvdream-sd21-sd.yaml configs/stage_1/text-to-3D/dog_superhero.yaml --train --gpu 0 system.SD_view=180

# Image-to-3D
# This is our implemented version using zero123 and deep-floyd-if guidance, which converges faster than threestudio's implementation.
python launch.py --config configs/stage_1/magic123-coarse-if-new.yaml configs/stage_1/image-to-3D/corgi.yaml --train --gpu 0  

# Personalized-3D
# In configs/stage_1/personalized_3D/subjects/dog[#].yaml, we provide the lora attention processor weights for the personalized StableDiffusion models trained with Dreambooth.
# The dreambooth loras are trained with: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py
# See instructions here: https://github.com/huggingface/diffusers/tree/main/examples/dreambooth#training-with-low-rank-adaptation-of-large-language-models-lora
python launch.py --config configs/stage_1/mvdream-sd21-sd.yaml configs/stage_1/personalized_3D/prompts/superhero_sks_dog_wearing_red_cape_is_flying_through_the_sky.yaml configs/stage_1/personalized_3D/subjects/dog8.yaml --train --gpu 0 system.SD_view=180
```

### Stage 2
```sh
# 3D-to-4D
# In configs/stage_2/prompts/[config_name].yaml, modify the text prompt and set system.geometry_convert_from to the ckpt from the static stage
# (Optional) setting system.guidance.num_hifa_steps=4 can leads to more stable motions, at the cost of training time. By default, system.guidance.num_hifa_steps=1. 
python launch.py --config configs/stage_2/stage2_zeroscope_144x80.yaml configs/stage_2/prompts/fish.yaml --train --gpu 0 
```
### Resume from checkpoints

If you want to resume from a checkpoint, do:
```sh
# Resume training from the last checkpoint, you may replace last.ckpt with any other checkpoints
python launch.py --config path/to/trial/dir/configs/parsed.yaml --train --gpu 0 resume=path/to/trial/dir/ckpts/last.ckpt
```
Also check the [threestudio](https://github.com/threestudio-project/threestudio) repo for a complete guide on various features.
## Credits

This code is built on the [threestudio-project](https://github.com/threestudio-project/threestudio) and the [MVDream-project](https://github.com/bytedance/MVDream-threestudio). Thanks to the maintainers for their contribution to the community!

## Citing

If you find Dream-in-4D helpful, please consider citing:

```
@InProceedings{zheng2024unified,
  title     = {A Unified Approach for Text- and Image-guided 4D Scene Generation},
  author    = {Yufeng Zheng and Xueting Li and Koki Nagano and Sifei Liu and Otmar Hilliges and Shalini De Mello},
  booktitle = {CVPR},
  year      = {2024}
}

```
