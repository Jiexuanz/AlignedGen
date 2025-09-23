<div align="center">

# [NIPS 2025🔥] AlignedGen: Aligning Style Across Generated Images

[![Project Website](https://img.shields.io/badge/🌐-Project%20Website-deepgray)](https://jiexuanz.github.io/AlignedGen/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.17088-b31b1b.svg)](https://arxiv.org/abs/2509.17088 )

Jiexuan Zhang, Yiheng Du, [Qian Wang](https://akaneqwq.github.io/), Yu
Gu, [Weiqi Li](https://scholar.google.com/citations?user=SIkQdEsAAAAJ), [Jian Zhang](https://jianzhang.tech/)

School of Electronic and Computer Engineering, Peking University

</div>

----

## 🔥 Introduction


AlignedGen generates a set of images with a consistent style from a set of prompts.

For example, given the prompts: `{Anchor, Clock, Globe, Bicycle} in 3D realism style.`, AlignedGen will produce the results shown below. For more details on how to run the model, please see the [Inference section](#-inference). Additional examples are available on our [project website](https://jiexuanz.github.io/AlignedGen/).

## <img id="main" width="100%" src="asset/main.jpg"> 




## 🚩 News

- 09.23 Released paper and code.

## 🔧 Dependencies and Installation

```bash
conda create -n aligned python=3.10
conda activate aligned
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch
pip install diffusers transformers sentencepiece protobuf==3.19.0
```

## ⚡ Inference

Before running, please ensure FLUX.1-dev model is accessible to the script.
Then, run the inference script with the following command:

```
python inference.py --model_path black-forest-labs/FLUX.1-dev --style_lambda 1.1
```

This will generate the image shown at the beginning of this README.

**Note on VRAM**: If you encounter out-of-memory errors, try reducing the number of prompts or enabling the offload option within the pipeline.

## 📊 Compare With Other Methods

## <img id="compare" width="100%" src="asset/compare.png"> 


## 🌟 User-Provided Image As Style Reference

## <img id="seamless" width="100%" src="asset/seamless.png"> 

## 🤔 Control & DreamBooth


<div align="center">
  <img src="asset/supp-control.png" width="250" style="display:inline-block; margin:5px;" />
  <img src="asset/supp-dream.png" width="350" style="display:inline-block; margin:5px;" />
</div>

## ✏️ To Do List

- [x] Release the paper and code
- [ ] Release ControlNet Code
- [ ] Release DreamBooth Code
- [ ] Release Attention Map Visualization Code
- [ ] Release User-Provided Image As Style Inference Code
- [ ] Support Qwen-Image
- [ ] Support ComfyUI
- [ ] Support Gradio demo

## Acknowledgement

We appreciate the releasing codes of [StyleAligned](https://github.com/google/style-aligned)
and [Diffusers](https://github.com/huggingface/diffusers).

[//]: # (## Citation)

[//]: # (If our work assists your research, feel free to give us a star ⭐ or cite us using:)

[//]: # ()
[//]: # (```)

[//]: # (@article{zhang2025alignedgen,)

[//]: # (  title={AlignedGen: Aligning Style Across Generated Images}, )

[//]: # (  author={Jiexuan Zhang and Yiheng Du and Qian Wang and Weiqi Li and Yu Gu and Jian Zhang},)

[//]: # (  journal={arXiv preprint arXiv:2509.17088},)

[//]: # (  year={2025})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (---)

[//]: # (## ⭐️ Star History)

[//]: # ()
[//]: # ([![Star History Chart]&#40;https://api.star-history.com/svg?repos=Jiexuanz/AlignedGen&type=Date&#41;]&#40;https://www.star-history.com/#Jiexuanz/AlignedGen&Date&#41;)

[//]: # ()

