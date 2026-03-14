# 🌐 MTPano: Multi-Task Panoramic Scene Understanding via Label-Free Integration of Dense Prediction Priors

<p align="center">
  <video src="https://raw.githubusercontent.com/Evergreen0929/MTPano/main/assets/mtpano_demo_fast.mp4" width="900" autoplay loop muted playsinline></video>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.05330">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="Paper">
  </a>
  <a href="https://huggingface.co/jdzhang0929/MTPano">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-ffc107?color=ffc107&logoColor=white" alt="Hugging Face">
  </a>
  <a href="https://evergreen0929.github.io/projects/mtpano.html">
    <img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Page">
  </a>
</p>

## 📢 News
- **[2026/03]** 🚀 We release the inference code and pretrained weights on Hugging Face! You can now run local inference.
- **[TODO]** 🛠️ The complete training code and the synthetic data generation pipeline will be released progressively. Stay tuned!

## 📜 Introduction

This repository contains the official implementation of **MTPano**:
> **MTPano: Multi-Task Panoramic Scene Understanding via Label-Free Integration of Dense Prediction Priors**  
> *Jingdong Zhang, Xiaohang Zhan, Lingzhi Zhang, Yizhou Wang, Zhengming Yu, Jionghao Wang, Wenping Wang, Xin Li*  

**MTPano** is a robust multi-task panoramic foundation model established by a label-free training pipeline. It addresses the critical challenges of geometric distortions and data scarcity in 360° vision:
- **Label-Free Training Pipeline:** We circumvent data scarcity by projecting panoramas into perspective patches, generating pseudo-labels using off-the-shelf perspective foundation models, and re-projecting them for patch-wise supervision.
- **PD-BridgeNet:** We propose the Panoramic Dual BridgeNet to tackle the interference between task types. It explicitly disentangles rotation-invariant (e.g., depth, segmentation) and rotation-variant (e.g., surface normals) features via geometry-aware modulation.

MTPano achieves state-of-the-art performance on multiple benchmarks including Structured3D and Stanford2D3D.

---

<p align="center">
  <img alt="architecture" src="assets/main_arch_1.png" width="900">
  <br>
    <em>The overview of our proposed MTPano framework and PD-BridgeNet architecture.</em>
</p>

## ⚙️ Environment Setup

The codebase is built with `torch==2.5.0` and `torchvision==0.20.0`. We provide a convenient shell script to configure the environment in one click.

```bash
# Clone the repository
git clone https://github.com/Evergreen0929/MTPano.git
cd MTPano

# Run the setup script to install all dependencies
pip install torch==2.5.0 torchvision==0.20.0  
bash setup_env.sh
```

## 🤗 Pretrained Weights

We host our pretrained model weights on **Hugging Face**. 
Currently, we provide two versions of weights: `140k` and `408k`. 

Our inference script integrates `huggingface_hub`, which means **you don't need to manually download them**. The script will automatically fetch and cache the requested weights directly from `jdzhang0929/MTPano` during your first run.

### Model Comparison & Training Data
| Dataset | 140k Weights | 408k Weights |
| :--- | :---: | :---: |
| [Structured3D](https://github.com/bertjiazheng/Structured3D/) | 16.6k | 16.6k |
| Sun360 | 34.3k | 34.3k |
| [Matterport3D](https://github.com/niessner/Matterport/) | 7.9k | 7.9k |
| [DiT360_gen](https://github.com/Insta360-Research-Team/DiT360) (Synthetic) | 82k | 182k |
| [Hunyuan_gen](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0) (Synthetic) | - | 100k |
| [ZInD](https://github.com/zillow/zind) | - | 67.4k |
| **Total Images** | **140k** | **408k** |

## 🚀 Inference

You can easily run MTPano on a single panoramic image or a folder of images. The script will automatically generate Semantic Segmentation, Depth Maps, and Surface Normals. 

```bash
python inference.py --input ./examples --output ./results
```

### Advanced Usage
If you want to render a **Flythrough Video**, simply add the corresponding flags:

```bash
python inference.py \
    --input ./examples \
    --output ./results \
    --weight 408k \
    --save_video 
```
- `--weight`: Choose between `140k` or `408k` (defaults to `408k`).
- `--save_video`: Renders a 15-second cinematic flythrough video showing the multi-task predictions.

## 🎓 Citation

If you find our work or this repository useful, please consider giving us a ⭐ star and citing our paper:

```bibtex
@article{zhang2026mtpano,
  title={MTPano: Multi-Task Panoramic Scene Understanding via Label-Free Integration of Dense Prediction Priors},
  author={Zhang, Jingdong and Zhan, Xiaohang and Zhang, Lingzhi and Wang, Yizhou and Yu, Zhengming and Wang, Jionghao and Wang, Wenping and Li, Xin},
  journal={arXiv preprint},
  year={2026}
}
```

## 📬 Contact
If you have any questions, please feel free to reach out to [Jingdong Zhang](https://evergreen0929.github.io/).

## 👏 Acknowledgement
This project heavily relies on the excellent works from [DINOv3](https://github.com/facebookresearch/dinov2), [MoGe](https://github.com/microsoft/MoGe), and [BridgeNet](https://github.com/Evergreen0929/BridgeNet). We express our sincere gratitude to the authors for open-sourcing their code and models.

## Related Project
**[BridgeNet](https://github.com/Evergreen0929/EEMTL/tree/main/BridgeNet):** This project proposes BridgeNet for Multi-task Dense Predictions, leveraging Bridge-Feature as intermediate representations.  
**[HiTTs](https://github.com/Evergreen0929/EEMTL/tree/main/HiTTs):** This project targeting Partially-Supervisioned Multi-Task Dense Predictions with Hierarchical-Task-Tokens (HiTTs).
