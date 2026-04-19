```markdown
# SketchForge
> Unpaired face sketch ↔ photo translation · CycleGAN · CUHK-CUFS

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cycle-gan-8yjwgqsepaiam7dzrzypxc.streamlit.app/)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

## Overview
SketchForge trains a CycleGAN to translate between face sketches and photographs — entirely without paired examples. Feed it a pencil sketch, get a photorealistic face. Feed it a photo, get a hand-drawn portrait. No alignment, no paired labels, just cycle consistency.

## Architecture
| Component | Role |
|-----------|------|
| **G_AB** | Sketch → Photo (ResNet, 6 blocks) |
| **G_BA** | Photo → Sketch (ResNet, 6 blocks) |
| **D_A / D_B** | PatchGAN discriminators (LSGAN loss) |

Cycle consistency (`λ=10`) enforces A→B→A reconstruction. Identity loss (`λ=5`) preserves colour and tone. Learning rate decays linearly after epoch 15.

## Training Config
```
dataset   : CUHK-CUFS (max 5,000 images/domain)
img_size  : 128×128
batch     : 4
epochs    : 30
lr        : 0.0002  |  betas: (0.5, 0.999)
λ_cycle   : 10      |  λ_identity: 5
adv_loss  : MSELoss (LSGAN)
platform  : Kaggle T4 ×2
```

## Results
| Metric | Value |
|--------|-------|
| Cycle Reconstruction SSIM | _see pkl_ |
| Cycle Reconstruction PSNR | _see pkl_ |

## Run Locally
```bash
git clone https://github.com/your-username/sketchforge
cd sketchforge
pip install -r requirements.txt
streamlit run app.py
```
