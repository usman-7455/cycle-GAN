

````markdown
# SketchForge
> Unpaired face sketch ↔ photo translation using CycleGAN — CUHK-CUFS dataset

[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?logo=streamlit)](https://cycle-gan-8yjwgqsepaiam7dzrzypxc.streamlit.app/)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4%20×2-20BEFF?logo=kaggle)

---

## What it does
SketchForge learns to translate between pencil face sketches and photographs without ever seeing a matched pair. Given a sketch it produces a photorealistic face; given a photo it produces a hand-drawn portrait. Trained on the CUHK-CUFS dataset using cycle consistency — no paired supervision required.

Two ResNet generators (`G_AB`: Sketch→Photo, `G_BA`: Photo→Sketch) compete against two PatchGAN discriminators. A 50-image history buffer on each discriminator stabilises training and prevents oscillation.

---

## Architecture

| Component | Role | Details |
|-----------|------|---------|
| `G_AB` | Sketch → Photo | ResNet generator, 6 res blocks, InstanceNorm, Tanh output |
| `G_BA` | Photo → Sketch | Identical architecture, shared `opt_G` via `itertools.chain` |
| `D_A` | Sketch discriminator | 4-layer PatchGAN, LeakyReLU(0.2), no Sigmoid, raw logits |
| `D_B` | Photo discriminator | Same PatchGAN as D_A, separate Adam optimizer |

**Forward pass (one step):**
```
real_A (sketch) → G_AB → fake_B → G_BA → rec_A   (cycle A)
real_B (photo)  → G_BA → fake_A → G_AB → rec_B   (cycle B)
real_A → G_BA → id_A  |  real_B → G_AB → id_B    (identity)
```

---

## Loss functions

| Loss | Formula | Weight |
|------|---------|--------|
| Adversarial (LSGAN) | `MSELoss` — no BCE, more stable | × 1 |
| Cycle consistency | `L1(rec_A, real_A) + L1(rec_B, real_B)` | × 10 |
| Identity | `L1(id_A, real_A) + L1(id_B, real_B)` | × 5 |
| Discriminator | Real + fake terms averaged (`× 0.5`) | × 0.5 |

Label tensors use `torch.ones_like` / `torch.zeros_like` to match PatchGAN output shape dynamically.

---

## Training config

```python
IMG_SIZE       = 128
BATCH_SIZE     = 4        # CycleGAN is memory-heavy
LR             = 0.0002
BETAS          = (0.5, 0.999)
EPOCHS         = 30
LAMBDA_CYCLE   = 10
LAMBDA_ID      = 5
N_RES_BLOCKS   = 6        # 9 in paper; reduced to 6 for Kaggle T4 memory
MAX_PER_DOMAIN = 5000
SAVE_EVERY     = 5

# LR schedule: constant for epochs 1–15, linear decay to 0 by epoch 30
```

All Conv layers initialised with `Normal(mean=0.0, std=0.02)`. `DataParallel` wraps all four models when >1 GPU is detected.

---

## Dataset

**CUHK Face Sketch Database (CUFS)** — `arbazkhan971/cuhk-face-sketch-database-cufs` on Kaggle.

- Sketches: `/sketches/` — Domain A
- Photos: `/photos/` — Domain B
- Images collected recursively, shuffled, capped at 5 000 per domain
- `UnpairedDataset` samples a **random** photo index every iteration — sketch `i` is never aligned with photo `i`
- Transform: `Resize(128) → ToTensor → Normalize([0.5]*3, [0.5]*3)`

---

## Evaluation

Metrics are computed on **cycle reconstruction** (Sketch→Photo→Sketch), not translation quality directly:

- **SSIM** — structural similarity (skimage)
- **PSNR** — peak signal-to-noise ratio in dB
- Evaluated over 5 batches, saved to `cyclegan_loss_history.pkl`

---

## Saved outputs

```
outputs/
├── cyclegan/
│   ├── domain_samples.png          ← 8 sketches + 8 photos (Cell 5)
│   ├── training_losses.png         ← G / DA+DB / Cycle curves (Cell 10)
│   ├── qualitative_AtoB.png        ← Real / Translated / Reconstructed (Cell 11)
│   ├── AtoB/epoch_001–030.png      ← fixed_A | fake_B per epoch
│   └── BtoA/epoch_001–030.png      ← fixed_B | fake_A per epoch
├── cyclegan_G_AB_final.pth         ← Sketch→Photo final weights
├── cyclegan_G_BA_final.pth         ← Photo→Sketch final weights
└── cyclegan_loss_history.pkl       ← G, DA, DB, cycle losses + ssim + psnr

checkpoints/
└── cyclegan_005.pth … cyclegan_030.pth  ← all 4 state_dicts every 5 epochs
```

---

## Run locally

```bash
git clone https://github.com/your-username/sketchforge
cd sketchforge
pip install torch torchvision streamlit scikit-image tqdm pillow matplotlib
streamlit run app.py
```

## Re-train on Kaggle

Attach dataset `arbazkhan971/cuhk-face-sketch-database-cufs`, select **T4 ×2 GPU**, run all 13 cells. Checkpoints save every 5 epochs. Final weights and loss history export to `outputs/`.

---



