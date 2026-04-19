import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# HF MODELS
# ========================
G_AB_URL = "https://huggingface.co/zentom/cyclegangAB/resolve/main/cyclegan_G_AB_final.pth"
G_BA_URL = "https://huggingface.co/zentom/cyclegangba/resolve/main/cyclegan_G_BA_final.pth"


# ========================
# EXACT ARCHITECTURE (FROM YOUR NOTEBOOK)
# ========================
class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, f=64, n_blocks=6):  # ← IMPORTANT: 6 blocks
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, f, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(f),
            nn.ReLU(True),

            nn.Conv2d(f, f*2, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(f*2),
            nn.ReLU(True),

            nn.Conv2d(f*2, f*4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(f*4),
            nn.ReLU(True),
        ]

        for _ in range(n_blocks):
            layers.append(ResNetBlock(f*4))

        layers += [
            nn.ConvTranspose2d(f*4, f*2, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(f*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(f*2, f, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(f),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(f, out_ch, 7, 1, 0),  # last layer HAS bias
            nn.Tanh(),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ========================
# FIX STATE DICT
# ========================
def fix_state_dict(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")  # remove DataParallel
        new_state[k] = v              # DO NOT rename net → model
    return new_state


# ========================
# LOAD MODEL
# ========================
@st.cache_resource
def load_model(url):
    model = ResNetGenerator().to(DEVICE)

    response = requests.get(url)
    state_dict = torch.load(BytesIO(response.content), map_location=DEVICE)

    state_dict = fix_state_dict(state_dict)

    model.load_state_dict(state_dict)
    model.eval()
    return model


# ========================
# TRANSFORMS (MATCH TRAINING)
# ========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def tensor_to_image(t):
    t = t.squeeze().cpu()
    t = (t * 0.5) + 0.5
    return transforms.ToPILImage()(t.clamp(0, 1))


# ========================
# UI
# ========================
st.title("🔥 CycleGAN: Sketch ↔ Image")

mode = st.radio(
    "Select Conversion",
    ["Sketch → Image", "Image → Sketch"]
)

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_column_width=True)

    G_AB = load_model(G_AB_URL)
    G_BA = load_model(G_BA_URL)

    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if mode == "Sketch → Image":
            out = G_AB(x)
        else:
            out = G_BA(x)

    result = tensor_to_image(out)

    st.image(result, caption="Output", use_column_width=True)


# ========================
# DEBUG VIEW (VERY USEFUL)
# ========================
st.markdown("### 🔍 Debug View")

if uploaded:
    col1, col2 = st.columns(2)

    with torch.no_grad():
        out_ab = tensor_to_image(G_AB(x))
        out_ba = tensor_to_image(G_BA(x))

    with col1:
        st.image(out_ab, caption="G_AB (Sketch → Photo)")

    with col2:
        st.image(out_ba, caption="G_BA (Photo → Sketch)")