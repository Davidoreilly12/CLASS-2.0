import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import swin_v2_b
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# --- Config ---------------------------------------------------------------
HF_REPO = "DOReilly2/swin_regressor"  # Hugging Face repo
DEVICE = "cpu"  # change to "cuda" if available in your runtime

# Human-readable labels for UI (keeps CSV pretty)
dimension_labels = [
    "Layers of the Landscape",
    "Landform",
    "Biodiversity",
    "Color and Light",
    "Compatibility",
    "Archetypal Elements",
    "Character of Peace and Silence"
]

# Map human labels to exact filenames in your HF repo (exact spelling/case/spacing)
filename_map = {
    "Layers of the Landscape": "Layers of the Landscape_embedding.pt",
    "Landform": "Landform_embedding.pt",
    "Biodiversity": "Biodiversity_embedding.pt",
    "Color and Light": "Color and Light_embedding.pt",
    "Compatibility": "Compatibility_embedding.pt",
    "Archetypal Elements": "Archetypal Elements_embedding.pt",
    "Character of Peace and Silence": "Character of Peace and Silence_embedding.pt"
}

# --- Utilities ------------------------------------------------------------
class TargetScaler:
    """Restore target scaling saved during training."""
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        # x: tensor in scaled (z) space
        # ensure dtype/shape match
        return x * self.std.to(x.device) + self.mean.to(x.device)

# --- Load context embeddings ----------------------------------------------
@st.cache_resource
def load_context_embeddings():
    embeddings = {}
    for label in dimension_labels:
        filename = f"context_embeddings/{filename_map[label]}"
        path = hf_hub_download(repo_id=HF_REPO, filename=filename)
        emb = torch.load(path, map_location="cpu")
        # ensure it's a 1-D tensor
        emb = emb.squeeze()
        embeddings[label] = emb
    return embeddings

context_embeddings = load_context_embeddings()

# --- Model definition -----------------------------------------------------
class MultiContextSwinRegressor(nn.Module):
    def __init__(self, context_embeddings: dict):
        super().__init__()
        # load Swin backbone (feature extractor)
        self.swin = swin_v2_b(weights="IMAGENET1K_V1")
        # remove classification head, keep representation
        self.swin.head = nn.Identity()

        # store context embeddings as non-trainable params
        self.context_embeddings = nn.ParameterDict({
            label: nn.Parameter(context_embeddings[label].float().unsqueeze(0), requires_grad=False).squeeze(0)
            for label in context_embeddings
        })

        # build a small fusion head for each dimension
        # swin_v2_b outputs 1024-d features by default
        base_feat_dim = 1024
        self.fusion_heads = nn.ModuleDict({
            label: nn.Sequential(
                nn.Linear(base_feat_dim + self.context_embeddings[label].shape[0], 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
            for label in self.context_embeddings
        })

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B, C, H, W]
        image_feat = self.swin(image)  # -> [B, 1024]
        outputs = []
        for label in self.context_embeddings:
            context = self.context_embeddings[label].expand(image_feat.size(0), -1)
            fused = torch.cat([image_feat, context], dim=1)
            score = self.fusion_heads[label](fused)  # [B,1]
            outputs.append(score)
        # concat -> [B, D]
        return torch.cat(outputs, dim=1)

# --- Load model + scaler -------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    # download checkpoint
    model_path = hf_hub_download(repo_id=HF_REPO, filename="swin_regressor.pt")
    checkpoint = torch.load(model_path, map_location="cpu")

    # instantiate model and load weights
    model = MultiContextSwinRegressor(context_embeddings)
    state_dict = checkpoint.get("model_state_dict", checkpoint)  # support either full ckpt or raw state_dict
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    # restore scaler (must exist in your saved checkpoint)
    if "scaler_mean" in checkpoint and "scaler_std" in checkpoint:
        scaler = TargetScaler(mean=checkpoint["scaler_mean"], std=checkpoint["scaler_std"])
    else:
        # fallback: identity scaler (will return scaled predictions unchanged)
        scaler = TargetScaler(mean=[0.0] * len(dimension_labels), std=[1.0] * len(dimension_labels))
        st.warning("Scaler info not found in checkpoint — outputs will not be unscaled properly.")

    st.info("Model and scaler loaded.")
    return model, scaler

model, scaler = load_model_and_scaler()

# --- Image preprocessing -------------------------------------------------
val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = val_transform(image)
    return image.unsqueeze(0).to(DEVICE)  # [1, C, H, W]

# --- Streamlit UI --------------------------------------------------------
st.title("CLASS 2.0 — Landscape Contemplative Scorer")

uploaded_files = st.file_uploader("Upload landscape images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # CSV header (human readable)
    header = "Image Name," + ",".join(dimension_labels) + "\n"
    table_text = header

    for uploaded_file in uploaded_files:
        # read image
        image = Image.open(uploaded_file)
        image_tensor = preprocess_image(image)

        # predict (with inverse scaling)
        with torch.no_grad():
            preds_scaled = model(image_tensor)               # [1, D], scaled (z) space
            preds_unscaled = scaler.inverse_transform(preds_scaled)  # [1, D] in original units
            predicted_scores = preds_unscaled.squeeze(0).cpu().numpy()  # [D] numpy

        # format row
        row = uploaded_file.name + "," + ",".join([f"{float(score):.2f}" for score in predicted_scores]) + "\n"
        table_text += row

    st.subheader("Copy-Paste Table (Excel Friendly)")
    st.text_area("Results", table_text, height=400)

    st.download_button("Download as CSV", table_text, "predictions.csv", "text/csv")


