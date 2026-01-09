import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------- CONFIG ----------------
HF_REPO = "DOReilly2/swin_regressor"  # Hugging Face repo
DEVICE = "cpu"  # or "cuda" if available

dimension_labels = [
    "Layers of the Landscape_embedding",
    "Landform_embedding",
    "Biodiversity_embedding",
    "Color and Light_embedding",
    "Compatibility_embedding",
    "Archetypal Elements_embedding",
    "Character of Peace and Silence_embedding"
]

# ---------------- UTILITIES ----------------
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor for model."""
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = image.convert("RGB")
    return val_transform(image).unsqueeze(0).to(DEVICE)

# ---------------- LOAD CONTEXT EMBEDDINGS ----------------
@st.cache_resource
def load_context_embeddings():
    embeddings = {}
    for label in dimension_labels:
        path = hf_hub_download(repo_id=HF_REPO, filename=f"context_embeddings/{label}.pt")
        emb = torch.load(path, map_location="cpu")
        embeddings[label] = emb.squeeze()
    return embeddings

context_embeddings = load_context_embeddings()

# ---------------- MODEL ----------------
class MultiContextSwinRegressor(nn.Module):
    def __init__(self, context_embeddings: dict):
        super().__init__()
        # SwinV2 pretrained feature extractor
        self.swin = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        self.swin.head = nn.Identity()  # remove classifier head

        # context embeddings
        self.context_embeddings = nn.ParameterDict({
            label: nn.Parameter(context_embeddings[label].float().unsqueeze(0), requires_grad=False).squeeze(0)
            for label in context_embeddings
        })

        # per-dimension fusion heads
        self.fusion_heads = nn.ModuleDict({
            label: nn.Sequential(
                nn.Linear(1024 + self.context_embeddings[label].shape[0], 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
            for label in self.context_embeddings
        })

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # extract image features
        image_feat = self.swin(image)  # [B, 1024]
        outputs = []
        for label in self.context_embeddings:
            context = self.context_embeddings[label].expand(image_feat.size(0), -1)
            fused = torch.cat([image_feat, context], dim=1)
            score = self.fusion_heads[label](fused)
            outputs.append(score)
        return torch.cat(outputs, dim=1)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO, filename="swin_regressor.pt")
    state_dict = torch.load(model_path, map_location="cpu")

    model = MultiContextSwinRegressor(context_embeddings)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    st.info("Model loaded successfully (raw state_dict).")
    return model

model = load_model()

# ---------------- STREAMLIT UI ----------------
st.title("CLASS 2.0")

uploaded_files = st.file_uploader(
    "Upload landscape images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    # CSV header
    table_text = "Image Name," + ",".join(dimension_labels) + "\n"

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_tensor = preprocess_image(image)

        with torch.no_grad():
            preds = model(image_tensor)  # [1, D]
            predicted_scores = preds.squeeze(0).cpu().numpy()*6  # numpy array
            predicted_scores = torch.clamp(predicted_scores, min=1.0, max=6.0)
        # Build CSV row
        row = uploaded_file.name + "," + ",".join([f"{float(score):.2f}" for score in predicted_scores]) + "\n"
        table_text += row

    st.subheader("Copy-Paste Table (Excel Friendly)")
    st.text_area("Results", table_text, height=400)
    st.download_button("Download as CSV", table_text, "predictions.csv", "text/csv")









