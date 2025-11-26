import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import swin_v2_b
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download

# Define contemplative dimensions
dimension_names = [
    "Layers of the Landscape_embedding", "Landform_embedding", "Biodiversity_embedding", "Color and Light_embedding",
    "Compatibility_embedding", "Archetypal Elements_embedding", "Character of Peace and Silence_embedding"
]

# Hugging Face repo
HF_REPO = "DOReilly2/swin_regressor"  # Replace with your repo name

# Load context embeddings from Hugging Face
@st.cache_resource
def load_context_embeddings():
    embeddings = {}
    for dim in dimension_names:
        filename = f"context_embeddings/{dim.replace(' ', '_')}.pt"
        path = hf_hub_download(repo_id=HF_REPO, filename=filename)
        embeddings[dim] = torch.load(path, map_location="cpu")
    return embeddings

context_embeddings = load_context_embeddings()

# Define model
class MultiContextSwinRegressor(nn.Module):
    def __init__(self, context_embeddings):
        super().__init__()
        self.swin = swin_v2_b(weights="IMAGENET1K_V1")
        self.swin.head = nn.Identity()
        self.context_embeddings = nn.ParameterDict({
            dim: nn.Parameter(embed.squeeze(), requires_grad=False)
            for dim, embed in context_embeddings.items()
        })
        self.fusion_heads = nn.ModuleDict({
            dim: nn.Sequential(
                nn.Linear(1024 + self.context_embeddings[dim].shape[0], 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
            for dim in self.context_embeddings
        })

    def forward(self, image):
        image_feat = self.swin(image)
        outputs = []
        for dim in self.context_embeddings:
            context = self.context_embeddings[dim].expand(image_feat.size(0), -1)
            fused = torch.cat([image_feat, context], dim=1)
            score = self.fusion_heads[dim](fused)
            outputs.append(score)
        return torch.cat(outputs, dim=1)

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO, filename="swin_regressor.pt")
    checkpoint = torch.load(model_path, map_location="cpu")
    model = MultiContextSwinRegressor(context_embeddings)
    model.load_state_dict(checkpoint)  # ✅ Directly load state dict
    model.eval()
    st.write("Model loaded successfully!")
    return model


model = load_model()

# Image transform
val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    image = image.convert("RGB")
    image = val_transform(image)
    return image.unsqueeze(0)

# Streamlit UI
st.title("CLASS 2.0")
uploaded_files = st.file_uploader("Upload landscape images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    table_text = "Image Name," + ",".join(dimension_names) + "\n"
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            predicted_scores = model(image_tensor)
            predicted_scores = predicted_scores.squeeze().numpy() * 6.0
        row = uploaded_file.name + "," + ",".join([f"{score:.2f}" for score in predicted_scores]) + "\n"
        table_text += row

    st.subheader("Copy-Paste Table (Excel Friendly)")
    st.text_area("Results", table_text, height=400)

    st.download_button("Download as CSV", table_text, "predictions.csv", "text/csv")

