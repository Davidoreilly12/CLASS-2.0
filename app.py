import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import swin_v2_b
from PIL import Image
import os
import numpy as np

# Define contemplative dimensions
dimension_names = [
    "Layers of the Landscape", "Landform", "Biodiversity", "Color and Light",
    "Compatibility", "Archetypal Elements", "Character of Peace and Silence"
]

# Load context embeddings
@st.cache_resource
def load_context_embeddings(load_dir="./context_embeddings"):
    embeddings = {}
    for dim in dimension_names:
        filename = f"{dim.replace(' ', '_').lower()}.pt"
        path = os.path.join(load_dir, filename)
        embeddings[dim] = torch.load(path)
    return embeddings

context_embeddings = load_context_embeddings('./context_embeddings/context_embeddings')

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
        image_feat = self.swin(image)  # [B, 1024]
        outputs = []
        for dim in self.context_embeddings:
            context = self.context_embeddings[dim].expand(image_feat.size(0), -1)
            fused = torch.cat([image_feat, context], dim=1)
            score = self.fusion_heads[dim](fused)
            outputs.append(score)
        return torch.cat(outputs, dim=1)  # [B, 7]

# Load model weights
@st.cache_resource
def load_model():
    model = MultiContextSwinRegressor(context_embeddings)
    model.load_state_dict(torch.load("swin_regressor_lambda01.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Define image transform
val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    image = image.convert("RGB")
    image = val_transform(image)
    return image.unsqueeze(0)  # [1, 3, 512, 512]

# Streamlit UI
st.title("CLASS 2.0 - Multi Image Processing")
uploaded_files = st.file_uploader("Upload landscape images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results_text = ""
    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Image {idx}: {uploaded_file.name}", use_column_width=True)

        image_tensor = preprocess_image(image)

    with torch.no_grad():
        predicted_scores = model(image_tensor)
        predicted_scores = predicted_scores.squeeze().numpy() * 6.0  # scale back to 0–6

        # Display results
        st.subheader(f"Predicted Scores for {uploaded_file.name}")
        for dim, score in zip(dimension_names, predicted_scores):
            st.write(f"**{dim}**: {score:.2f}")

        # Append to copy-paste text
        results_text += f"\nImage {idx}: {uploaded_file.name}\n"
        for dim, score in zip(dimension_names, predicted_scores):
            results_text += f"{dim}: {score:.2f}\n"
        results_text += "-" * 40 + "\n"

    # Show copy-paste block
    st.text_area("Copy-Paste Results", results_text, height=400)
