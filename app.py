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
    model_path = "train.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}. Please upload it.")
        st.stop()

    checkpoint = torch.load(model_path, map_location="cpu")
    model = MultiContextSwinRegressor(context_embeddings)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    st.write(f"Model loaded from epoch: {checkpoint.get('epoch', 'Unknown')}")
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
st.title("CLASS 2.0")
uploaded_files = st.file_uploader("Upload landscape images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Prepare CSV header
    table_text = "Image Name," + ",".join(dimension_names) + "\n"

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_tensor = preprocess_image(image)

        with torch.no_grad():
            predicted_scores = model(image_tensor)
            predicted_scores = predicted.squeeze().numpy() * 6.0  # scale back to 0–6

        # Append row to table
        row = uploaded_file.name + "," + ",".join([f"{score:.2f}" for score in predicted_scores]) + "\n"
        table_text += row

    # Display copy-paste table
    st.subheader("Copy-Paste Table (Excel Friendly)")
    st.text_area("Results", table_text, height=400)

    # Add Copy button
    st.button("Copy Table to Clipboard", on_click=lambda: st.write("✅ Copy"))

    # Optional: Download button
    st.download_button(
        label="Download as CSV",
        data=table_text,
        file_name="predictions.csv",
        mime="text/csv"
    )
