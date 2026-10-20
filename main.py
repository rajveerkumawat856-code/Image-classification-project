
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import torch

from PIL import Image, UnidentifiedImageError
from torchvision import models


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

EMBEDDINGS_PATH = BASE_DIR / "embeddings.npy"
METADATA_PATH = BASE_DIR / "embedded_image_metadata.csv"

st.set_page_config(
    page_title="AI Image Similarity Search",
    page_icon="🖼️",
    layout="wide"
)


# --------------------------------------------------
# 2. Load ResNet50 model only once
# --------------------------------------------------

@st.cache_resource
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(weights=weights)
    model = model.to(device)
    model.eval()

    preprocess = weights.transforms()

    return model, preprocess, device


# --------------------------------------------------
# 3. Load saved embeddings and metadata only once
# --------------------------------------------------

@st.cache_resource
def load_data():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    if "image_path" not in metadata.columns:
        raise ValueError(
            "The metadata CSV must contain an 'image_path' column."
        )

    if len(embeddings) != len(metadata):
        raise ValueError(
            "The number of embeddings does not match "
            "the number of metadata rows."
        )

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )

    embeddings = embeddings / np.maximum(norms, 1e-12)

    return embeddings, metadata


# --------------------------------------------------
# 4. Convert an uploaded image into an embedding
# --------------------------------------------------

def generate_embedding(image, model, preprocess, device):
    image = image.convert("RGB")

    image_tensor = preprocess(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)

    embedding = output.squeeze(0).cpu().numpy()

    # Normalize query embedding
    norm = np.linalg.norm(embedding)

    if norm < 1e-12:
        raise ValueError("Could not generate a valid embedding.")

    embedding = embedding / norm

    return embedding


# --------------------------------------------------
# 5. Find similar images
# --------------------------------------------------

def find_similar_images(
    query_embedding,
    embeddings,
    metadata,
    top_n
):
    if query_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError(
            f"Dimension mismatch: uploaded image produces "
            f"{query_embedding.shape[0]} values, but saved "
            f"embeddings contain {embeddings.shape[1]} values."
        )

    # Calculate cosine similarity with all saved embeddings
    scores = embeddings @ query_embedding

    # Get indices of highest similarity scores
    indices = np.argsort(scores)[::-1][:top_n]

    results = []

    for index in indices:
        row = metadata.iloc[int(index)]

        image_path = Path(str(row["image_path"]))

        # Resolve relative paths from the project folder
        if not image_path.is_absolute():
            image_path = BASE_DIR / image_path

        results.append({
            "image_path": image_path,
            "similarity_score": float(scores[index]),
            "metadata": row
        })

    return results


# --------------------------------------------------
# 6. Streamlit user interface
# --------------------------------------------------

st.title("🖼️ AI Image Similarity Search")

st.write(
    "Upload a product image to find visually similar "
    "images from the product dataset."
)

st.divider()

try:
    with st.spinner("Loading model and image database..."):
        model, preprocess, device = load_model()
        embeddings, metadata = load_data()

    st.success(
        f"Model ready | {len(metadata):,} images available"
    )

except Exception as error:
    st.error(f"Could not initialize the application: {error}")
    st.stop()


# Sidebar
st.sidebar.header("Search settings")

top_n = st.sidebar.slider(
    "Number of similar images",
    min_value=1,
    max_value=20,
    value=5
)

st.sidebar.write(f"Device: `{device}`")

st.sidebar.write(
    f"Database size: {len(metadata):,} images"
)


# Image upload
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)


if uploaded_file is not None:

    try:
        query_image = Image.open(
            BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file is not a valid image.")
        st.stop()

    st.subheader("Uploaded image")

    st.image(
        query_image,
        caption="Your query image",
        width=300
    )

    if st.button("Find Similar Images", type="primary"):

        try:
            with st.spinner("Searching for similar images..."):

                # Generate embedding for uploaded image
                query_embedding = generate_embedding(
                    query_image,
                    model,
                    preprocess,
                    device
                )

                # Search saved embeddings
                results = find_similar_images(
                    query_embedding,
                    embeddings,
                    metadata,
                    top_n
                )

            st.subheader("Similar Products")

            # Display matching images
            columns = st.columns(3)

            for i, result in enumerate(results):

                image_path = result["image_path"]
                score = result["similarity_score"]
                row = result["metadata"]

                with columns[i % 3]:

                    if image_path.is_file():
                        st.image(
                            str(image_path),
                            use_container_width=True
                        )
                    else:
                        st.warning(
                            f"Image file not found: {image_path}"
                        )

                    st.write(
                        f"**Similarity score:** {score:.4f}"
                    )

                    # Display product name if available
                    if "productDisplayName" in row.index:
                        st.caption(
                            str(row["productDisplayName"])
                        )

                    st.caption(
                        f"Result {i + 1}"
                    )

        except Exception as error:
            st.error(f"Search failed: {error}")

else:
    st.info(
        "Upload a product image above to start searching."
    )
