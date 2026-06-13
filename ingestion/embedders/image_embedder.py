"""
CLIP-based image embedder.
Produces 512-dim vectors suitable for cross-modal similarity search.
Text embeddings from the same CLIP encoder can be compared against image embeddings.
"""
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch


_MODEL_ID = "openai/clip-vit-base-patch32"


class ImageEmbedder:
    def __init__(self):
        self._model = CLIPModel.from_pretrained(_MODEL_ID)
        self._processor = CLIPProcessor.from_pretrained(_MODEL_ID)
        self._model.eval()

    def embed_image(self, image: Image.Image) -> list[float]:
        inputs = self._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().tolist()

    def embed_text(self, text: str) -> list[float]:
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().tolist()
