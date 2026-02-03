import torch
from typing import Dict, List, Optional, Tuple, Union
from tinyrag.embedding.base_emb import BaseEmbedding
from sentence_transformers import SentenceTransformer, util

class HFSTEmbedding(BaseEmbedding):
    """
    class for Hugging face sentence embeddings
    """
    def __init__(self, path: str, is_api: bool = False, device: str = "") -> None:
        super().__init__(path, is_api)
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.st_model = SentenceTransformer(path, device=device)
        if str(device).lower().startswith("cuda"):
            try:
                self.st_model.half()
            except Exception:
                pass
        self.name = "hf_model"

    def get_embedding(self, text: str) -> List[float]:
        st_embedding = self.st_model.encode([text], normalize_embeddings=True)
        return st_embedding[0].tolist()

    def get_embeddings(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        st_embeddings = self.st_model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return st_embeddings.tolist()
