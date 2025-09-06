from __future__ import annotations

from typing import List, Optional

import torch


class GGUFEmbedder:
    """Thin wrapper over llama-cpp-python to load a GGUF embedding model and compute embeddings.

    Requires `pip install llama-cpp-python` and a GGUF embedding model.
    """

    def __init__(
        self,
        model_path: str,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        embedding: bool = True,
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise ImportError(
                "llama-cpp-python is required to use GGUF embeddings. Install with `pip install llama-cpp-python`."
            ) from e

        if n_threads is None:
            import os

            n_threads = max(1, os.cpu_count() or 1)

        # Initialize Llama in embedding mode
        self._llama = Llama(
            model_path=model_path,
            embedding=embedding,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            out = self.embed(["test"])
            self._dim = out.shape[-1]
        return self._dim

    def embed(self, texts: List[str]) -> torch.Tensor:
        """Return embeddings as a CPU float32 tensor of shape [B, D]."""
        embs: List[List[float]] = []
        for t in texts:
            r = self._llama.embed(t)
            # llama-cpp-python returns dict with 'embedding'
            vec = r["data"][0]["embedding"] if isinstance(r, dict) else r  # compatibility
            embs.append(vec)
        return torch.tensor(embs, dtype=torch.float32)
