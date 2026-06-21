from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from documents import Document
from embeddings.embedding_model import Distance, EmbeddingModel, EmbeddingSpec


class FakeEmbeddings(BaseModel, EmbeddingModel):
    dim: int
    distance: Distance

    _spec: EmbeddingSpec = PrivateAttr()

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )

    def model_post_init(self, __context: Any) -> None:
        self._spec = EmbeddingSpec(
            dim=self.dim,
            distance=self.distance,
        )

    def embed_documents(
        self,
        docs: list[Document],
    ) -> list[list[float]]:
        n = len(docs)
        return np.random.rand(n, self.spec.dim).astype(np.float32).tolist()

    def embed_query(
        self,
        query: dict,
    ) -> list[float]:
        return np.random.rand(self.spec.dim).astype(np.float32).tolist()

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec
