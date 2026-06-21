from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from sentence_transformers import SentenceTransformer

from documents import Document
from embeddings.embedding_model import Distance, EmbeddingModel, EmbeddingSpec


class HuggingFaceEmbeddings(BaseModel, EmbeddingModel):
    """HuggingFace sentence_transformers embedding models.

    To use, you should have the `sentence_transformers` python package installed.
    """

    model_name: str = Field(default="Qwen/Qwen3-VL-Embedding-2B")
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    query_encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    multi_process: bool = False
    show_progress: bool = False

    dim: int
    distance: Distance

    _spec: EmbeddingSpec = PrivateAttr()
    _client: SentenceTransformer = PrivateAttr()

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )

    def model_post_init(self, __context: Any) -> None:
        self._client = SentenceTransformer(
            self.model_name,
            **self.model_kwargs,
        )

        # if (self_dim := self._client.get_embedding_dimension()) is not None:
        #     if self.dim != self_dim:
        #         warnings.warn(
        #             f"Input embedding dim {self.dim} was redefined to {self_dim}."
        #         )
        #     self.dim = self_dim

        self._spec = EmbeddingSpec(
            dim=self.dim,
            distance=self.distance,
        )

    def _embed(
        self,
        inputs: list[dict],
        encode_kwargs: dict[str, Any],
    ) -> list[list[float]]:
        if self.multi_process:
            pool = self._client.start_multi_process_pool()
            embeddings = self._client.encode(
                inputs,
                pool,
                **encode_kwargs,
            )
            SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self._client.encode(
                inputs,
                show_progress_bar=self.show_progress,
                **encode_kwargs,
            )

        if isinstance(embeddings, list):
            msg = (
                "Expected embeddings to be a Tensor or a numpy array, "
                "got a list instead."
            )
            raise TypeError(msg)

        return embeddings.tolist()

    def embed_documents(
        self,
        docs: list[Document],
    ) -> list[list[float]]:
        inputs = [doc.data for doc in docs]
        return self._embed(inputs, self.encode_kwargs)

    def embed_query(
        self,
        query: dict,
    ) -> list[float]:
        embed_kwargs = (
            self.query_encode_kwargs
            if len(self.query_encode_kwargs) > 0
            else self.encode_kwargs
        )
        return self._embed([query], embed_kwargs)[0]

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec
