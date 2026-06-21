from __future__ import annotations

from abc import ABC, abstractmethod

from documents import Document, RankedDocument
from embeddings.embedding_model import EmbeddingModel


class AsyncCollectionWithEmbeddings(ABC):
    @abstractmethod
    async def add_documents(
        self,
        docs: list[Document],
    ): ...

    @abstractmethod
    async def query(
        self,
        query: dict,
        top_k: int = 10,
    ) -> list[RankedDocument]: ...

    @property
    @abstractmethod
    def collection(self) -> AsyncCollection: ...

    @property
    @abstractmethod
    def embeddings(self) -> EmbeddingModel: ...


class AsyncCollection(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def add_documents(
        self,
        embeddings: list[list[float]],
        docs: list[Document],
    ) -> bool: ...

    @abstractmethod
    async def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[RankedDocument]: ...

    @abstractmethod
    def bind(
        self,
        embeddings: EmbeddingModel,
    ) -> AsyncCollectionWithEmbeddings: ...
