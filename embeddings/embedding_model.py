from enum import Enum
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from documents import Document
from utils.async_utils import run_as_async_function


class Distance(str, Enum):
    def __str__(self) -> str:
        return str(self.value)

    COSINE = "cosine"
    EUCLID = "euclid"
    DOT = "dot"
    MANHATTAN = "manhattan"


class EmbeddingSpec(BaseModel):
    dim: int
    distance: Distance

    model_config = ConfigDict(
        extra="allow",
        protected_namespaces=(),
    )


class EmbeddingModel(ABC):
    @property
    @abstractmethod
    def spec(self) -> EmbeddingSpec: ...

    @abstractmethod
    def embed_documents(
        self,
        docs: list[Document],
    ) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(
        self,
        query: dict,
    ) -> list[float]: ...

    async def embed_documents_async(
        self,
        docs: list[Document],
    ) -> list[list[float]]:
        return await run_as_async_function(
            self.embed_documents,
            docs,
        )

    async def embed_query_async(
        self,
        query: dict,
    ) -> list[float]:
        return await run_as_async_function(
            self.embed_query,
            query,
        )
