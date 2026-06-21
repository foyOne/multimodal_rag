from __future__ import annotations

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from documents import Document, RankedDocument
from embeddings import EmbeddingModel
from vector_store.common_collections import (
    AsyncCollection,
    AsyncCollectionWithEmbeddings,
)


class QdrantAsyncCollectionWithEmbeddings(AsyncCollectionWithEmbeddings):
    def __init__(
        self,
        collection: QdrantAsyncCollection,
        embeddings: EmbeddingModel,
    ):
        self._collection = collection
        self._embeddings = embeddings

    @property
    def collection(self) -> AsyncCollection:
        return self._collection

    @property
    def embeddings(self) -> EmbeddingModel:
        return self._embeddings

    async def add_documents(
        self,
        docs: list[Document],
    ):
        embeddings = await self._embeddings.embed_documents_async(docs=docs)
        return await self._collection.add_documents(embeddings, docs)

    async def query(
        self,
        query: dict,
        top_k: int = 10,
    ) -> list[RankedDocument]:
        query_embedding = await self._embeddings.embed_query_async(query)
        return await self._collection.query(
            query_embedding=query_embedding,
            top_k=top_k,
        )


class QdrantAsyncCollection(AsyncCollection):
    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
    ):
        self._client = client
        self._collection_name = collection_name

    @property
    def name(self) -> str:
        return self._collection_name

    async def add_documents(
        self,
        embeddings: list[list[float]],
        docs: list[Document],
    ):
        points = [
            PointStruct(
                id=doc.id or uuid.uuid4(),
                vector=embedding,
                payload={
                    # "data": doc.data,
                    "metadata": doc.metadata,
                },
            )
            for embedding, doc in zip(embeddings, docs)
        ]
        return await self._client.upsert(
            collection_name=self._collection_name,
            points=points,
        )

    async def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[RankedDocument]:
        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
        )

        ranked_docs = [
            RankedDocument(
                rank=rank,
                score=x.score,
                doc=Document(
                    id=x.id,
                    # data=x.payload["data"],
                    metadata=x.payload["metadata"],
                ),
            )
            for rank, x in enumerate(response.points, start=1)
        ]

        return ranked_docs

    def bind(self, embeddings: EmbeddingModel) -> QdrantAsyncCollectionWithEmbeddings:
        return QdrantAsyncCollectionWithEmbeddings(self, embeddings)
