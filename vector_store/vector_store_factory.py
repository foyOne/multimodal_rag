from qdrant_client import AsyncQdrantClient

from embeddings import EmbeddingModel
from vector_store.common_collections import AsyncCollectionWithEmbeddings
from vector_store.qdrant.collection_registry import QdrantCollectionRegistry
from vector_store.qdrant.collections import (
    QdrantAsyncCollectionWithEmbeddings,
)


async def create_qdrant_vector_store(
    connection_kwargs: dict,
    collection_name: dict,
    embeddings: EmbeddingModel,
) -> QdrantAsyncCollectionWithEmbeddings:
    client = AsyncQdrantClient(**connection_kwargs)
    registry = QdrantCollectionRegistry(client)

    return (
        await registry.create_collection_from_embeddings(
            collection_name=collection_name,
            embeddings=embeddings,
        )
    ).bind(embeddings)


async def create_vector_store(
    key: str,
    cfg: dict,
    embeddings: EmbeddingModel,
) -> AsyncCollectionWithEmbeddings:

    match key:
        case "qdrant":
            return await create_qdrant_vector_store(
                connection_kwargs=cfg["connection_kwargs"],
                collection_name=cfg["collection_name"],
                embeddings=embeddings,
            )
        case _:
            raise Exception(f"The specified type of vector store '{key}' is missing.")
