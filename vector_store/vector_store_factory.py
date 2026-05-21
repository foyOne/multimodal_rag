import os

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from embeddings.embeding_factory import EmbeddingModelContainer


def init_qdrant_remote_vector_store(
    url: str,
    collection_name: str,
    distance: Distance,
    embeddings: Embeddings,
    embedding_dim: int,
) -> QdrantVectorStore:
    client = QdrantClient(
        url=url,
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance(distance),
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    return vector_store


def init_qdrant_local_vector_store(
    path: str,
    collection_name: str,
    distance: Distance,
    embeddings: Embeddings,
    embedding_dim: int,
) -> QdrantVectorStore:
    client = QdrantClient(
        path=path,
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance(distance),
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

    return vector_store


def init_inmemory_vector_store(
    embeddings: Embeddings,
) -> VectorStore:
    return InMemoryVectorStore(embeddings)


def create_vector_store(
    key: str,
    cfg: dict,
    embedding_container: EmbeddingModelContainer,
) -> VectorStore:

    match key:
        case "qdrant.remote":
            return init_qdrant_remote_vector_store(
                **cfg,
                embeddings=embedding_container.model,
                embedding_dim=embedding_container.metadata.dim,
            )
        case "qdrant.local":
            return init_qdrant_local_vector_store(
                **cfg,
                embeddings=embedding_container.model,
                embedding_dim=embedding_container.metadata.dim,
            )
        case "inmemory":
            return init_inmemory_vector_store(embedding_container.model)
        case _:
            raise Exception(f"The specified type of vector store '{key}' is missing.")
