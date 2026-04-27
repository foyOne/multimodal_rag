from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore


def init_qdrant_vector_store(
    url: str,
    collection_name: str,
    distance: Distance,
    embeddings: Embeddings,
    embedding_dim: int,
) -> QdrantVectorStore:
    client = QdrantClient(url=url)

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
    client = QdrantClient(path=path)

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
