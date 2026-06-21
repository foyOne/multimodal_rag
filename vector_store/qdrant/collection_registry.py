from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import VectorParams

from vector_store.qdrant.collections import QdrantAsyncCollection
from embeddings import EmbeddingModel
from vector_store.qdrant.utils import convert_to_qdrant_distance


class QdrantCollectionRegistry:
    def __init__(self, client: AsyncQdrantClient):
        self._client = client

    async def create_collection_from_embeddings(
        self,
        collection_name: str,
        embeddings: EmbeddingModel,
    ) -> QdrantAsyncCollection:
        if not await self._client.collection_exists(collection_name):
            created = await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embeddings.spec.dim,
                    distance=convert_to_qdrant_distance(embeddings.spec.distance),
                ),
            )

            if not created:
                raise Exception("Collection creation failed.")

        collection = QdrantAsyncCollection(
            client=self._client,
            collection_name=collection_name,
        )

        return collection
