from documents import RankedDocument
from vector_store.common_collections import AsyncCollectionWithEmbeddings


class MultiSourceRetriever:
    def __init__(
        self,
        sources: dict[str, AsyncCollectionWithEmbeddings],
    ):
        self.sources = sources

    async def query_async(
        self,
        query: dict,
        top_k: dict,
    ) -> dict[str, list[RankedDocument]]:
        results = dict()
        for key, vs in self.sources.items():
            ranked_docs = await vs.query(query, top_k=top_k[key])
            results[key] = ranked_docs

        return results

    async def batch_async(
        self,
        queries: list[dict],
        top_k: dict,
    ) -> list[dict[str, list[RankedDocument]]]:
        results = []

        for query in queries:
            query_result = await self.query_async(query=query, top_k=top_k)
            results.append(query_result)

        return results
