from operator import itemgetter
from typing import Hashable

from pydantic import BaseModel, Field

from documents import Document, RankedDocument


class GeneralRRFScorer(BaseModel):
    store: dict[Hashable, Document] = Field(default_factory=dict)
    ranked_docs_bunch: list[tuple[Hashable, int]] = Field(default_factory=list)
    k: float = 60.0

    def add_ranked_docs(
        self,
        doc_keys: list[Hashable],
        ranked_docs: list[RankedDocument],
    ) -> None:
        if len(doc_keys) != len(ranked_docs):
            raise Exception("Keys and Docs must be the same length.")

        ranks = (x.rank for x in ranked_docs)
        docs = (x.doc for x in ranked_docs)

        self.store.update(zip(doc_keys, docs))
        self.ranked_docs_bunch.extend(zip(doc_keys, ranks))

    def score(self) -> list[RankedDocument]:
        scores = dict.fromkeys(self.store.keys(), 0.0)

        for doc_key, rank in self.ranked_docs_bunch:
            scores[doc_key] += 1 / (rank + self.k)

        keys, rrf_scores = zip(*sorted(scores.items(), key=itemgetter(1), reverse=True))
        rrf_ranked_docs = itemgetter(*keys)(self.store)

        return [
            RankedDocument(rank=rank, score=score, doc=doc)
            for rank, (score, doc) in enumerate(
                zip(rrf_scores, rrf_ranked_docs),
                start=1,
            )
        ]

    def clear(self) -> None:
        self.store.clear()
        self.ranked_docs_bunch.clear()
