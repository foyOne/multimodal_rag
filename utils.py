import hashlib
import json
from collections import defaultdict
from itertools import chain

from operator import itemgetter
from typing import Hashable

from langchain_core.documents import Document


def stable_doc_id(doc: Document) -> str:
    normalized = {
        "content": doc.page_content.strip().lower(),
        "metadata": {k: doc.metadata[k] for k in sorted(doc.metadata)},
    }

    raw = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def reciprocal_rank_fusion(
    ranked_search_results: list[list[Hashable]],
    k: float = 60,
) -> list[tuple[Hashable, float]]:
    scores = defaultdict(float)
    for ranked_query_result in ranked_search_results:
        for rank, doc_key in enumerate(ranked_query_result, start=1):
            scores[doc_key] += 1 / (k + rank)

    return sorted(scores.items(), key=itemgetter(1), reverse=True)


def rrf_langchain_docs(
    ranked_search_results: list[list[Document]],
    k: float = 60,
) -> list[tuple[Document, float]]:
    ids = [[stable_doc_id(doc) for doc in docs] for docs in ranked_search_results]
    doc_map = dict(zip(chain(*ids), chain(*ranked_search_results)))
    fused = reciprocal_rank_fusion(ranked_search_results=ids, k=k)

    doc_ids, scores = zip(*fused)
    return list(
        zip(
            itemgetter(*doc_ids)(doc_map),
            scores,
        )
    )
