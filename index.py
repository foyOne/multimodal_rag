import asyncio
import functools
import itertools
from pathlib import Path

import dotenv

from embeddings.embeding_factory import create_embeddings
from indexing.index_utils import (
    clean_doc,
    filter_metadata,
    init_doc_iterator_from_pdf_dir,
    init_simple_splitter,
    transform_metadata,
)
from settings import get_cfg
from vector_store.vector_store_factory import create_vector_store


async def main():
    dotenv.load_dotenv()
    cfg = get_cfg()

    pdf_dir = Path("./data/pdfs/education/")
    doc_iterator = init_doc_iterator_from_pdf_dir(pdf_dir)

    splitter = init_simple_splitter(**cfg["chunking"]["baseline"])

    embedding_container = create_embeddings(
        "huggingface",
        cfg["embeddings"]["huggingface"]["bge_m3"],
    )

    vector_store = create_vector_store(
        key="qdrant.remote",
        cfg=cfg["vector_store"]["qdrant_remote"],
        embedding_container=embedding_container,
    )

    filter_metadata_adapter = functools.partial(
        filter_metadata,
        target_keys=[
            "author",
            "title",
            "page",
            "custom.source",
            "custom.domain",
        ],
        save=True,
    )

    for batch in itertools.batched(doc_iterator, cfg["indexing"]["batch_size"]):
        batch = map(clean_doc, batch)
        batch = map(transform_metadata, batch)
        batch = map(filter_metadata_adapter, batch)
        batch = tuple(batch)

        splitted_docs = splitter(batch)
        await vector_store.aadd_documents(splitted_docs)


if __name__ == "__main__":
    asyncio.run(main())
