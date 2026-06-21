import asyncio
import functools
import itertools
import os
from pathlib import Path

import dotenv
from langchain_core.documents import Document as LangChainDocument

from documents import Document
from embeddings.embedding_factory import create_embeddings
from embeddings.embedding_model import Distance
from indexing.index_utils import (
    clean_doc,
    create_doc_iterator_from_pdf_dir,
    create_simple_splitter,
    filter_metadata,
    transform_metadata,
)
from vector_store.vector_store_factory import create_vector_store

dotenv.load_dotenv()


def langchain_doc_to_local_doc(doc: LangChainDocument) -> Document:
    return Document(
        id=doc.id,
        data=dict(text=doc.page_content),
        metadata={
            **doc.metadata,
            "text": doc.page_content,
        },
    )


async def main():
    embeddings = create_embeddings(
        key="huggingface",
        cfg=dict(
            model_name="BAAI/bge-m3",
            model_kwargs=dict(device="cuda:0"),
            encode_kwargs=dict(normalize_embeddings=True),
            query_encode_kwargs=dict(),
            dim=1024,
            distance=Distance("cosine"),
        ),
    )

    collection = await create_vector_store(
        key="qdrant",
        cfg=dict(
            # connection_kwargs=dict(path="./data/tmp/qdrant_local"),
            connection_kwargs=dict(
                url="http://localhost:6333",
                api_key=os.getenv("QDRANT_API_KEY"),
            ),
            collection_name="texts",
        ),
        embeddings=embeddings,
    )

    pdf_dir = Path("./data/test/docs")
    doc_iterator = create_doc_iterator_from_pdf_dir(pdf_dir)

    chunk_size = 1000
    chunk_overlap = 200
    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    splitter = create_simple_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    filter_metadata_adapter = functools.partial(
        filter_metadata,
        target_keys=[
            "author",
            "title",
            "page",
            "doc",
            "domain",
            "type",
        ],
        save=True,
    )

    for batch in itertools.batched(doc_iterator, 8):
        batch = map(clean_doc, batch)
        batch = map(transform_metadata, batch)
        batch = map(filter_metadata_adapter, batch)
        batch = tuple(batch)

        splitted_docs = splitter(batch)
        local_docs = list(map(langchain_doc_to_local_doc, splitted_docs))

        await collection.add_documents(local_docs)


if __name__ == "__main__":
    asyncio.run(main())
