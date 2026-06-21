import asyncio
import os
from itertools import batched
from pathlib import Path

import dotenv
import tqdm

from documents import Document
from embeddings.embedding_factory import create_embeddings
from embeddings.embedding_model import Distance
from vector_store.vector_store_factory import create_vector_store

dotenv.load_dotenv()


def file_to_doc(file: Path) -> Document:
    domain = file.parent.parent.name
    doc = file.parent.name
    page = file.stem

    data = dict(image=str(file))
    metadata = dict(
        domain=file.parent.parent.name,
        doc=doc,
        page=int(page),
        path=(domain, doc, page),
        type="image",
    )

    return Document(
        data=data,
        metadata=metadata,
    )


async def main():
    embeddings = create_embeddings(
        key="huggingface",
        cfg=dict(
            model_name="Qwen/Qwen3-VL-Embedding-2B",
            model_kwargs=dict(
                device="cuda:0",
                processor_kwargs=dict(
                    min_pixels=256 * 256,
                    max_pixels=784 * 1024,
                ),
                truncate_dim=1024,  # reduce memory usage with matryoshka embeddings
            ),
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
            collection_name="images",
        ),
        embeddings=embeddings,
    )

    input_dir = Path("./data/test/docs_as_images")
    domain = "education"
    files = list(input_dir.joinpath(domain).rglob("*.jpg"))

    batch_size = 8
    it = batched(files, batch_size)
    num_butches = len(files) // batch_size

    for batch in tqdm.tqdm(it, total=num_butches):
        docs = tuple(map(file_to_doc, batch))

        await collection.add_documents(docs)


if __name__ == "__main__":
    asyncio.run(main())
