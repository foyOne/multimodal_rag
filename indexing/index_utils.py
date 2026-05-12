from typing import Callable, Iterable, Iterator
import copy

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
)

from pathlib import Path
from langchain_core.documents import Document

from .clean_text_functions import (
    normalize_whitespace,
    remove_page_numbers,
    join_lines_with_incorrect_endings,
    fix_hyphenation,
)


def init_simple_splitter(
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> Callable[[Iterable[Document]], list[Document]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
    )

    def wrapper(docs: Iterable[Document]) -> list[Document]:
        return splitter.split_documents(docs)

    return wrapper


def clean_doc(doc: Document) -> Document:
    text = normalize_whitespace(doc.page_content)
    sections = text.split("\n\n")

    for idx, _ in enumerate(sections):
        sections[idx] = fix_hyphenation(sections[idx])
        sections[idx] = join_lines_with_incorrect_endings(sections[idx])
        sections[idx] = remove_page_numbers(sections[idx])
        sections[idx] = sections[idx].strip()

    return Document(
        page_content="\n\n".join(sections),
        metadata=doc.metadata,
    )


def init_doc_iterator_from_pdf_dir(pdf_dir: Path) -> Iterator[Document]:
    loader = DirectoryLoader(
        pdf_dir,
        loader_cls=PyPDFLoader,
        show_progress=True,
        silent_errors=True,
    )
    doc_iterator = loader.lazy_load()
    return doc_iterator


def filter_metadata(doc: Document, target_keys: list[str], save: bool) -> Document:
    target_keys = set(target_keys)
    keys = set(doc.metadata.keys())
    filtered_keys = (
        (keys.intersection(target_keys)) if save else (keys.difference(target_keys))
    )

    extra_metadata = copy.deepcopy(doc.metadata)
    for k in list(extra_metadata.keys()):
        if k not in filtered_keys:
            extra_metadata.pop(k)

    return Document(
        page_content=doc.page_content,
        metadata=extra_metadata,
    )


def transform_metadata(doc: Document) -> Document:
    if (source := doc.metadata.get("source", None)) is not None:
        source = Path(source)
        return Document(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "custom.source": source.name,
                "custom.domain": source.parent.name,
            },
        )
    else:
        return doc


# def create_index(cfg: dict):
#     cfg = get_cfg()

#     pdf_dir = Path("./data/pdfs/education/")
#     doc_iterator = init_doc_iterator_from_pdf_dir(pdf_dir)

#     splitter = init_simple_splitter(**cfg["chunking"]["baseline"])

#     embedding_container = create_embeddings(
#         "huggingface",
#         cfg["embeddings"]["huggingface"]["bge_m3"],
#     )

#     vector_store = create_vector_store(
#         key="qdrant.local",
#         cfg=cfg["vector_store"]["qdrant_local"],
#         embedding_container=embedding_container,
#     )

#     filter_metadata_adapter = functools.partial(
#         filter_metadata,
#         target_keys=[
#             "author",
#             "title",
#             "page",
#             "custom.source",
#             "custom.domain",
#         ],
#         save=True,
#     )

#     for batch in itertools.batched(doc_iterator, cfg["indexing"]["batch_size"]):
#         batch = map(clean_doc, batch)
#         batch = map(transform_metadata, batch)
#         batch = map(filter_metadata_adapter, batch)
#         batch = tuple(batch)

#         splitted_docs = splitter(batch)
#         vector_store.add_documents(splitted_docs)
#         break


# if __name__ == "__main__":
#     create_index()
