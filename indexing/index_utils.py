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
from langchain_core.documents import Document as LangChainDocument

from .clean_text_functions import (
    normalize_whitespace,
    remove_page_numbers,
    join_lines_with_incorrect_endings,
    fix_hyphenation,
)


def create_simple_splitter(
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> Callable[[Iterable[LangChainDocument]], list[LangChainDocument]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
    )

    def wrapper(docs: Iterable[LangChainDocument]) -> list[LangChainDocument]:
        return splitter.split_documents(docs)

    return wrapper


def clean_doc(doc: LangChainDocument) -> LangChainDocument:
    text = normalize_whitespace(doc.page_content)
    sections = text.split("\n\n")

    for idx, _ in enumerate(sections):
        sections[idx] = fix_hyphenation(sections[idx])
        sections[idx] = join_lines_with_incorrect_endings(sections[idx])
        sections[idx] = remove_page_numbers(sections[idx])
        sections[idx] = sections[idx].strip()

    return LangChainDocument(
        page_content="\n\n".join(sections),
        metadata=doc.metadata,
    )


def create_doc_iterator_from_pdf_dir(pdf_dir: Path) -> Iterator[LangChainDocument]:
    loader = DirectoryLoader(
        pdf_dir,
        loader_cls=PyPDFLoader,
        loader_kwargs=dict(mode="page"),
        show_progress=True,
        silent_errors=True,
    )
    doc_iterator = loader.lazy_load()
    return doc_iterator


def filter_metadata(doc: LangChainDocument, target_keys: list[str], save: bool) -> LangChainDocument:
    target_keys = set(target_keys)
    keys = set(doc.metadata.keys())
    filtered_keys = (
        (keys.intersection(target_keys)) if save else (keys.difference(target_keys))
    )

    extra_metadata = copy.deepcopy(doc.metadata)
    for k in list(extra_metadata.keys()):
        if k not in filtered_keys:
            extra_metadata.pop(k)

    return LangChainDocument(
        page_content=doc.page_content,
        metadata=extra_metadata,
    )


def transform_metadata(doc: LangChainDocument) -> LangChainDocument:
    if (source := doc.metadata.get("source", None)) is not None:
        source = Path(source)
        return LangChainDocument(
            page_content=doc.page_content,
            metadata={
                **doc.metadata,
                "doc": source.stem,
                "domain": source.parent.name,
                "type": "text",
            },
        )
    else:
        return doc
