# from langchain_ollama import ChatOllama
from operator import itemgetter
from pathlib import Path

import dotenv
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import ChatOpenAI

from embeddings import init_hf_embeddings
from settings import get_cfg
from vector_store import init_qdrant_local_vector_store


def init_retriever_from_cfg(cfg: dict) -> VectorStoreRetriever:
    embeddings = init_hf_embeddings(
        model_name=cfg["embeddings"]["bge_m3"]["model_name"],
        device=cfg["embeddings"]["bge_m3"]["device"],
        normalize_embeddings=cfg["embeddings"]["bge_m3"]["normalize_embeddings"],
    )

    vector_store = init_qdrant_local_vector_store(
        path=cfg["vectore_store"]["qdrant_local"]["path"],
        collection_name=cfg["vectore_store"]["qdrant_local"]["collection_name"],
        distance=cfg["vectore_store"]["qdrant_local"]["distance"],
        embeddings=embeddings,
        embedding_dim=cfg["embeddings"]["bge_m3"]["dim"],
    )

    return vector_store.as_retriever()


def build_base_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    template = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template(template=system_prompt),
            HumanMessagePromptTemplate.from_template(template="Context:\n{context}"),
            HumanMessagePromptTemplate.from_template(template="User request: {query}"),
        ],
        template_format="f-string",
    )
    return template


def build_one_chunk_data(doc: Document) -> str:
    parts = []

    if (source := doc.metadata.get("custom.source")) is not None:
        parts.append(f"File: {source}")

    if (page := doc.metadata.get("page")) is not None:
        parts.append(f"Page: {page}")

    if (title := doc.metadata.get("title")) is not None:
        parts.append(f"Title: {title}")

    parts.append(f"Content:\n{doc.page_content}")
    return "\n".join(parts)


def build_context_from_docs(docs: list[Document]) -> str:
    context_data = []
    for idx, doc in enumerate(docs, start=1):
        data = f"[Document {idx}]" + "\n" + build_one_chunk_data(doc)
        context_data.append(data)
    context = "\n\n".join(context_data)
    return context


def main():
    dotenv.load_dotenv()
    cfg = get_cfg()
    system_prompt = Path("./data/prompts/system_prompt.txt").read_text(encoding="utf-8")

    retriever = init_retriever_from_cfg(cfg)
    prompt = build_base_prompt_template(system_prompt=system_prompt)

    user_qeury = "What documents contain quotes from the book of the prophet Ezekiel?"

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.6,
    )

    chain = (
        {
            "context": itemgetter("query") | retriever | build_context_from_docs,
            "query": itemgetter("query"),
        }
        | prompt
        | llm
    )

    response = chain.invoke({"query": user_qeury})
    print(response.content)


if __name__ == "__main__":
    main()
