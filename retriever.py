# from langchain_ollama import ChatOllama
from operator import itemgetter
from typing import Any, Union

from langchain.chat_models import BaseChatModel
from langchain_core.documents import Document
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_core.vectorstores.base import VectorStore, VectorStoreRetriever
from pydantic import BaseModel

from llm_factory import create_llm
from settings import load_prompts
from utils import rrf_langchain_docs
from vector_store.vector_store_factory import create_vector_store
from embeddings.embeding_factory import create_embeddings


def init_vector_store_from_cfg(cfg: dict) -> VectorStore:

    embedding_container = create_embeddings(
        key="huggingface",
        cfg=cfg["embeddings"]["huggingface"]["bge_m3"],
    )

    vector_store = create_vector_store(
        key="qdrant.local",
        cfg=cfg["vector_store"]["qdrant_local"],
        embedding_container=embedding_container,
    )

    return vector_store


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


def build_multi_query_chain(
    llm: BaseChatModel, multi_query_template: str
) -> RunnableSerializable[dict[str, Any], list[str]]:

    class QueryVariants(BaseModel):
        variants: list[str]

    def merge_queries(data: dict) -> list[str]:
        return [data["query"]] + data["new_queries"].variants

    multi_query_prompt = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template(template=multi_query_template),
            HumanMessagePromptTemplate.from_template(template="User request: {query}"),
        ],
        template_format="f-string",
    )

    llm_with_so = llm.with_structured_output(QueryVariants)

    multi_query_chain = RunnablePassthrough.assign(
        new_queries=multi_query_prompt | llm_with_so
    ) | RunnableLambda(merge_queries)

    return multi_query_chain


def build_retriever_chain(
    retriever: VectorStoreRetriever, top_k: int
) -> RunnableSerializable[list[str], list[Document]]:

    async def process_batch(queries: list[str]) -> list[list[Document]]:
        return await retriever.abatch(queries)

    def rrf_langchain_docs_adapter(
        ranked_search_results: list[list[Document]],
    ) -> list[tuple[Document, float]]:
        docs, _ = zip(*rrf_langchain_docs(ranked_search_results, k=60))
        return list(docs)

    def top_k_adapter(docs: list[Document]) -> list[Document]:
        return docs[:top_k]

    retriever_chain = (
        RunnableLambda(process_batch)
        | RunnableLambda(rrf_langchain_docs_adapter)
        | RunnableLambda(top_k_adapter)
    )

    return retriever_chain


class RAGService:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        llm_provider = "openai"

        self.llm = create_llm(llm_provider, cfg["llm"][llm_provider])

        self.prompt_db = load_prompts()

        self.retriever = init_vector_store_from_cfg(cfg).as_retriever(
            k=self.cfg["retriever"]["vector_store_top_k"]
        )

        multi_query_chain = build_multi_query_chain(
            self.llm,
            self.prompt_db["multi_query_prompt"].format(
                num_queries=self.cfg["retriever"]["num_extra_queries"],
            ),
        )

        retriever_chain = build_retriever_chain(
            retriever=self.retriever, top_k=self.cfg["retriever"]["top_k"]
        )

        chat_prompt = build_base_prompt_template(self.prompt_db["system_prompt"])

        self.rag_chain = (
            {
                "query": RunnablePassthrough(),
                "docs": multi_query_chain | retriever_chain,
            }
            | RunnablePassthrough.assign(
                context=lambda x: build_context_from_docs(x["docs"])
            )
            | {
                "answer": chat_prompt | self.llm,
                "docs": itemgetter("docs"),
            }
        )

    async def make_query_async(
        self,
        query: str,
        raw: bool = False,
    ) -> Union[dict, str]:
        response = await self.rag_chain.ainvoke({"query": query})
        return response if raw else response["answer"].content
