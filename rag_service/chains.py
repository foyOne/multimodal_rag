from collections import defaultdict
from operator import itemgetter
from typing import Any

from langchain.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
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
from pydantic import BaseModel

from documents import Document, RankedDocument
from rag_service.multi_source_retriever import MultiSourceRetriever
from rag_service.utils import convert_docs_to_content_repr, generate_id_from_doc
from scores.general_rrf_scorer import GeneralRRFScorer


def build_multi_query_chain(
    llm: BaseChatModel,
    multi_query_template: str,
) -> RunnableSerializable[dict[str, Any], list[dict]]:

    class QueryVariants(BaseModel):
        variants: list[str]

    def merge_queries(data: dict) -> list[dict]:
        queries = [data["text"]] + data["new_queries"].variants
        return [dict(text=text) for text in queries]

    multi_query_prompt = ChatPromptTemplate.from_messages(
        messages=[
            SystemMessagePromptTemplate.from_template(template=multi_query_template),
            HumanMessagePromptTemplate.from_template(template="User request: {text}"),
        ],
        template_format="f-string",
    )

    llm_with_so = llm.with_structured_output(QueryVariants)

    multi_query_chain = RunnablePassthrough.assign(
        new_queries=multi_query_prompt | llm_with_so
    ) | RunnableLambda(merge_queries)

    return multi_query_chain


def build_multi_source_retriever_chain(
    retriever: MultiSourceRetriever,
    retriever_top_k: dict,
    final_top_k: int,
) -> RunnableSerializable[list[dict], list[Document]]:

    async def process_batch(
        queries: list[dict],
    ) -> list[dict[str, list[RankedDocument]]]:
        return await retriever.batch_async(queries, top_k=retriever_top_k)

    def flatten(
        multi_query_result: list[dict[str, list[RankedDocument]]],
    ) -> list[RankedDocument]:
        doc_bunch = []

        for query_result in multi_query_result:
            for ranked_docs in query_result.values():
                doc_bunch.extend(ranked_docs)

        return doc_bunch

    def score_docs(ranked_docs: list[RankedDocument]) -> list[RankedDocument]:
        doc_keys = list(map(generate_id_from_doc, [x.doc for x in ranked_docs]))

        scorer = GeneralRRFScorer()
        scorer.add_ranked_docs(doc_keys, ranked_docs)
        rrf_ranked_docs = scorer.score()
        return rrf_ranked_docs

    def select_top_k(ranked_docs: list[RankedDocument]) -> list[RankedDocument]:
        return ranked_docs[:final_top_k]

    def unpack(ranked_docs: list[RankedDocument]) -> list[Document]:
        return [x.doc for x in ranked_docs]

    retriever_chain = (
        RunnableLambda(process_batch)
        | RunnableLambda(flatten)
        | RunnableLambda(score_docs)
        | RunnableLambda(select_top_k)
        | RunnableLambda(unpack)
    )

    return retriever_chain


def create_simplification_chain() -> RunnableSerializable[
    list[Document],
    dict[str, list[Document]],
]:
    def simplify_docs(docs: list[Document]):
        pages = defaultdict(lambda: dict(image=None, texts=[]))

        for doc in docs:
            key = (doc.metadata["domain"], doc.metadata["doc"], doc.metadata["page"])
            match doc.metadata["type"]:
                case "text":
                    pages[key]["texts"].append(doc)
                case "image":
                    pages[key]["image"] = doc

        image_docs = []
        text_docs = []
        for page in pages.values():
            if page["image"] is None:
                text_docs.extend(page["texts"])
            else:
                image_docs.append(page["image"])

        return dict(image_docs=image_docs, text_docs=text_docs)

    return RunnableLambda(simplify_docs)


def create_message_builder_chain(
    system_prompt: str,
) -> RunnableSerializable[dict, ChatPromptTemplate]:
    def build_messsges(data: dict) -> ChatPromptTemplate:
        query_text = data["query"]["text"]
        content = data["content"]

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
            HumanMessage(content=f"User request: {query_text}"),
        ]

    chain = {
        "query": itemgetter("query"),
        "content": itemgetter("docs") | RunnableLambda(convert_docs_to_content_repr),
    } | RunnableLambda(build_messsges)

    return chain
