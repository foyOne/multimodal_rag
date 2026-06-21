from operator import itemgetter

from langchain.chat_models import BaseChatModel
from langchain_core.runnables import (
    RunnablePassthrough,
)

from settings import load_prompts

from rag_service.chains import (
    build_multi_query_chain,
    build_multi_source_retriever_chain,
    create_message_builder_chain,
    create_simplification_chain,
)
from rag_service.multi_source_retriever import MultiSourceRetriever


class RAG:
    def __init__(self, llm: BaseChatModel, retriever: MultiSourceRetriever):
        self.llm = llm
        self.retriever = retriever
        self.prompt_db = load_prompts()

        multi_query_chain = build_multi_query_chain(
            self.llm,
            self.prompt_db["multi_query_prompt"].format(
                num_queries=5,
            ),
        )

        retriever_chain = build_multi_source_retriever_chain(
            retriever=self.retriever,
            retriever_top_k=dict(text=20, vision=10),
            final_top_k=5,
        )

        simplification_chain = create_simplification_chain()
        message_builder_chain = create_message_builder_chain(
            self.prompt_db["system_prompt"]
        )

        self.rag_chain = (
            RunnablePassthrough()
            | {
                "query": itemgetter("query"),
                "docs": itemgetter("query")
                | multi_query_chain
                | retriever_chain
                | simplification_chain,
            }
            | {
                "query": itemgetter("query"),
                "docs": itemgetter("docs"),
                "answer": message_builder_chain | llm,
            }
        )

    async def ask_async(
        self,
        query: dict,
    ) -> dict:
        inputs = dict(query=query)
        response = await self.rag_chain.ainvoke(inputs)
        return response
