import os
from contextlib import asynccontextmanager

import dotenv
from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from embeddings import Distance, create_embeddings
from llm_factory import create_llm
from rag_service.multi_source_retriever import MultiSourceRetriever
from rag_service.rag import RAG
from vector_store.qdrant.collections import QdrantAsyncCollection

dotenv.load_dotenv()


def init_retriever() -> MultiSourceRetriever:
    text_embeddings = create_embeddings(
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

    vision_embeddings = create_embeddings(
        key="huggingface",
        cfg=dict(
            model_name="Qwen/Qwen3-VL-Embedding-2B",
            model_kwargs=dict(
                device="cuda:0",
                processor_kwargs=dict(
                    min_pixels=256 * 256,
                    max_pixels=784 * 1024,
                ),
                truncate_dim=1024,
            ),
            encode_kwargs=dict(normalize_embeddings=True),
            query_encode_kwargs=dict(),
            dim=1024,
            distance=Distance("cosine"),
        ),
    )

    qdrant_client = AsyncQdrantClient(
        url="http://localhost:6333",
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    text_collection = QdrantAsyncCollection(
        client=qdrant_client, collection_name="texts"
    ).bind(text_embeddings)

    vision_collection = QdrantAsyncCollection(
        client=qdrant_client, collection_name="images"
    ).bind(vision_embeddings)

    retriever = MultiSourceRetriever(
        dict(
            text=text_collection,
            vision=vision_collection,
        )
    )

    return retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = create_llm(
        provider="openai",
        cfg=dict(
            model="gpt-4o-mini",
            temperature=0.6,
        ),
    )
    app.state.retriever = init_retriever()
    app.state.rag = RAG(app.state.llm, app.state.retriever)
    yield
    pass


def get_rag(request: Request) -> RAG:
    return request.app.state.rag


app = FastAPI(lifespan=lifespan)


class UserRequest(BaseModel):
    query: str


@app.post("/ask")
async def ask(
    user_request: UserRequest,
    rag: RAG = Depends(get_rag),
):
    response = await rag.ask_async(query=dict(text=user_request.query))

    return {"answer": response["answer"].content}
