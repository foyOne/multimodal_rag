from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
import uvicorn
import dotenv
from settings import get_cfg
from retriever import RAGService

dotenv.load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cfg = get_cfg()
    app.state.rag_service = RAGService(app.state.cfg)
    yield
    pass


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


app = FastAPI(lifespan=lifespan)


class UserRequest(BaseModel):
    query: str


@app.post("/ask")
async def ask(
    user_request: UserRequest,
    rag: RAGService = Depends(get_rag_service),
):
    answer = await rag.make_query_async(query=user_request.query, raw=False)

    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
