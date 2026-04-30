from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import dotenv
from settings import get_cfg
from retriever import init_rag_chain_from_cfg

dotenv.load_dotenv()
app_state = dict()


class UserRequest(BaseModel):
    query: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["cfg"] = get_cfg()
    app_state["chain"] = init_rag_chain_from_cfg(app_state["cfg"])
    yield
    app_state.clear()


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(req: UserRequest):
    result = await app_state["chain"].ainvoke({"query": req.query})

    return {"answer": result["answer"].content}


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
