from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings

from pydantic import BaseModel, ConfigDict


class EmbeddingModelMetadata(BaseModel):
    dim: int
    model_config = ConfigDict(extra="allow")


class EmbeddingModelContainer(BaseModel):
    model: Embeddings
    metadata: EmbeddingModelMetadata

    model_config = ConfigDict(arbitrary_types_allowed=True)


def init_hf_embeddings(
    model_name: str,
    device: str,
    normalize_embeddings: bool,
    dim: int,
) -> EmbeddingModelContainer:
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": normalize_embeddings}

    hf_emdeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder="./data/tmp/hf_cache"
    )

    embedding_dim = hf_emdeddings._client.get_embedding_dimension()
    if embedding_dim is not None:
        dim = embedding_dim

    return EmbeddingModelContainer(
        model=hf_emdeddings,
        metadata=EmbeddingModelMetadata(
            dim=dim,
            **model_kwargs,
            **encode_kwargs,
        ),
    )


def init_fake_embeddings(dim: int) -> EmbeddingModelContainer:
    return EmbeddingModelContainer(
        model=FakeEmbeddings(size=dim),
        metadata=EmbeddingModelMetadata(dim=dim),
    )


def init_openai_embeddings(
    model: str,
    dim: int,
) -> EmbeddingModelContainer:
    embedding_model = OpenAIEmbeddings(model=model)

    expected_dim = embedding_model.dimensions
    if expected_dim is not None:
        dim = expected_dim

    return EmbeddingModelContainer(
        model=embedding_model, metadata=EmbeddingModelMetadata(dim=dim)
    )


def create_embeddings(key: str, cfg: dict) -> EmbeddingModelContainer:
    match key:
        case "huggingface":
            return init_hf_embeddings(**cfg)
        case "openai":
            return init_openai_embeddings(**cfg)
        case _:
            raise Exception(f"The specified type of embeddings '{key}' is missing.")
