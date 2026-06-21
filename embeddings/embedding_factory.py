from embeddings.embedding_model import EmbeddingModel, Distance
from embeddings.hugging_face_embeddings import HuggingFaceEmbeddings
from embeddings.fake_embeddings import FakeEmbeddings


def create_hf_embeddings(
    model_name: str,
    model_kwargs: dict,
    encode_kwargs: dict,
    query_encode_kwargs: dict,
    dim: int,
    distance: str,
) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_encode_kwargs=query_encode_kwargs,
        dim=dim,
        distance=Distance(distance),
    )


def create_fake_embeddings(
    dim: int,
    distance: str,
) -> FakeEmbeddings:
    return FakeEmbeddings(
        dim=dim,
        distance=Distance(distance),
    )


def create_embeddings(
    key: str,
    cfg: dict,
) -> EmbeddingModel:
    match key:
        case "huggingface":
            return create_hf_embeddings(**cfg)
        case "fake":
            return create_fake_embeddings(**cfg)
        case _:
            raise Exception(f"The specified type of embeddings '{key}' is missing.")
