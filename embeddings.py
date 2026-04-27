from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import FakeEmbeddings


def init_hf_embeddings(
    model_name: str,
    device: str,
    normalize_embeddings: bool,
) -> HuggingFaceEmbeddings:
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": normalize_embeddings}

    hf_emdeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return hf_emdeddings


def init_fake_embeddings(dim: int) -> FakeEmbeddings:
    return FakeEmbeddings(size=dim)
