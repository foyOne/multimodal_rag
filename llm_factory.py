from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel


def create_llm(provider: str, cfg: dict) -> BaseChatModel:

    match provider:
        case "openai":
            return ChatOpenAI(**cfg)
        case _:
            raise Exception("There should be an error message here")
