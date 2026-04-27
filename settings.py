import tomllib
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_cfg() -> dict[str, Any]:
    with open("config.toml", "rb") as stream:
        cfg = tomllib.load(stream)
    return cfg
