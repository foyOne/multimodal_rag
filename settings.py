from pathlib import Path
import tomllib
from typing import Any


def get_cfg() -> dict[str, Any]:
    config_file = Path("./config.toml")
    with open(config_file, "rb") as stream:
        cfg = tomllib.load(stream)
    return cfg


def load_prompts() -> dict[str, str]:
    prompt_dir = Path("./data/prompts")
    prompt_files = prompt_dir.glob("*.txt")
    prompt_db = {file.stem: file.read_text(encoding="utf-8") for file in prompt_files}
    return prompt_db
