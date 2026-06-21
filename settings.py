from pathlib import Path
from functools import lru_cache

DATA_DIR = Path("./data")
PROMPT_DIR = Path("./data/prompts")
VISUAL_DOC_STORE_DIR = Path("./data/storage")


@lru_cache(maxsize=1)
def load_prompts() -> dict[str, str]:
    prompt_files = PROMPT_DIR.glob("*.txt")
    prompt_db = {file.stem: file.read_text(encoding="utf-8") for file in prompt_files}
    return prompt_db
