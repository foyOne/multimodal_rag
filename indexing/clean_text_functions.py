import re


def fix_hyphenation(text: str) -> str:
    return re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)


def join_lines_with_incorrect_endings(text: str) -> str:
    lines = text.split("\n")
    new_lines = [lines[0].strip()]

    for i in range(1, len(lines)):
        prev_line = lines[i - 1]
        current_line = lines[i].strip()

        if len(prev_line) != 0 and not prev_line.strip().endswith((".", "!", "?", ":")):
            new_lines[-1] += " " + current_line
        else:
            new_lines.append(current_line)

    return "\n".join(new_lines)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", r" ", text)
    text = re.sub(r"\n\s+\n", r"\n\n", text)
    text = re.sub(r"\n{2,}", r"\n\n", text)
    return text.strip()


def remove_page_numbers(text: str) -> str:
    return re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
