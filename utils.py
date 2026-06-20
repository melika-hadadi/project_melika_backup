import re

def normalize(text: str) -> str:
    if not text:
        return ""

    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def highlight_query(text, query):
    if not text:
        return text

    words = query.split()

    for w in words:
        text = re.sub(
            f"({w})",
            r"<mark style='background:#fde68a;padding:2px 4px;border-radius:4px'>\1</mark>",
            text,
            flags=re.IGNORECASE,
        )

    return text


def extract_lines(text, max_lines=4):

    if not text:
        return ""

    lines = text.split("\n")
    lines = [l.strip() for l in lines if l.strip()]

    return "\n".join(lines[:max_lines])
