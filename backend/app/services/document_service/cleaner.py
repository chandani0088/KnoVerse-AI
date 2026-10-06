import re


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text.

    Removes unnecessary whitespace and blank lines
    while preserving the actual document content.
    """

    if not text:
        return ""

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces at the beginning/end of each line
    text = "\n".join(line.strip() for line in text.splitlines())

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text