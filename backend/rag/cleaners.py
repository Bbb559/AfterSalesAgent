import re


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = []
    for line in text.split("\n"):
        line = line.strip()

        if not line:
            lines.append("")
            continue

        # 保留短行，避免误删表格、编号、选项、单位
        lines.append(line)

    return "\n".join(lines).strip()


def clean_pages(pages):
    cleaned_pages = []

    for page in pages:
        cleaned = clean_text(page.get("text", ""))

        if cleaned:
            new_page = dict(page)
            new_page["text"] = cleaned
            cleaned_pages.append(new_page)

    return cleaned_pages