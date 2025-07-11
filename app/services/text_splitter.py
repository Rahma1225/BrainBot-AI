import re

def split_into_sections(text: str, max_length: int = 1000) -> list:
    sections = re.split(r"\n\s*\n", text)  # split on empty lines
    merged = []
    buffer = ""
    for section in sections:
        if len(buffer) + len(section) < max_length:
            buffer += section + "\n\n"
        else:
            merged.append(buffer.strip())
            buffer = section
    if buffer:
        merged.append(buffer.strip())
    return merged
