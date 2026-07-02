import fitz

def extract_page_elements(page: fitz.Page):
    elements = []
    blocks = page.get_text("blocks", sort=True)
    for block in blocks:
        x0, y0, x1, y1, content, block_no, block_type = block
        if block_type == 0 and content.strip():
            elements.append({"type": "text", "y": y0, "x": x0, "text": content.strip()})
    elements.sort(key=lambda e: (e["y"], e["x"]))
    return elements

def process_pdf_stream(pdf_bytes: bytes) -> list[str]:
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    output_lines: list[str] = []
    
    for page in doc:
        elements = extract_page_elements(page)
        if not elements:
            continue
        page_content = [e["text"] for e in elements if e["type"] == "text"]
        output_lines.append("\n".join(page_content))
    doc.close()
    return output_lines

def chunking_pdf(text: list[str], chunk_size: int = 3) -> list[list[str]]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

