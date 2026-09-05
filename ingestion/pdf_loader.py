import pdfplumber
import os
import re


def clean_text(text):
    """
    Strips Chrome's print-to-PDF artifacts:
    - Header timestamp line, e.g. "03/09/2026, 20:15 aapl-20240928"
    - Footer URL + page counter, e.g. "https://www.sec.gov/.../aapl-20240928.htm 3/64"

    These are print artifacts, not document content, and add noise
    to every single chunk if left in — worth stripping at the source
    rather than downstream in the chunker.
    """
    # Remove header: date/time + filename pattern at start of a line
    text = re.sub(r"^\d{2}/\d{2}/\d{4}, \d{2}:\d{2} \S+\s*", "", text)

    # Remove footer: full URL followed by "N/M" page counter
    text = re.sub(r"https?://\S+\s+\d+/\d+\s*$", "", text)

    return text.strip()


def load_pdf(filepath):
    """
    Extracts text from a PDF, page by page.
    Returns a list of dicts: [{"page_number": 1, "text": "..."}, ...]
    """
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                text = clean_text(text)
                pages.append({
                    "page_number": i + 1,
                    "text": text,
                    "source_file": os.path.basename(filepath)
                })
    return pages


def load_all_pdfs(directory):
    all_pages = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            print(f"Loading {filename}...")
            pages = load_pdf(filepath)
            all_pages.extend(pages)
    return all_pages


if __name__ == "__main__":
    import pathlib
    project_root = pathlib.Path(__file__).resolve().parent.parent
    pdf_dir = project_root / "data" / "raw_pdfs"

    pages = load_all_pdfs(str(pdf_dir))
    print(f"Loaded {len(pages)} total pages.")
    if pages:
        print("--- Sample from first page ---")
        print(pages[0]["text"][:500])