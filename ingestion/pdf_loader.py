import pdfplumber
import os


def load_pdf(filepath):
    """
    Extracts text from a PDF, page by page.
    Returns a list of dicts: [{"page_number": 1, "text": "..."}, ...]

    We keep page numbers attached to each chunk of text because
    later, when we cite sources, we need to tell the user
    "this answer came from page 12" — not just "somewhere in the doc".
    """
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:  # some pages (cover pages, images) may return None
                pages.append({
                    "page_number": i + 1,
                    "text": text,
                    "source_file": os.path.basename(filepath)
                })
    return pages


def load_all_pdfs(directory):
    """
    Loads every PDF in a directory and returns one combined list
    of page dicts, tagged with which file each page came from.
    """
    all_pages = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory, filename)
            print(f"Loading {filename}...")
            pages = load_pdf(filepath)
            all_pages.extend(pages)
    return all_pages


if __name__ == "__main__":
    # Quick manual test — run this file directly to sanity-check extraction
    pages = load_all_pdfs("data/raw_pdfs")
    print(f"Loaded {len(pages)} total pages.")
    if pages:
        print("--- Sample from first page ---")
        print(pages[0]["text"][:500])