import tiktoken


# tiktoken lets us count tokens the way an LLM actually sees them,
# not just characters — a "token" is roughly 4 characters of English text,
# but punctuation/numbers throw that estimate off, so we measure directly.
encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text):
    return len(encoding.encode(text))


def chunk_pages(pages, max_tokens=400, overlap_tokens=80):
    """
    Takes the output of pdf_loader.load_all_pdfs() and splits it into
    retrieval-sized chunks.

    Strategy: split each page's text into paragraphs, then greedily pack
    paragraphs into a chunk until adding the next paragraph would exceed
    max_tokens. This respects natural text boundaries instead of cutting
    mid-sentence.

    Overlap: the last `overlap_tokens` worth of text from a chunk is
    prepended to the next chunk, so context isn't lost at the boundary.

    Returns a list of dicts:
    [{"text": "...", "source_file": "...", "page_number": 3, "chunk_id": 0}, ...]
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        paragraphs = [p.strip() for p in page["text"].split("\n") if p.strip()]

        current_chunk_paragraphs = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)

            # If a single paragraph alone is already too big, split it further
            if para_tokens > max_tokens:
                # crude fallback: hard split by sentences on periods
                sentences = para.split(". ")
                for sentence in sentences:
                    sentence_tokens = count_tokens(sentence)
                    if current_tokens + sentence_tokens > max_tokens:
                        _flush_chunk(chunks, current_chunk_paragraphs, page, chunk_id)
                        chunk_id += 1
                        current_chunk_paragraphs = _get_overlap(current_chunk_paragraphs, overlap_tokens)
                        current_tokens = count_tokens(" ".join(current_chunk_paragraphs))
                    current_chunk_paragraphs.append(sentence)
                    current_tokens += sentence_tokens
                continue

            if current_tokens + para_tokens > max_tokens:
                _flush_chunk(chunks, current_chunk_paragraphs, page, chunk_id)
                chunk_id += 1
                current_chunk_paragraphs = _get_overlap(current_chunk_paragraphs, overlap_tokens)
                current_tokens = count_tokens(" ".join(current_chunk_paragraphs))

            current_chunk_paragraphs.append(para)
            current_tokens += para_tokens

        # flush whatever's left at the end of the page
        if current_chunk_paragraphs:
            _flush_chunk(chunks, current_chunk_paragraphs, page, chunk_id)
            chunk_id += 1

    return chunks


def _flush_chunk(chunks, paragraphs, page, chunk_id):
    if not paragraphs:
        return
    chunks.append({
        "text": " ".join(paragraphs),
        "source_file": page["source_file"],
        "page_number": page["page_number"],
        "chunk_id": chunk_id
    })


def _get_overlap(paragraphs, overlap_tokens):
    """Keep the tail-end paragraphs of the previous chunk, up to overlap_tokens,
    to seed the start of the next chunk."""
    overlap = []
    tokens_so_far = 0
    for para in reversed(paragraphs):
        t = count_tokens(para)
        if tokens_so_far + t > overlap_tokens:
            break
        overlap.insert(0, para)
        tokens_so_far += t
    return overlap


if __name__ == "__main__":
    import pathlib
    import sys

    project_root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from ingestion.pdf_loader import load_all_pdfs

    pdf_dir = project_root / "data" / "raw_pdfs"
    pages = load_all_pdfs(str(pdf_dir))
    chunks = chunk_pages(pages)

    print(f"Loaded {len(pages)} pages -> produced {len(chunks)} chunks.")
    print("--- Sample chunk ---")
    print(chunks[5])