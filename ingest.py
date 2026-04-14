"""
ingest.py — Document loading, cleaning, and chunking.

Supports .pdf and .txt files. PDFs are cleaned of Gutenberg boilerplate
and browser-print artifacts. Text is chunked with RecursiveCharacterTextSplitter.
"""

import os
import re
import pandas as pd

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


# Loaders

def load_pdf(path: str) -> str:
    """
    Extract full text from a PDF using pymupdf.

    - Injects [Page N] markers so chunking can track page origins.
    - Strips Project Gutenberg header/footer boilerplate.
    - Removes browser-print URL artifacts (timestamp + URL lines).
    """
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append(f"[Page {i + 1}]\n{text.strip()}")
    full_text = "\n\n".join(pages)

    # Strip Gutenberg header/footer (title varies between books)
    start_match = re.search(
        r"\*{3} START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*{3}", full_text
    )
    end_match = re.search(
        r"\*{3} END OF THE PROJECT GUTENBERG EBOOK[^\n]*\*{3}", full_text
    )

    if start_match:
        full_text = full_text[start_match.end():]
        print(f"  Stripped Gutenberg header: '{start_match.group().strip()}'")
    else:
        print("  No Gutenberg header found.")

    if end_match:
        end_match = re.search(
            r"\*{3} END OF THE PROJECT GUTENBERG EBOOK[^\n]*\*{3}", full_text
        )
        if end_match:
            full_text = full_text[: end_match.start()]
        print("  Stripped Gutenberg footer.")
    else:
        print("  No Gutenberg footer found.")

    # Remove PDF browser-print artifacts: "4/12/26, 5:48 PM gutenberg.org/..."
    full_text = re.sub(r"\d+/\d+/\d+,\s*\d+:\d+\s*[AP]M\s+\S+\n", "", full_text)
    full_text = re.sub(r"https?://\S+\s+\d+/\d+\s*\n", "", full_text)

    # Collapse excessive blank lines
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    clean_len = len(re.sub(r"\[Page \d+\]\n?", "", full_text))
    print(f"  {len(doc)} pages → {clean_len:,} characters after cleaning")
    doc.close()
    return full_text


def load_text_file(path: str) -> str:
    """Load a plain .txt file and normalise whitespace."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    print(f"  {os.path.basename(path)}: {len(text):,} characters")
    return text


# Chunking

def chunk_text(text: str, source_name: str) -> pd.DataFrame:
    """
    Recursively split text into ~200-word chunks with 15% overlap.

    [Page N] markers embedded in the text are extracted as metadata
    and stripped from the stored chunk text before embedding.

    Returns a DataFrame with columns:
        source, chunk_index, chunk_text, word_count, pages, strategy
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        is_separator_regex=False,
    )

    chunks = splitter.split_text(text)
    rows = []
    for j, chunk in enumerate(chunks):
        page_nums = re.findall(r"\[Page (\d+)\]", chunk)
        pages = sorted(set(int(p) for p in page_nums))
        clean_chunk = re.sub(r"\[Page \d+\]\n?", "", chunk).strip()
        rows.append(
            {
                "source":      source_name,
                "chunk_index": j,
                "chunk_text":  clean_chunk,
                "word_count":  len(clean_chunk.split()),
                "pages":       pages,
                "strategy":    f"recursive_{CHUNK_SIZE}_{CHUNK_OVERLAP}",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  Warning: no chunks generated for {source_name}")
        return df

    with_pages = (df["pages"].apply(len) > 0).sum()
    print(
        f"  {len(df)} chunks | "
        f"avg {df['word_count'].mean():.0f} words | "
        f"min {df['word_count'].min()} | "
        f"max {df['word_count'].max()} | "
        f"{with_pages}/{len(df)} chunks have page info"
    )
    return df


# Orchestrator

def load_and_chunk_all(docs_folder: str = DOCUMENTS_DIR):
    """
    Scan docs_folder for .pdf and .txt files, load, clean, and chunk each one.

    Returns:
        all_chunks (list[dict])  — flat list ready for ChromaDB upsert
        merged_df  (pd.DataFrame) — combined chunk DataFrame for inspection
    """
    os.makedirs(docs_folder, exist_ok=True)
    all_chunks = []
    tables = []

    files = [f for f in sorted(os.listdir(docs_folder)) if os.path.isfile(os.path.join(docs_folder, f))]
    supported = [f for f in files if os.path.splitext(f)[1].lower() in (".pdf", ".txt")]

    if not supported:
        print(f"No .pdf or .txt files found in {docs_folder}/")
        return [], pd.DataFrame()

    print(f"Found {len(supported)} file(s) in {docs_folder}/\n")

    for fname in supported:
        fpath = os.path.join(docs_folder, fname)
        ext = os.path.splitext(fname)[1].lower()
        print(f"Loading: {fname}")

        doc_text = load_pdf(fpath) if ext == ".pdf" else load_text_file(fpath)
        df = chunk_text(doc_text, fname)

        if df.empty:
            continue

        tables.append(df)
        for row in df.to_dict(orient="records"):
            pages = row.get("pages", [])
            all_chunks.append(
                {
                    "id":   f"{fname}_chunk_{int(row['chunk_index']):04d}",
                    "text": row["chunk_text"],
                    "metadata": {
                        "source":      fname,
                        "chunk_index": int(row["chunk_index"]),
                        "pages":       ",".join(str(p) for p in pages),
                        "strategy":    row["strategy"],
                        "word_count":  int(row["word_count"]),
                    },
                }
            )
        print()

    merged = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    print(f"Total: {len(all_chunks)} chunks across {len(tables)} document(s).")
    return all_chunks, merged
