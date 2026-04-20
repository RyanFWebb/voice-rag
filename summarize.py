"""
summarize.py — Pre-computed hierarchical summaries via map-reduce.

Plain RAG fails on summarization queries ("summarize this document",
"what is this about?") because retrieval returns only the chunks most
similar to the query — never the full document. This module fixes that
by running map-reduce summarization once at ingestion time and caching
the result on disk, so summarization queries return a summary of the
whole document rather than a summary of three similar chunks.

Pipeline:
  1. Map:    group chunks into batches of SUMMARY_MAP_BATCH → one LLM
             call per batch produces a "section summary"
  2. Reduce: fold all section summaries into a single document summary
             (recursively, in case there are too many to fit in context)

Caches are keyed by (source filename, chunk-content hash). Re-ingesting
with the same chunks reuses the cached summary; changing the chunking
config invalidates the cache automatically.
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone

import llm
from config import (
    SUMMARY_CACHE_DIR,
    SUMMARY_MAP_BATCH,
    SUMMARY_REDUCE_BATCH,
    SUMMARY_MODEL,
    SUMMARY_CTX,
)


# Intent classification

# Patterns that indicate the user wants a whole-document summary rather than
# a specific factual question. Kept deliberately narrow to avoid routing
# questions like "summarize the fight scene" (which is specific) to the
# pre-computed summary path.
_SUMMARY_PATTERNS = [
    r"\bsummari[sz]e\b",
    r"\bsummary\b",
    r"\btl;?dr\b",
    r"\bgist\b",
    r"\boverview\b",
    r"\babstract\b",
    r"\bsynopsi[sz]\b",
    r"\bmain (points?|ideas?|themes?|takeaways?)\b",
    r"\bkey (points?|ideas?|themes?|takeaways?)\b",
    r"what(?:'s|s|\s+is|\s+was)\s+(this|the)\s+(document|book|text|story|pdf|file)\s+about",
    r"what (does|do) (this|these) (document|book|text|story|pdf|file|documents) (say|cover|describe)",
]

_SUMMARY_RE = re.compile("|".join(_SUMMARY_PATTERNS), re.IGNORECASE)


def is_summarization_query(question: str) -> bool:
    """Return True if the question looks like a whole-document summary request."""
    return bool(_SUMMARY_RE.search(question or ""))


# Small instruction-tuned models (gemma3:4b, llama3.2:3b, etc.) frequently
# ignore "do not begin with..." instructions and emit a preamble line like
# "Here's a summary of the document in 12-16 sentences:" before the real
# content. Strip the first line if it's clearly metadata about the response
# rather than content from the document.
_PREAMBLE_RE = re.compile(
    r"^\s*"
    r"(?:here(?:'?s| is| are| follows)?|this is|below (?:is|follows)|"
    r"a (?:summary|condensed summary|document summary)|"
    r"sure[,!]?|okay[,!]?|certainly[,!]?)"
    r"[^\n]*:\s*\n+",
    re.IGNORECASE,
)


def _strip_preamble(text: str) -> str:
    """Remove a leading metadata-style preamble line from an LLM response."""
    return _PREAMBLE_RE.sub("", text, count=1).strip()


# Document targeting for multi-doc summarization queries
#
# When more than one document is cached, a query like "summarize the hamlet book"
# should return only the Hamlet summary, not all of them. We score each source
# filename against the query's content tokens and pick the winner, falling back
# to "return all" when the query is generic ("summarize everything") or when no
# filename tokens match.

# Query phrasings that explicitly ask for every document (override targeting).
_ALL_DOCS_RE = re.compile(
    r"\b(all|every|each|both)\s+(books?|documents?|docs?|pdfs?|texts?|files?|stor(?:ies|y))\b",
    re.IGNORECASE,
)

# Tokens to drop from the query before comparing to filenames — they carry no
# identifying information about which document is being requested.
_QUERY_STOPWORDS = frozenset({
    "summarize", "summarise", "summary", "summaries",
    "overview", "gist", "tldr", "abstract", "synopsis",
    "main", "key", "points", "ideas", "themes", "takeaways",
    "what", "whats", "is", "was", "are", "were",
    "this", "that", "these", "those", "the",
    "book", "books", "document", "documents", "doc", "docs",
    "pdf", "pdfs", "text", "texts", "story", "stories", "file", "files",
    "about", "describe", "cover", "covers", "say", "says",
    "please", "give", "tell", "show",
    "and", "for", "with", "from", "into",
})

# Filename tokens that don't identify the document (format markers, language
# tags, common source-repo prefixes, etc.).
_SOURCE_STOPWORDS = frozenset({
    "gutenberg", "pdf", "txt", "epub", "doc", "docx", "html", "htm",
    "book", "document", "text", "story",
    "the", "and",
    "ebook", "ebooks", "vol", "volume", "part", "chapter",
})


def _tokenize(text: str, stopwords: frozenset[str]) -> set[str]:
    """Return lowercased alphabetic tokens >= 3 chars that aren't in stopwords."""
    return {t for t in re.findall(r"[A-Za-z]{3,}", text.lower()) if t not in stopwords}


def _source_tokens(source: str) -> set[str]:
    """Extract identifying tokens from a source filename (stem, separators split)."""
    stem = re.sub(r"\.(pdf|txt|md|epub|docx?|html?)$", "", source, flags=re.IGNORECASE)
    return _tokenize(stem, _SOURCE_STOPWORDS)


def _select_target(question: str, records: list[dict]) -> list[dict]:
    """
    Return the subset of cached summary records that the question refers to.

    Resolution order:
      1. If only one record exists, return it (nothing to pick).
      2. If the query explicitly asks for "all/every/each/both" docs, return all.
      3. Score each source's filename tokens against the query's content tokens.
         Clear winner (more matches than runner-up) → return just that record.
      4. Tie at the top score → return all tied records.
      5. No matches anywhere → return all (safe fallback; the LLM didn't name a doc).
    """
    if len(records) <= 1:
        return records

    if _ALL_DOCS_RE.search(question):
        return records

    q_tokens = _tokenize(question, _QUERY_STOPWORDS)
    if not q_tokens:
        return records

    scored = [(len(q_tokens & _source_tokens(r["source"])), r) for r in records]
    scored.sort(key=lambda x: -x[0])

    top_score = scored[0][0]
    if top_score == 0:
        return records  # query doesn't mention any filename token

    tied = [r for score, r in scored if score == top_score]
    return tied  # one element if clear winner, multiple if tied


# Prompt templates

_MAP_PROMPT = """\
Extract the factual content of this excerpt from "{source}" in exactly 3-4 sentences.

Requirements:
- Identify specific entities named in the text (people, places, concepts, systems,
  variables, quantities) and what the text says about them.
- Record concrete details — actions, events, decisions, claims, findings, mechanisms,
  definitions, figures — rather than generic themes or abstractions.
- Do not invent or generalize beyond what the excerpt actually states.
- Start directly with the content. Do NOT begin with "Here's a summary", "This excerpt",
  "In this passage", or any similar preamble.

Excerpt:
{text}

Summary:"""


_REDUCE_PROMPT = """\
Below are section summaries of "{source}", in document order. Write a single cohesive
summary of the entire document in 12-16 sentences.

Requirements:
- Preserve the progression across sections: state what each part establishes or what
  happens in it, and how each step relates to or follows from the previous one.
- Refer to specific entities named in the sections (people, places, concepts, systems,
  findings, quantities); avoid generic placeholders like "something" or "a thing".
- Keep concrete details — specific actions, decisions, items, messages, methods,
  claims, or findings — rather than compressing everything into emotional or thematic
  framing.
- Cover the full span of the document: opening, key developments in order, and
  conclusion or resolution.
- Do not introduce details absent from the section summaries.
- Start directly with the summary. Do NOT begin with "Here's a summary", "This is a
  summary", or any similar preamble.

Section summaries:
{summaries}

Document summary:"""


_REDUCE_INTERMEDIATE_PROMPT = """\
Below is a batch of consecutive section summaries from "{source}", in document order.
Condense them into 3-5 sentences that preserve specific entities named, concrete
events or claims, and how each one relates to the next.

Requirements:
- Do not introduce details absent from the input.
- Start directly with the condensed summary. Do NOT begin with "Here's a summary" or
  any similar preamble.

Section summaries:
{summaries}

Condensed summary:"""


# Cache I/O

def _cache_path(source: str) -> str:
    """Return the JSON cache path for a given source filename."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", source)
    return os.path.join(SUMMARY_CACHE_DIR, f"{safe}.json")


def _chunk_hash(chunks: list[dict]) -> str:
    """Hash the chunk texts so we can invalidate the cache when chunking changes."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_summary(source: str) -> dict | None:
    """Read a cached summary for a source. Returns None if missing or unreadable."""
    path = _cache_path(source)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_summary(record: dict) -> str:
    """Persist a summary record to disk and return the path."""
    os.makedirs(SUMMARY_CACHE_DIR, exist_ok=True)
    path = _cache_path(record["source"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


# Map-reduce

def _map_sections(source: str, chunks: list[dict], model: str) -> list[dict]:
    """Map stage: group chunks into batches, summarize each batch into a section."""
    sections = []
    for i in range(0, len(chunks), SUMMARY_MAP_BATCH):
        batch = chunks[i : i + SUMMARY_MAP_BATCH]
        joined = "\n\n".join(c["text"] for c in batch)
        prompt = _MAP_PROMPT.format(source=source, text=joined)
        summary = _strip_preamble(llm.generate(prompt, model=model))
        sections.append(
            {
                "chunk_range": [batch[0]["metadata"]["chunk_index"],
                                batch[-1]["metadata"]["chunk_index"]],
                "summary": summary,
            }
        )
        print(f"    map {i // SUMMARY_MAP_BATCH + 1}/"
              f"{(len(chunks) + SUMMARY_MAP_BATCH - 1) // SUMMARY_MAP_BATCH}: "
              f"chunks {batch[0]['metadata']['chunk_index']}-"
              f"{batch[-1]['metadata']['chunk_index']} "
              f"({len(summary.split())} words)")
    return sections


def _reduce_sections(source: str, sections: list[dict], model: str) -> str:
    """
    Reduce stage: fold section summaries into a single document summary.
    If there are more than SUMMARY_REDUCE_BATCH sections, reduce in passes
    so we never exceed the LLM context window.
    """
    summaries = [s["summary"] for s in sections]
    pass_num = 0

    while len(summaries) > SUMMARY_REDUCE_BATCH:
        pass_num += 1
        print(f"    reduce pass {pass_num}: {len(summaries)} summaries -> ", end="")
        condensed = []
        for i in range(0, len(summaries), SUMMARY_REDUCE_BATCH):
            batch = summaries[i : i + SUMMARY_REDUCE_BATCH]
            joined = "\n\n".join(f"- {s}" for s in batch)
            prompt = _REDUCE_INTERMEDIATE_PROMPT.format(source=source, summaries=joined)
            condensed.append(
                _strip_preamble(llm.generate(
                    prompt, model=model,
                    options={"num_predict": 400, "num_ctx": SUMMARY_CTX},
                ))
            )
        summaries = condensed
        print(f"{len(summaries)}")

    joined = "\n\n".join(f"- {s}" for s in summaries)
    prompt = _REDUCE_PROMPT.format(source=source, summaries=joined)
    # Allow room for 12-16 sentences (~ 400-600 words ~ 800-1000 tokens).
    return _strip_preamble(llm.generate(
        prompt, model=model,
        options={"num_predict": 1024, "num_ctx": SUMMARY_CTX},
    ))


def build_summary(
    source: str,
    chunks: list[dict],
    model: str = SUMMARY_MODEL,
    force: bool = False,
) -> dict:
    """
    Build (or load) a hierarchical summary for one document's chunks.

    Cache hits short-circuit the LLM calls unless force=True. Chunk-content
    hashing ensures the cache is invalidated if the chunks change.

    Args:
        source: Filename the chunks came from (used as cache key).
        chunks: List of chunk dicts with keys: id, text, metadata.
        model:  Ollama model to use for map+reduce.
        force:  Rebuild even if a valid cache exists.

    Returns a dict: {source, chunk_count, chunk_hash, sections, summary,
                     model, created_at}.
    """
    chunk_hash = _chunk_hash(chunks)

    if not force:
        cached = load_summary(source)
        if cached and cached.get("chunk_hash") == chunk_hash:
            print(f"  Summary cache hit: {source} "
                  f"({cached.get('chunk_count', '?')} chunks, "
                  f"{len(cached.get('summary', '').split())} words)")
            return cached

    print(f"  Summarizing {source} ({len(chunks)} chunks, model={model})")
    t0 = time.time()

    sections = _map_sections(source, chunks, model)
    doc_summary = _reduce_sections(source, sections, model)

    record = {
        "source":      source,
        "chunk_count": len(chunks),
        "chunk_hash":  chunk_hash,
        "sections":    sections,
        "summary":     doc_summary,
        "model":       model,
        "created_at":  datetime.now(timezone.utc).isoformat(),
    }
    save_summary(record)
    print(f"  Summary built in {time.time() - t0:.1f}s "
          f"({len(sections)} sections -> {len(doc_summary.split())} words)")
    return record


def build_all(
    chunks: list[dict],
    model: str = SUMMARY_MODEL,
    force: bool = False,
) -> dict[str, dict]:
    """
    Build summaries for every source represented in a flat chunk list.
    Returns a mapping {source: summary_record}.
    """
    by_source: dict[str, list[dict]] = {}
    for c in chunks:
        by_source.setdefault(c["metadata"]["source"], []).append(c)

    # Keep each source's chunks in document order (they already are, but be safe).
    for src in by_source:
        by_source[src].sort(key=lambda c: c["metadata"]["chunk_index"])

    summaries = {}
    for src, src_chunks in by_source.items():
        summaries[src] = build_summary(src, src_chunks, model=model, force=force)
    return summaries


# Query-time helpers

def get_cached_summaries() -> list[dict]:
    """Return every summary record currently on disk, sorted by source name."""
    if not os.path.isdir(SUMMARY_CACHE_DIR):
        return []
    records = []
    for fname in sorted(os.listdir(SUMMARY_CACHE_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(SUMMARY_CACHE_DIR, fname), "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return records


def answer_summarization_query(question: str) -> dict | None:
    """
    Build an answer for a whole-document summarization query using cached
    summaries. Returns None if no summaries are cached (caller should fall
    back to normal RAG).

    In a multi-doc setup, natural-language targeting picks which summaries
    to return: "summarize the hamlet book" -> only Hamlet's summary, even
    if Romeo and Juliet is also ingested. See _select_target for details.

    Returns a dict shaped like rag.generate()'s output so the voice loop
    and --query path can consume it uniformly.
    """
    records = get_cached_summaries()
    if not records:
        return None

    selected = _select_target(question, records)
    if len(selected) < len(records):
        picked = ", ".join(r["source"] for r in selected)
        print(f"  [summary] targeting {len(selected)}/{len(records)} docs: {picked}")

    if len(selected) == 1:
        answer = selected[0]["summary"]
    else:
        answer = "\n\n".join(f"{r['source']}:\n{r['summary']}" for r in selected)

    # Shape the response to match rag.generate() so callers don't branch.
    contexts = [
        {
            "text":        r["summary"],
            "source":      r["source"],
            "chunk_index": -1,
            "pages":       [],
            "word_count":  len(r["summary"].split()),
            "distance":    None,
            "summary":     True,
        }
        for r in selected
    ]
    return {
        "question": question,
        "answer":   answer,
        "contexts": contexts,
        "timings":  {"summary_lookup": 0.0},
        "route":    "summary_cache",
    }
