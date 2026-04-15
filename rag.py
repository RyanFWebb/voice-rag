"""
rag.py — Retrieval-Augmented Generation.

Composes vector_store.query() + llm.generate() into a single
rag_generate() call. The prompt template lives here so it can
be tuned independently of the retrieval or generation code.

Retrieval uses a two-stage pipeline:
  1. ChromaDB cosine similarity → N_RETRIEVE candidates (fast, approximate)
  2. Cross-encoder reranking    → top N_RESULTS (slower, precise)
"""

import time

import vector_store
import llm
from sentence_transformers import CrossEncoder
from config import N_RESULTS, N_RETRIEVE, GEN_MODEL, RERANK_MODEL, NEIGHBOR_WINDOW


_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    """Load the cross-encoder reranker on first use."""
    global _reranker
    if _reranker is None:
        print(f"Loading reranker ({RERANK_MODEL})...")
        _reranker = CrossEncoder(RERANK_MODEL)
        print("Reranker ready.")
    return _reranker


def rerank(question: str, contexts: list[dict], top_k: int = N_RESULTS) -> list[dict]:
    """
    Score each (question, chunk) pair with a cross-encoder and return
    the top_k most relevant contexts, sorted by reranker score.
    """
    if not contexts:
        return contexts

    reranker = _get_reranker()
    pairs = [(question, ctx["text"]) for ctx in contexts]
    scores = reranker.predict(pairs)

    for ctx, score in zip(contexts, scores):
        ctx["rerank_score"] = float(score)

    ranked = sorted(contexts, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


RAG_PROMPT = """\
You are a helpful assistant that answers questions using the provided document excerpts.

Context from documents:

{context}

Question: {question}

Instructions:
- Use the provided context as your primary source of truth
- You may supplement with your own knowledge to fill gaps or add clarity, but always defer to the context when there is a conflict
- You may synthesize and infer from the context (e.g., infer character relationships from dialogue)
- If the context does not contain relevant information, say so clearly and rely on your own knowledge
- Answer in 3-5 sentences, covering all key details\
"""


def generate(
    question: str,
    collection,
    n_results: int = N_RESULTS,
    n_retrieve: int = N_RETRIEVE,
    neighbor_window: int = NEIGHBOR_WINDOW,
    model: str = GEN_MODEL,
) -> dict:
    """
    Retrieve relevant chunks for question, then generate a grounded answer.

    Three-stage retrieval:
      1. Fetch n_retrieve candidates from ChromaDB (fast cosine similarity)
      2. Rerank with cross-encoder, keep top n_results
      3. Expand with ±neighbor_window adjacent chunks, sorted by document order

    Returns a dict with keys:
        question  (str)
        answer    (str)
        contexts  (list[dict])
        timings   (dict)  — stage-by-stage latency breakdown
    """
    timings = {}

    t0 = time.time()
    candidates = vector_store.query(question, collection, n_results=n_retrieve)
    timings["retrieve"] = time.time() - t0

    t0 = time.time()
    reranked = rerank(question, candidates, top_k=n_results)
    timings["rerank"] = time.time() - t0

    t0 = time.time()
    contexts = vector_store.expand_neighbors(reranked, collection, window=neighbor_window)
    timings["expand"] = time.time() - t0

    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']} | "
        f"Pages: {','.join(str(p) for p in c['pages']) if c['pages'] else 'N/A'}]\n"
        f"{c['text']}"
        for c in contexts
    )

    prompt = RAG_PROMPT.format(context=context_text, question=question)

    t0 = time.time()
    answer = llm.generate(prompt, model=model)
    timings["llm"] = time.time() - t0

    return {"question": question, "answer": answer, "contexts": contexts, "timings": timings}


def print_contexts(contexts: list[dict]) -> None:
    """Pretty-print retrieved chunk metadata to stdout."""
    for i, ctx in enumerate(contexts, 1):
        pages = ", ".join(str(p) for p in ctx["pages"]) if ctx["pages"] else "N/A"
        is_neighbor = ctx.get("neighbor", False)
        tag = " [neighbor]" if is_neighbor else ""
        print(f"  [{i}] {ctx['source']}{tag}")
        score_parts = []
        if ctx.get("distance") is not None:
            score_parts.append(f"Distance: {ctx['distance']:.4f}")
        if "rerank_score" in ctx:
            score_parts.append(f"Rerank: {ctx['rerank_score']:.4f}")
        scores = " | ".join(score_parts)
        print(
            f"      Pages: {pages} | "
            f"Chunk #{ctx['chunk_index']} | "
            f"{ctx['word_count']} words"
            f"{' | ' + scores if scores else ''}"
        )
        print(f"      Preview: {ctx['text'][:150]}...")
        print()


def print_timings(timings: dict) -> None:
    """Print a latency breakdown for the RAG pipeline."""
    total = sum(timings.values())
    print("Timings:")
    for stage, elapsed in timings.items():
        pct = elapsed / total * 100 if total else 0
        print(f"  {stage:>10s}: {elapsed:5.2f}s  ({pct:.0f}%)")
    print(f"  {'total':>10s}: {total:5.2f}s")
