"""
rag.py — Retrieval-Augmented Generation.

Composes vector_store.query() + llm.generate() into a single
rag_generate() call. The prompt template lives here so it can
be tuned independently of the retrieval or generation code.
"""

import vector_store
import llm
from config import N_RESULTS, GEN_MODEL


RAG_PROMPT = """\
You are a helpful assistant that answers questions using the provided document excerpts.

Context from documents:

{context}

Question: {question}

Instructions:
- Use the provided context to answer the question thoroughly
- You may synthesize and infer from the context (e.g., infer character relationships from dialogue)
- If the context does not contain relevant information, say so clearly
- Answer in 2-3 sentences maximum\
"""


def generate(
    question: str,
    collection,
    n_results: int = N_RESULTS,
    model: str = GEN_MODEL,
) -> dict:
    """
    Retrieve relevant chunks for question, then generate a grounded answer.

    Returns a dict with keys:
        question  (str)
        answer    (str)
        contexts  (list[dict])  — see vector_store.query() for shape
    """
    contexts = vector_store.query(question, collection, n_results=n_results)

    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']} | "
        f"Pages: {','.join(str(p) for p in c['pages']) if c['pages'] else 'N/A'}]\n"
        f"{c['text']}"
        for c in contexts
    )

    prompt = RAG_PROMPT.format(context=context_text, question=question)
    answer = llm.generate(prompt, model=model)

    return {"question": question, "answer": answer, "contexts": contexts}


def print_contexts(contexts: list[dict]) -> None:
    """Pretty-print retrieved chunk metadata to stdout."""
    for i, ctx in enumerate(contexts, 1):
        pages = ", ".join(str(p) for p in ctx["pages"]) if ctx["pages"] else "N/A"
        print(f"  [{i}] {ctx['source']}")
        print(
            f"      Pages: {pages} | "
            f"Chunk #{ctx['chunk_index']} | "
            f"{ctx['word_count']} words | "
            f"Distance: {ctx['distance']:.4f}"
        )
        print(f"      Preview: {ctx['text'][:150]}...")
        print()
