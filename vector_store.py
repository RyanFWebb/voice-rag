"""
vector_store.py — ChromaDB persistence layer.

Handles collection creation, batch upsert, and semantic query.
The collection uses cosine similarity (hnsw:space = cosine).
"""

import time
import chromadb
from config import CHROMA_PATH, COLLECTION_NAME, N_RESULTS, N_RETRIEVE, NEIGHBOR_WINDOW
import llm


def get_collection(reset: bool = False) -> chromadb.Collection:
    """
    Return (or create) the persistent ChromaDB collection.

    Args:
        reset: If True, delete and recreate the collection.
               Use this when re-ingesting from scratch.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def upsert_chunks(chunks: list[dict], collection: chromadb.Collection) -> None:
    """
    Embed and upsert a list of chunk dicts into ChromaDB.

    Each chunk dict must have keys: id, text, metadata.
    Embeddings are generated in batches via llm.embed().
    """
    if not chunks:
        print("No chunks to upsert.")
        return

    print(f"Embedding {len(chunks)} chunks...")
    t0 = time.time()
    texts = [c["text"] for c in chunks]
    embeddings = llm.embed(texts)
    print(f"Embedding complete in {time.time() - t0:.1f}s")

    UPSERT_BATCH = 100
    for i in range(0, len(chunks), UPSERT_BATCH):
        batch = chunks[i : i + UPSERT_BATCH]
        emb_batch = embeddings[i : i + UPSERT_BATCH]
        collection.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=emb_batch,
            metadatas=[c["metadata"] for c in batch],
        )

    print(f"Upserted {len(chunks)} chunks. Collection size: {collection.count()}")


def query(
    question: str,
    collection: chromadb.Collection,
    n_results: int = N_RESULTS,
) -> list[dict]:
    """
    Embed the question and retrieve the top-n most similar chunks.

    Returns a list of dicts with keys:
        text, source, chunk_index, pages, word_count, distance
    """
    q_embedding = llm.embed(question)[0]
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    contexts = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        pages_str = meta.get("pages", "")
        pages = [int(p) for p in pages_str.split(",") if p] if pages_str else []
        contexts.append(
            {
                "text":        doc,
                "source":      meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "pages":       pages,
                "word_count":  meta.get("word_count", 0),
                "distance":    dist,
            }
        )
    return contexts


def expand_neighbors(
    contexts: list[dict],
    collection: chromadb.Collection,
    window: int = NEIGHBOR_WINDOW,
) -> list[dict]:
    """
    For each retrieved chunk, fetch its neighboring chunks (±window) from the
    same document. Returns a deduplicated list sorted by (source, chunk_index).

    Uses the known ID format: {source}_chunk_{index:04d}
    """
    if window == 0 or not contexts:
        return contexts

    # Build set of IDs we need to fetch. Overlapping windows (e.g. two reranked
    # hits 2 chunks apart) can target the same neighbor — dedupe via a set.
    existing = {(c["source"], c["chunk_index"]): c for c in contexts}
    fetch_set = set()

    for ctx in contexts:
        source = ctx["source"]
        idx = ctx["chunk_index"]
        for offset in range(-window, window + 1):
            neighbor_idx = idx + offset
            if neighbor_idx < 0 or (source, neighbor_idx) in existing:
                continue
            fetch_set.add((source, neighbor_idx))

    ids_to_fetch = list(fetch_set)

    if not ids_to_fetch:
        return sorted(contexts, key=lambda c: (c["source"], c["chunk_index"]))

    # Fetch neighbors by ID (fast — no embedding, just a key lookup)
    neighbor_ids = [f"{src}_chunk_{idx:04d}" for src, idx in ids_to_fetch]
    result = collection.get(ids=neighbor_ids, include=["documents", "metadatas"])

    for doc_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
        pages_str = meta.get("pages", "")
        pages = [int(p) for p in pages_str.split(",") if p] if pages_str else []
        key = (meta.get("source", "unknown"), meta.get("chunk_index", -1))
        if key not in existing:
            existing[key] = {
                "text":        doc,
                "source":      key[0],
                "chunk_index": key[1],
                "pages":       pages,
                "word_count":  meta.get("word_count", 0),
                "distance":    None,  # not retrieved by similarity
                "neighbor":    True,  # flag so we know this was expanded
            }

    return sorted(existing.values(), key=lambda c: (c["source"], c["chunk_index"]))
