"""
vector_store.py — ChromaDB persistence layer.

Handles collection creation, batch upsert, and semantic query.
The collection uses cosine similarity (hnsw:space = cosine).
"""

import time
import chromadb
from config import CHROMA_PATH, COLLECTION_NAME, N_RESULTS
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
