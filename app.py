"""
app.py — Gradio web UI for the Voice RAG assistant.

Run with:  python app.py
Then open: http://localhost:7860
"""

import sys

# Force UTF-8 for stdout/stderr so Unicode chars like "→" don't crash on Windows cp1252.
# Must run before any module prints anything.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
import shutil
import time
import traceback

import gradio as gr

import llm
import rag
import speech
import vector_store
from config import DOCUMENTS_DIR, N_RESULTS
from ingest import load_and_chunk_all


# Helpers

OLLAMA_HELP = (
    "Ollama is not running. This app needs Ollama for embeddings and generation.\n"
    "Fix:\n"
    "  1. Install Ollama from https://ollama.com/download\n"
    "  2. Open a terminal and run:  ollama serve\n"
    "  3. In another terminal run:  ollama pull nomic-embed-text\n"
    "                               ollama pull gemma3:4b"
)


def _ollama_reachable() -> bool:
    try:
        llm.check_ollama()
        return True
    except Exception:
        return False


def _collection_status() -> str:
    if not _ollama_reachable():
        return "Ollama offline — please start Ollama (see Ingestion Status for help)"
    try:
        col = vector_store.get_collection()
        count = col.count()
        return f"{count} chunk(s) indexed" if count else "Empty — upload and ingest documents first"
    except Exception as e:
        return f"Error reading store: {e}"


def _format_contexts(contexts: list[dict]) -> str:
    if not contexts:
        return "No chunks retrieved."
    lines = []
    for i, ctx in enumerate(contexts, 1):
        pages = ", ".join(str(p) for p in ctx["pages"]) if ctx["pages"] else "N/A"
        dist = ctx.get("distance")
        dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "n/a"
        lines.append(
            f"[{i}] {ctx['source']}  |  Pages: {pages}  |  Distance: {dist_str}\n"
            f"    {ctx['text'][:250]}{'...' if len(ctx['text']) > 250 else ''}"
        )
    return "\n\n".join(lines)


# Event handlers  (plain functions, not generators — Gradio 6 compatible)

def upload_and_ingest(files, reset_flag: bool, progress=gr.Progress()):
    if not files:
        return "No files selected.", _collection_status()

    if not _ollama_reachable():
        return OLLAMA_HELP, _collection_status()

    try:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)

        progress(0.05, desc="Copying files to documents/...")
        uploaded = []
        for f in files:
            # Gradio 6 passes file paths as plain strings
            fpath = f if isinstance(f, str) else getattr(f, "name", str(f))
            fname = os.path.basename(fpath)
            dest = os.path.join(DOCUMENTS_DIR, fname)
            shutil.copy(fpath, dest)
            uploaded.append(fname)

        progress(0.25, desc="Loading and chunking documents...")
        chunks, _ = load_and_chunk_all(DOCUMENTS_DIR)
        if not chunks:
            return "No supported documents found in documents/.", _collection_status()

        progress(0.55, desc=f"Embedding {len(chunks)} chunks — this may take a minute...")
        collection = vector_store.get_collection(reset=reset_flag)
        vector_store.upsert_chunks(chunks, collection)

        progress(1.0, desc="Ingestion complete!")
        summary = (
            f"Done! Ingested {len(uploaded)} file(s): {', '.join(uploaded)}\n"
            f"Total chunks stored: {len(chunks)}"
        )
        return summary, _collection_status()

    except Exception as e:
        tb = traceback.format_exc()
        with open("app_error.log", "a", encoding="utf-8") as log:
            log.write(tb + "\n")
        return f"Error during ingestion:\n{e}", _collection_status()


def text_query(question: str, progress=gr.Progress()):
    if not question.strip():
        return "Please enter a question.", "", ""

    if not _ollama_reachable():
        return OLLAMA_HELP, "", ""

    try:
        progress(0.1, desc="Connecting to vector store...")
        collection = vector_store.get_collection()
        if collection.count() == 0:
            return "Vector store is empty. Upload and ingest documents first.", "", ""

        progress(0.4, desc="Retrieving relevant chunks...")
        t0 = time.time()

        progress(0.6, desc="Generating answer with LLM (may take ~10-30s)...")
        result = rag.generate(question, collection, n_results=N_RESULTS)
        elapsed = time.time() - t0

        progress(1.0, desc="Done!")
        return result["answer"], f"{elapsed:.1f}s", _format_contexts(result["contexts"])

    except Exception as e:
        tb = traceback.format_exc()
        with open("app_error.log", "a", encoding="utf-8") as log:
            log.write(tb + "\n")
        return f"Error: {e}", "", ""


def voice_query(audio, progress=gr.Progress()):
    # audio is a filepath string (gr.Audio type="filepath") — Gradio has already
    # saved the recording to disk, so we don't ship a numpy array over the
    # websocket or keep one in the browser component's state.
    if not audio:
        return "No audio recorded.", "", "", ""

    if not _ollama_reachable():
        return "", OLLAMA_HELP, "", ""

    try:
        progress(0.20, desc="Loading Whisper (first run ~30s)...")
        speech.load_whisper()

        progress(0.40, desc="Transcribing with Whisper...")
        question = speech.transcribe(audio)

        if not question.strip():
            return "No speech detected — please try again.", "", "", ""

        progress(0.55, desc="Connecting to vector store...")
        collection = vector_store.get_collection()
        if collection.count() == 0:
            return question, "Vector store is empty. Upload and ingest documents first.", "", ""

        progress(0.70, desc="Generating answer with LLM (may take ~10-30s)...")
        t0 = time.time()
        result = rag.generate(question, collection, n_results=N_RESULTS)
        elapsed = time.time() - t0

        progress(1.0, desc="Done!")
        return question, result["answer"], f"{elapsed:.1f}s", _format_contexts(result["contexts"])

    except Exception as e:
        tb = traceback.format_exc()
        with open("app_error.log", "a", encoding="utf-8") as log:
            log.write(tb + "\n")
        return f"Error: {e}", "", "", ""


# UI layout

with gr.Blocks(title="Voice RAG") as demo:
    gr.Markdown("# Voice RAG Assistant")
    gr.Markdown("Ask questions about your documents using text or your microphone.")

    with gr.Row():
        # Compute once at build time. Passing a callable here makes Gradio run
        # it as a load event on every (re)connect, which re-opens Chroma +
        # re-pings Ollama and can back up behind a queued query.
        status_box = gr.Textbox(
            label="Vector Store Status",
            value=_collection_status(),
            interactive=False,
            max_lines=1,
            scale=4,
        )
        refresh_btn = gr.Button("Refresh", scale=1)
        refresh_btn.click(fn=_collection_status, inputs=None, outputs=status_box)

    with gr.Tabs():

        # Documents
        with gr.Tab("Documents"):
            gr.Markdown("Upload PDF or TXT files, then click **Ingest Documents** to index them.")
            file_upload = gr.File(
                label="Drop files here or click to upload",
                file_types=[".pdf", ".txt"],
                file_count="multiple",
            )
            reset_check = gr.Checkbox(
                label="Reset vector store before ingesting (wipes existing index)",
                value=False,
            )
            ingest_btn = gr.Button("Ingest Documents", variant="primary")
            ingest_status = gr.Textbox(label="Ingestion Status", interactive=False, lines=3)

            ingest_btn.click(
                fn=upload_and_ingest,
                inputs=[file_upload, reset_check],
                outputs=[ingest_status, status_box],
            )

        # Text Query 
        with gr.Tab("Text Query"):
            gr.Markdown("Type your question and press **Enter** or click **Ask**.")
            text_input = gr.Textbox(
                label="Your question",
                placeholder="e.g. Who is the main character?",
                lines=2,
            )
            text_btn = gr.Button("Ask", variant="primary")
            with gr.Row():
                text_answer = gr.Textbox(label="Answer", interactive=False, lines=5, scale=4)
                text_time   = gr.Textbox(label="Time", interactive=False, max_lines=1, scale=1)
            text_contexts = gr.Textbox(label="Retrieved Chunks", interactive=False, lines=8)

            text_btn.click(
                fn=text_query,
                inputs=[text_input],
                outputs=[text_answer, text_time, text_contexts],
            )
            text_input.submit(
                fn=text_query,
                inputs=[text_input],
                outputs=[text_answer, text_time, text_contexts],
            )

        # Voice Query
        with gr.Tab("Voice Query"):
            gr.Markdown("Record your question, then press **Transcribe & Ask**.")
            mic_input = gr.Audio(
                label="Record Question",
                sources=["microphone"],
                type="filepath",
            )
            voice_btn = gr.Button("Transcribe & Ask", variant="primary")
            voice_transcription = gr.Textbox(label="Transcription", interactive=False, lines=2)
            with gr.Row():
                voice_answer = gr.Textbox(label="Answer", interactive=False, lines=5, scale=4)
                voice_time   = gr.Textbox(label="Time", interactive=False, max_lines=1, scale=1)
            voice_contexts = gr.Textbox(label="Retrieved Chunks", interactive=False, lines=8)

            voice_btn.click(
                fn=voice_query,
                inputs=[mic_input],
                outputs=[voice_transcription, voice_answer, voice_time, voice_contexts],
            )


def _prewarm_models() -> None:
    """Load heavy models once at startup so the first query isn't a cold start.

    Skips Kokoro — the web UI has no voice-out path, so its ~330MB stays unloaded.
    """
    print("Pre-warming models...")
    if _ollama_reachable():
        try:
            rag._get_reranker()
        except Exception as e:
            print(f"  Reranker pre-warm failed: {e}")
        try:
            speech.load_whisper()
        except Exception as e:
            print(f"  Whisper pre-warm failed: {e}")
    else:
        print("  Ollama not reachable — skipping pre-warm.")
    print("Pre-warm done.")


if __name__ == "__main__":
    _prewarm_models()
    # Serialize heavy work so impatient re-clicks don't spawn parallel model loads.
    demo.queue(default_concurrency_limit=1, max_size=16)
    demo.launch(theme=gr.themes.Soft())
