"""
main.py — Entry point for the Voice RAG assistant.

Usage:
    python main.py --check                          # verify Python, tools, Ollama
    python main.py --ingest                         # load docs → chunk → embed → store → summarize
    python main.py --ingest --reset                 # wipe collection first, then ingest
    python main.py --ingest --skip-summary          # ingest without building summaries
    python main.py --summarize                      # (re)build summaries for already-ingested docs
    python main.py --summarize --force-summary      # ignore cache and rebuild every summary
    python main.py --query "Who is Alice?"          # single text query, no mic
    python main.py --query "Summarize the book"     # auto-routed to pre-computed summary
    python main.py --audio test_queries/q01.wav     # run pipeline on a saved WAV file
    python main.py --generate-test-queries          # create test WAV files
    python main.py                                  # live voice loop (mic → answer → speaker)
"""

import sys

# Python version guard
# Runs before any other imports — blis (kokoro → spacy → blis) has no
# pre-built wheels for Python 3.13+ and fails to compile from source.
_MAJOR, _MINOR = sys.version_info[:2]
if (_MAJOR, _MINOR) >= (3, 13):
    print(
        f"ERROR: Python {_MAJOR}.{_MINOR} is not supported.\n"
        "\n"
        "  kokoro -> spacy -> blis does not have pre-built wheels for Python 3.13+\n"
        "  and fails to compile from source.\n"
        "\n"
        "  Required: Python 3.10, 3.11, or 3.12\n"
        f"  Current:  Python {_MAJOR}.{_MINOR}\n"
        "\n"
        "  Fix (macOS):\n"
        "    brew install python@3.11\n"
        "    python3.11 -m venv .venv\n"
        "    source .venv/bin/activate\n"
        "    pip install -r requirements.txt\n"
        "\n"
        "  Fix (Windows):\n"
        "    Download Python 3.11 from https://python.org/downloads\n"
        "    py -3.11 -m venv .venv\n"
        "    .venv\\Scripts\\activate\n"
        "    pip install -r requirements.txt\n"
    )
    sys.exit(1)

import argparse
import os
import time

import soundfile as sf

import llm
import rag
import speech
import summarize
import vector_store
from config import (
    DOCUMENTS_DIR,
    TEST_QUERY_DIR,
    N_RESULTS,
    RECORD_SECONDS,
    RECORD_SAMPLE_RATE,
    TTS_VOICE,
    TTS_SPEED,
    TTS_SAMPLE_RATE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE,
)
from ingest import load_and_chunk_all


# Subcommands

def cmd_check():
    """Verify Python version, external tools, and Ollama connection."""
    print("=== Environment Check ===\n")

    # Python version
    py_ver = f"{_MAJOR}.{_MINOR}.{sys.version_info[2]}"
    supported = (_MAJOR, _MINOR) in ((3, 10), (3, 11), (3, 12))
    tick = "\u2713" if supported else "\u2717  (3.10-3.12 required -- see requirements.txt)"
    print(f"Python: {py_ver}  {tick}")

    # Whisper acceleration
    print(f"\nWhisper: device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE}")

    # External tools
    print("\nExternal tools:")
    tools = speech.check_dependencies()
    for name, path in tools.items():
        status = path if path else "NOT FOUND"
        print(f"  {name}: {status}")

    # Ollama
    print("\nOllama:")
    try:
        models = llm.check_ollama()
        print(f"  Reachable — {len(models)} model(s) available:")
        for m in models:
            print(f"    - {m}")
    except Exception as e:
        print(f"  NOT reachable: {e}")
        print("  Start it with: ollama serve")

    print("\nDone.")


def cmd_ingest(reset: bool = False, skip_summary: bool = False, force_summary: bool = False):
    """Load all documents from documents/, chunk, embed, store, and summarize."""
    print("=== Ingestion ===\n")
    chunks, df = load_and_chunk_all(DOCUMENTS_DIR)

    if not chunks:
        print(f"\nNo documents found. Add .pdf or .txt files to {DOCUMENTS_DIR}/")
        sys.exit(1)

    print()
    collection = vector_store.get_collection(reset=reset)
    vector_store.upsert_chunks(chunks, collection)

    if skip_summary:
        print("\nSkipping summarization (--skip-summary).")
    else:
        print("\n=== Summarization (map-reduce) ===")
        summarize.build_all(chunks, force=force_summary)

    print("\nIngestion complete.")


def cmd_query(question: str):
    """Run a single text query and print retrieved chunks + answer."""
    collection = vector_store.get_collection()
    if collection.count() == 0:
        print("Vector store is empty. Run: python main.py --ingest")
        sys.exit(1)

    print(f"Query: {question}\n")
    result = rag.generate(question, collection, n_results=N_RESULTS)

    print("Retrieved chunks:")
    rag.print_contexts(result["contexts"])
    print()
    rag.print_timings(result["timings"])
    print(f"\nAnswer:")
    print(result["answer"])
    print("\nSources:")
    rag.print_sources(result["contexts"])


def cmd_voice(audio_input: str | None = None):
    """
    Voice RAG loop.
    - If audio_input is a file path, run the pipeline once on that file.
    - Otherwise, record from the microphone in a loop until Ctrl-C.
    """
    collection = vector_store.get_collection()
    if collection.count() == 0:
        print("Vector store is empty. Run: python main.py --ingest")
        sys.exit(1)

    speech.load_models()
    print("\n=== Voice RAG Ready ===")
    if not audio_input:
        print(f"Recording {RECORD_SECONDS}s per query. Press Ctrl-C to quit.\n")

    try:
        while True:
            # Step 1: get audio
            if audio_input:
                print(f"Using file: {audio_input}")
                wav_path = audio_input
            else:
                input("Press Enter to record...")
                wav_path = speech.record(
                    seconds=RECORD_SECONDS,
                    sample_rate=RECORD_SAMPLE_RATE,
                    output_path="rag_question.wav",
                )

            # Step 2: transcribe
            print("Transcribing...")
            t0 = time.time()
            question = speech.transcribe(wav_path)
            stt_time = time.time() - t0
            print(f"You asked ({stt_time:.1f}s): {question}\n")

            if not question.strip():
                print("No speech detected. Try again.\n")
                if audio_input:
                    break
                continue

            # Step 3: RAG
            print("Retrieving and generating answer...")
            t0 = time.time()
            result = rag.generate(question, collection, n_results=N_RESULTS)
            rag_time = time.time() - t0
            print(f"\nAnswer ({rag_time:.1f}s):")
            print(result["answer"])
            print("\nSources:")
            rag.print_sources(result["contexts"])
            print("\nRetrieved chunks:")
            rag.print_contexts(result["contexts"])

            # Step 4: TTS (strip [N] citations so they don't get spoken)
            print("Speaking answer...")
            t0 = time.time()
            speech.speak(rag.strip_citations(result["answer"]), output_path="rag_voice_reply.wav")
            tts_time = time.time() - t0
            print(f"TTS done ({tts_time:.1f}s) | Total: {stt_time + rag_time + tts_time:.1f}s\n")
            print("-" * 60 + "\n")

            if audio_input:
                break

    except KeyboardInterrupt:
        print("\nExiting.")


def cmd_generate_test_queries():
    """Generate a bank of test WAV files using Kokoro TTS."""
    speech.load_models()
    os.makedirs(TEST_QUERY_DIR, exist_ok=True)

    query_bank = [
        ("q01_character_intro",    "Who is the main character, and how are they introduced?"),
        ("q02_central_conflict",   "What is the central conflict of the story?"),
        ("q03_turning_point",      "What event becomes the turning point in the story?"),
        ("q04_theme",              "What are the main themes explored in this text?"),
        ("q05_supporting_cast",    "Who are the key supporting characters and what roles do they play?"),
        ("q06_ending",             "How does the story end, and what can we learn from it?"),
    ]

    print(f"Generating {len(query_bank)} test queries in {TEST_QUERY_DIR}/\n")
    for name, text in query_bank:
        out = os.path.join(TEST_QUERY_DIR, f"{name}.wav")
        audio = speech.synthesize(text, voice=TTS_VOICE, speed=TTS_SPEED)
        sf.write(out, audio, TTS_SAMPLE_RATE)
        duration = len(audio) / TTS_SAMPLE_RATE
        print(f"  [{name}] {duration:.1f}s — {text}")

    print(f"\nDone. Run with: python main.py --audio {TEST_QUERY_DIR}/<name>.wav")


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Voice RAG assistant — Whisper STT -> ChromaDB -> Ollama -> Kokoro TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--check",  action="store_true",
                   help="Check Python version, external tools, and Ollama")
    p.add_argument("--ingest", action="store_true",
                   help="Load, chunk, embed, and store documents in ChromaDB")
    p.add_argument("--reset",  action="store_true",
                   help="Wipe the ChromaDB collection before ingesting")
    p.add_argument("--skip-summary", action="store_true",
                   help="Skip map-reduce summarization during --ingest")
    p.add_argument("--force-summary", action="store_true",
                   help="Rebuild document summaries even if cached")
    p.add_argument("--summarize", action="store_true",
                   help="Build (or rebuild with --force-summary) summaries for ingested docs, then print them")
    p.add_argument("--query",  metavar="TEXT",
                   help="Run a single text query (no microphone)")
    p.add_argument("--audio",  metavar="FILE",
                   help="Run the full pipeline on a pre-recorded WAV file")
    p.add_argument("--generate-test-queries", action="store_true",
                   help="Generate generic test query WAV files using TTS")
    return p.parse_args()


def cmd_summarize(force: bool = False):
    """Build (or reuse) map-reduce summaries for every ingested source."""
    collection = vector_store.get_collection()
    if collection.count() == 0:
        print("Vector store is empty. Run: python main.py --ingest")
        sys.exit(1)

    # Reconstruct chunks from the collection so we can summarize without
    # re-reading the raw documents from disk.
    got = collection.get(include=["documents", "metadatas"])
    chunks = [
        {"id": cid, "text": doc, "metadata": meta}
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
    ]

    if not chunks:
        print("No chunks found in the collection.")
        sys.exit(1)

    print(f"=== Summarization === ({len(chunks)} chunks)\n")
    results = summarize.build_all(chunks, force=force)

    print("\n=== Document Summaries ===\n")
    for source, record in results.items():
        print(f"--- {source} ---")
        print(record["summary"])
        print()


def main():
    args = parse_args()

    if args.check:
        cmd_check()
    elif args.ingest:
        cmd_ingest(
            reset=args.reset,
            skip_summary=args.skip_summary,
            force_summary=args.force_summary,
        )
    elif args.summarize:
        cmd_summarize(force=args.force_summary)
    elif args.query:
        cmd_query(args.query)
    elif args.generate_test_queries:
        cmd_generate_test_queries()
    else:
        cmd_voice(audio_input=args.audio)


if __name__ == "__main__":
    main()
