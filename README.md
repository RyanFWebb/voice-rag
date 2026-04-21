# Voice RAG Assistant

A fully local, voice-in → voice-out RAG system.
**Pipeline:** Mic → Whisper STT → ChromaDB retrieval → Cross-encoder rerank → Neighbor expansion → Ollama LLM → Kokoro TTS → Speaker

**Features**
- Two-stage retrieval: ChromaDB cosine similarity → cross-encoder reranking
- Neighbor chunk expansion: widens each retrieved hit with its adjacent chunks for better context
- Map-reduce document summaries built at ingest time; "summarize..." / "what is this about" queries are auto-routed to the pre-computed summary instead of RAG
- Inline `[N]` citations in every answer with a compact source list for fact-checking (citations stripped before TTS so they aren't spoken)

---

## Quick Start

### 1. System dependencies

**macOS**
```bash
brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate
brew install espeak-ng portaudio ffmpeg
```
Whenever loading your directory, run `source .venv/bin/activate` to ensure you are running the correct dependencies.

**Windows**
```
winget install ffmpeg
# Download eSpeak NG installer from https://github.com/espeak-ng/espeak-ng/releases
# Install to default path: C:\Program Files\eSpeak NG
```

**Linux**
```bash
sudo apt install espeak-ng portaudio19-dev ffmpeg
```

### 2. Ollama models

```bash
ollama pull gemma3:4b
ollama pull mxbai-embed-large
```

### 3. Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify everything is working

```bash
python main.py --check
```

---

## Usage

### User Interface
Run the following:
```bash
python app.py
```
*Connect to http://localhost:7860*

### Command Line
#### Ingest documents

Drop `.pdf` or `.txt` files into `documents/`, then:

```bash
python main.py --ingest
```

Ingestion also builds a map-reduce summary per document (cached under `summaries/`). Options:

```bash
python main.py --ingest --reset            # wipe ChromaDB first
python main.py --ingest --skip-summary     # chunk/embed only, no summaries
```

#### (Re)build summaries without re-ingesting

```bash
python main.py --summarize                 # uses cached summaries if present
python main.py --summarize --force-summary # ignore cache, rebuild every summary
```

#### Ask a text question (no microphone needed)

```bash
python main.py --query "Who is the main character and how are they introduced?"
python main.py --query "Summarize the book"      # auto-routed to the cached summary
```

Answers include inline `[N]` citations; a numbered source list (only sources actually cited) is printed underneath the answer for quick fact-checking.

#### Live voice loop

```bash
python main.py
```

Speak your question when prompted. Press `Ctrl-C` to quit.

#### Replay a saved query file

```bash
python main.py --audio test_queries/q01_character_intro.wav
```

#### Generate test query WAV files

```bash
python main.py --generate-test-queries
```

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | STT model size (tiny/base/small/medium/large) |
| `WHISPER_DEVICE` | auto-detected | `cuda` if GPU available, else `cpu` |
| `WHISPER_LANGUAGE` | `en` | Language code (skips detection for speed) |
| `GEN_MODEL` | `gemma3:4b` | Ollama generation model |
| `EMBED_MODEL` | `mxbai-embed-large` | Ollama embedding model |
| `TTS_VOICE` | `af_heart` | Kokoro voice ID |
| `CHUNK_SIZE` | `1000` | Characters per chunk (~200 words) |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `N_RETRIEVE` | `20` | Initial candidates from ChromaDB |
| `N_RESULTS` | `3` | Final chunks kept after reranking |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `NEIGHBOR_WINDOW` | `1` | Chunks to fetch on each side of a hit |
| `SUMMARY_MAP_BATCH` | `5` | Chunks per section summary (map stage) |
| `SUMMARY_REDUCE_BATCH` | `40` | Section summaries folded per reduce call |
| `SUMMARY_MODEL` | `GEN_MODEL` | Model used for map-reduce summarization |
| `SUMMARY_CTX` | `8192` | `num_ctx` for reduce calls |
| `RECORD_SECONDS` | `7` | Mic recording duration |

---

## Project Structure

```
voice_rag/
├── main.py           # CLI entry point
├── config.py         # All constants — edit this to tune the system
├── ingest.py         # PDF/TXT loading, cleaning, chunking
├── vector_store.py   # ChromaDB setup, upsert, query, neighbor expansion
├── llm.py            # Ollama generate + embed wrappers
├── speech.py         # Whisper STT + Kokoro TTS
├── rag.py            # Retrieval + reranking + generation + inline citations
├── summarize.py      # Map-reduce document summaries + summary-query routing
├── documents/        # Add your PDFs and TXTs here
├── chroma_db/        # Auto-created on first ingest (gitignored)
├── summaries/        # Cached per-document summaries (gitignored)
├── test_queries/     # Auto-created by --generate-test-queries
└── requirements.txt
```

---

## Notes

- `chroma_db/`, `summaries/`, and `documents/` are gitignored. The vector store and summaries are fully reproducible from your documents by running `--ingest`.
- Tested on Windows (Python 3.12) and macOS (Python 3.11+).
- All inference is local — no API keys required.
