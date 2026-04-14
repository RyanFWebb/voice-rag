# Voice RAG Assistant

A fully local, voice-in → voice-out RAG system.  
**Pipeline:** Mic → Whisper STT → ChromaDB retrieval → Ollama LLM → Kokoro TTS → Speaker

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
ollama pull nomic-embed-text
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

### Ingest documents

Drop `.pdf` or `.txt` files into `documents/`, then:

```bash
python main.py --ingest
```

To wipe the vector store and start fresh:

```bash
python main.py --ingest --reset
```

### Ask a text question (no microphone needed)

```bash
python main.py --query "Who is Juliet and how is she introduced?"
```

### Live voice loop

```bash
python main.py
```

Speak your question when prompted. Press `Ctrl-C` to quit.

### Replay a saved query file

```bash
python main.py --audio test_queries/q01_juliet_identity.wav
```

### Generate test query WAV files

```bash
python main.py --generate-test-queries
```

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | STT model size (tiny/base/small/medium/large) |
| `GEN_MODEL` | `gemma3:4b` | Ollama generation model |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `TTS_VOICE` | `af_heart` | Kokoro voice ID |
| `CHUNK_SIZE` | `1000` | Characters per chunk (~200 words) |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `N_RESULTS` | `5` | Chunks retrieved per query |
| `RECORD_SECONDS` | `7` | Mic recording duration |

---

## Project Structure

```
voice_rag/
├── main.py           # CLI entry point
├── config.py         # All constants — edit this to tune the system
├── ingest.py         # PDF/TXT loading, cleaning, chunking
├── vector_store.py   # ChromaDB setup, upsert, query
├── llm.py            # Ollama generate + embed wrappers
├── speech.py         # Whisper STT + Kokoro TTS
├── rag.py            # Retrieval + generation + prompt template
├── documents/        # Add your PDFs and TXTs here
├── chroma_db/        # Auto-created on first ingest (gitignored)
├── test_queries/     # Auto-created by --generate-test-queries
└── requirements.txt
```

---

## Notes

- `chroma_db/` and `documents/` are gitignored. The vector store is fully reproducible from your documents by running `--ingest`.
- Tested on Windows (Python 3.12) and macOS (Python 3.11+).
- All inference is local — no API keys required.
