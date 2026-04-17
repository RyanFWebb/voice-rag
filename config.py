"""
config.py — Central configuration for voice_rag.
Edit this file to change models, paths, and tuning parameters.
"""

import platform
import shutil

def _detect_gpu() -> tuple[str, str]:
    """Return (device, compute_type) — prefer CUDA if available."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"

_WHISPER_DEVICE, _WHISPER_COMPUTE = _detect_gpu()

# Models
WHISPER_MODEL_SIZE  = "base"              # tiny | base | small | medium | large
WHISPER_COMPUTE     = _WHISPER_COMPUTE    # int8 (CPU) | float16 (GPU)
WHISPER_DEVICE      = _WHISPER_DEVICE     # auto-detected: cpu or cuda
WHISPER_LANGUAGE    = "en"                # skip language detection → faster + more accurate

GEN_MODEL   = "gemma3:4b"            # any model pulled in Ollama
EMBED_MODEL = "mxbai-embed-large"    # must be pulled in Ollama

TTS_VOICE   = "af_heart"             # Kokoro voice ID
TTS_SPEED   = 1.0
TTS_SAMPLE_RATE = 24000

# Paths
DOCUMENTS_DIR   = "./documents"
CHROMA_PATH     = "./chroma_db"
TEST_QUERY_DIR  = "./test_queries"

# ChromaDB
COLLECTION_NAME = "rag_documents"

# Chunking
CHUNK_SIZE    = 1000   # characters (~200 words)
CHUNK_OVERLAP = 150

# Retrieval
N_RESULTS   = 3
N_RETRIEVE  = 20       # initial candidates before reranking
EMBED_BATCH = 32       # chunks per Ollama embed request

# Reranking
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Chunk expansion — fetch N neighbors on each side of a retrieved chunk
NEIGHBOR_WINDOW = 1   # 1 = include 1 chunk before + 1 after each hit

# Recording
RECORD_SECONDS = 7
RECORD_SAMPLE_RATE = 16000

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT  = 180

# Platform helpers
IS_WINDOWS = platform.system() == "Windows"
WINDOWS_ESPEAK_DIR = r"C:\Program Files\eSpeak NG"
