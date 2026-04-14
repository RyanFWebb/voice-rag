"""
config.py — Central configuration for voice_rag.
Edit this file to change models, paths, and tuning parameters.
"""

import platform

# Models
WHISPER_MODEL_SIZE  = "base"          # tiny | base | small | medium | large
WHISPER_COMPUTE     = "int8"          # int8 (CPU) | float16 (GPU)
WHISPER_DEVICE      = "cpu"

GEN_MODEL   = "gemma3:4b"            # any model pulled in Ollama
EMBED_MODEL = "nomic-embed-text"     # must be pulled in Ollama

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
N_RESULTS   = 5
EMBED_BATCH = 32       # chunks per Ollama embed request

# Recording
RECORD_SECONDS = 7
RECORD_SAMPLE_RATE = 16000

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT  = 180

# Platform helpers
IS_WINDOWS = platform.system() == "Windows"
WINDOWS_ESPEAK_DIR = r"C:\Program Files\eSpeak NG"
