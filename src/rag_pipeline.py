from src.pdf_loader import read_pdf
from src.chunker import chunk_text
from src.embedder import Embedder
from src.vector_store import VectorStore
import os
import uuid

BASE_EMBEDDINGS_PATH = "embeddings"

def get_session_path(session_id):
    return os.path.join(BASE_EMBEDDINGS_PATH, session_id)

def build_vector_store_from_pdfs(pdf_paths, session_id):
    all_text = ""

    for path in pdf_paths:
        all_text += read_pdf(path)

    chunks = chunk_text(all_text)

    embedder = Embedder()
    embeddings = embedder.embed_texts(chunks)

    vector_store = VectorStore(dimension=embeddings.shape[1])
    vector_store.add(embeddings, chunks)

    session_path = get_session_path(session_id)
    os.makedirs(session_path, exist_ok=True)
    vector_store.save(session_path)

def load_vector_store(session_id):
    embedder = Embedder()
    vector_store = VectorStore(dimension=384)

    session_path = get_session_path(session_id)
    if not os.path.exists(session_path):
        raise FileNotFoundError("No vector store for this session")

    vector_store.load(session_path)
    return embedder, vector_store
