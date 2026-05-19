import os
import fitz  # PyMuPDF
import faiss
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

DOCS_DIR = "docs"
INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"text": text, "page": page_num, "source": Path(pdf_path).name})
    return pages

def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    for page_data in pages:
        text = page_data["text"]
        words = text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "text": chunk_text,
                "page": page_data["page"],
                "source": page_data["source"]
            })
            start += chunk_size - overlap
    return chunks

def build_index():
    print("Loading PDFs...")
    all_chunks = []
    pdf_files = list(Path(DOCS_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in '{DOCS_DIR}' folder!")
        return

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        pages = extract_text_from_pdf(str(pdf_path))
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks from {len(pages)} pages")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Generating embeddings (this takes 1-2 min)...")
    
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\n✅ Done! Index saved. {len(all_chunks)} chunks indexed.")

if __name__ == "__main__":
    build_index()