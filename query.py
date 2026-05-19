import os
from urllib import response
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq 
from dotenv import load_dotenv

load_dotenv()

INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
REFUSAL = "This information is not available in the provided document(s)."

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer(EMBED_MODEL)

index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

def retrieve(question, top_k=TOP_K):
    q_embedding = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_embedding, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append({**chunks[idx], "score": float(dist)})
    return results

def ask(question, debug=False):
    retrieved = retrieve(question)
    context = "\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in retrieved
    ])

    prompt = f"""You are an aviation expert assistant. Answer ONLY using the context below.
If the answer is not found in the context, respond exactly with:
"This information is not available in the provided document(s)."

Do NOT use any outside knowledge. Do NOT guess.

Context:
{context}

Question: {question}

Answer (cite the source document and page number):"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
)
    answer = response.choices[0].message.content.strip()
    citations = list({f"{c['source']} (Page {c['page']})" for c in retrieved})

    result = {
        "question": question,
        "answer": answer,
        "citations": citations,
    }
    if debug:
        result["retrieved_chunks"] = retrieved

    return result

if __name__ == "__main__":
    q = input("Ask a question: ")
    result = ask(q, debug=True)
    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- CITATIONS ---")
    for c in result["citations"]:
        print(" •", c)