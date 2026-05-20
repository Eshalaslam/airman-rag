import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

INDEX_FILE = "faiss_index.bin"
CHUNKS_FILE = "chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 10        # fetch more candidates first
RERANK_TOP = 5    # keep top 5 after reranking
REFUSAL = "This information is not available in the provided document(s)."

# Load models
print("Loading models...")
embedder = SentenceTransformer(EMBED_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)

# Load FAISS index and chunks
index = faiss.read_index(INDEX_FILE)
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

# Build BM25 index from chunks
print("Building BM25 index...")
tokenized_corpus = [c["text"].lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized_corpus)

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def vector_search(question, top_k=TOP_K):
    q_emb = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks):
            results.append({**chunks[idx], "score": float(dist), "source_type": "vector"})
    return results

def bm25_search(question, top_k=TOP_K):
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({**chunks[idx], "score": float(scores[idx]), "source_type": "bm25"})
    return results

def hybrid_retrieve(question, top_k=TOP_K, rerank_top=RERANK_TOP):
    # Get candidates from both
    vector_results = vector_search(question, top_k)
    bm25_results = bm25_search(question, top_k)

    # Merge and deduplicate by text
    seen = set()
    combined = []
    for r in vector_results + bm25_results:
        key = r["text"][:100]
        if key not in seen:
            seen.add(key)
            combined.append(r)

    # Rerank using cross-encoder
    if not combined:
        return []

    pairs = [(question, c["text"]) for c in combined]
    rerank_scores = reranker.predict(pairs)

    for i, score in enumerate(rerank_scores):
        combined[i]["rerank_score"] = float(score)

    # Sort by rerank score
    reranked = sorted(combined, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:rerank_top]

def ask_hybrid(question, debug=False):
    retrieved = hybrid_retrieve(question)

    if not retrieved:
        return {
            "question": question,
            "answer": REFUSAL,
            "citations": [],
            "retrieved_chunks": [] if debug else None,
            "retrieval_method": "hybrid"
        }

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

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.1
    )

    answer = response.choices[0].message.content.strip()
    citations = list({f"{c['source']} (Page {c['page']})" for c in retrieved})

    result = {
        "question": question,
        "answer": answer,
        "citations": citations,
        "retrieval_method": "hybrid (BM25 + Vector + Reranker)"
    }
    if debug:
        result["retrieved_chunks"] = retrieved

    return result

if __name__ == "__main__":
    q = input("Ask a question (hybrid mode): ")
    result = ask_hybrid(q, debug=True)
    print("\n--- ANSWER ---")
    print(result["answer"])
    print("\n--- CITATIONS ---")
    for c in result["citations"]:
        print(" •", c)
    print("\n--- METHOD ---")
    print(result["retrieval_method"])