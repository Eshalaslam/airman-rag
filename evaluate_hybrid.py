import json
import time
from query import ask
from hybrid_query import ask_hybrid

REFUSAL = "This information is not available in the provided document(s)."

with open("questions.json") as f:
    questions = json.load(f)

def run_eval(ask_fn, label):
    results = []
    print(f"\nRunning {label} ({len(questions)} questions)...\n")
    for q in questions:
        print(f"Q{q['id']}: {q['question'][:60]}...")
        try:
            result = ask_fn(q["question"], debug=True)
            answer = result["answer"]
            chunks = result.get("retrieved_chunks", [])
            refused = REFUSAL.lower() in answer.lower()
            results.append({
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "answer": answer,
                "citations": result["citations"],
                "refused": refused,
                "chunk_count": len(chunks) if chunks else 0,
                "retrieval_hit": (len(chunks) > 0 if chunks else False) and not refused,
            })
            time.sleep(1)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "id": q["id"], "type": q["type"],
                "question": q["question"],
                "answer": f"ERROR: {e}",
                "citations": [], "refused": False,
                "chunk_count": 0, "retrieval_hit": False
            })
    return results

# Run both
vector_results = run_eval(ask, "Vector-Only (Level 1)")
hybrid_results = run_eval(ask_hybrid, "Hybrid (Level 2)")

# Save
with open("eval_vector.json", "w") as f:
    json.dump(vector_results, f, indent=2)
with open("eval_hybrid.json", "w") as f:
    json.dump(hybrid_results, f, indent=2)

# Compare
def metrics(results, label):
    total = len(results)
    hits = sum(1 for r in results if r["retrieval_hit"])
    refusals = sum(1 for r in results if r["refused"])
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"Retrieval Hit Rate : {hits}/{total} ({hits/total*100:.1f}%)")
    print(f"Refusal Rate       : {refusals}/{total} ({refusals/total*100:.1f}%)")
    print(f"Hallucination Rate : {total-hits-refusals}/{total} ({(total-hits-refusals)/total*100:.1f}%)")

metrics(vector_results, "VECTOR ONLY (Level 1 Baseline)")
metrics(hybrid_results, "HYBRID BM25 + VECTOR + RERANKER (Level 2)")
print("\nResults saved to eval_vector.json and eval_hybrid.json")