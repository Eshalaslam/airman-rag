import json
import time
from query import ask

REFUSAL = "This information is not available in the provided document(s)."

with open("questions.json") as f:
    questions = json.load(f)

results = []
print(f"Running {len(questions)} questions...\n")

for q in questions:
    print(f"Q{q['id']}: {q['question'][:60]}...")
    try:
        result = ask(q["question"], debug=True)
        answer = result["answer"]
        chunks = result.get("retrieved_chunks", [])
        refused = REFUSAL.lower() in answer.lower()
        has_chunks = len(chunks) > 0

        results.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "answer": answer,
            "citations": result["citations"],
            "refused": refused,
            "chunk_count": len(chunks),
            "retrieval_hit": has_chunks and not refused,
        })
        time.sleep(3)  # avoid rate limiting
    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({
            "id": q["id"], "type": q["type"],
            "question": q["question"],
            "answer": f"ERROR: {e}",
            "citations": [], "refused": False,
            "chunk_count": 0, "retrieval_hit": False
        })

# Save results
with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Calculate metrics
total = len(results)
hits = sum(1 for r in results if r["retrieval_hit"])
refusals = sum(1 for r in results if r["refused"])
hit_rate = hits / total * 100

print(f"\n{'='*50}")
print(f"EVALUATION SUMMARY")
print(f"{'='*50}")
print(f"Total Questions : {total}")
print(f"Retrieval Hits  : {hits} ({hit_rate:.1f}%)")
print(f"Refusals        : {refusals}")
print(f"Hallucination   : Questions answered without chunks: {total - hits - refusals}")
print(f"{'='*50}")
print("\nResults saved to eval_results.json")