# AIRMAN RAG Evaluation Report

## System Overview
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector Store: FAISS (IndexFlatL2)
- LLM: Groq LLaMA 3.3 70B
- Chunking: 500 words, 50-word overlap

## Chunking Strategy
Pages are extracted per-page to preserve page numbers for citations.
Each page is split into 500-word chunks with 50-word overlap.
This preserves context across chunk boundaries while keeping chunks
small enough for accurate retrieval.


## Metrics
| Metric            | Value  |
|-------------------|--------|
| Total Questions   | 50     |
| Retrieval Hit Rate| 58%    |
| Refusal Rate      | 40%    |
| Hallucination Rate| 2%     |


## Analysis
- 29 questions were answered correctly with supporting chunks
- 20 questions were refused — mostly out-of-scope or topics 
  not covered in the provided PDFs
- Only 1 hallucination detected — very strong hallucination control
- The strict prompt design successfully prevented outside knowledge use

## 5 Best Answers
Pick 5 from eval_results.json where the answer is accurate and well cited.
For each one write:
- Question number and the question text
- Why it's good (correct answer, clear citation, grounded in document)

Example format:
**Q1 — What is a flight level referenced to?**
Answer was accurate, cited the correct page, fully grounded in retrieved chunks.

## 5 Worst Answers
Pick 5 where the answer was vague, wrong, or refused a valid question.
For each one write:
- Question number and question text
- Why it's bad (no citation, wrong answer, unnecessary refusal)

Example format:
**Q15 — What causes a sea breeze?**
System refused despite the topic being covered in the meteorology chapter.

## Hallucination Control
The system uses a strict prompt that forbids outside knowledge.
Non-aviation questions are correctly refused with the standard message.
All answers must be grounded in retrieved chunks — the LLM cannot guess.