# AIRMAN Aviation RAG Chatbot

An AI chatbot that answers aviation questions strictly from
provided documents using RAG (Retrieval-Augmented Generation).
No hallucinations — answers are always grounded in source material.

## Tech Stack
- PDF Parsing: PyMuPDF
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Vector Store: FAISS
- LLM: Groq LLaMA 3.3 70B
- API: FastAPI

## Setup

1. Clone the repo
2. Create virtual environment:
   python -m venv venv
   venv\Scripts\activate
3. Install dependencies:
   pip install -r requirements.txt
4. Copy .env.example to .env and add your Groq API key
5. Add aviation PDFs to the docs/ folder

## Running the Project

# Step 1 — Ingest documents
python ingest.py

# Step 2 — Start the API
uvicorn api:app --reload

# Step 3 — Open the interactive docs
Visit http://localhost:8000/docs in your browser

## API Endpoints

| Endpoint       | Method | Description                        |
|----------------|--------|------------------------------------|
| /health        | GET    | Check if API is running            |
| /ingest        | POST   | Re-run ingestion on docs/ folder   |
| /ask           | POST   | Ask a question, get answer+citation|

## Evaluation

python evaluate.py

Results saved to eval_results.json
Summary printed to terminal

## Project Structure

airman-rag/
├── docs/              # Aviation PDF documents
├── ingest.py          # PDF parsing + FAISS index builder
├── query.py           # RAG retrieval + LLM answer generation
├── api.py             # FastAPI endpoints
├── evaluate.py        # Automated evaluation script
├── questions.json     # 50 test questions
├── eval_results.json  # Evaluation output
├── report.md          # Evaluation report
├── .env.example       # API key template
└── README.md          # This file