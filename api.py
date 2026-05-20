from fastapi import FastAPI
from hybrid_query import ask_hybrid
from pydantic import BaseModel
from query import ask
import subprocess
import sys

app = FastAPI(title="AIRMAN Aviation RAG API")

class AskRequest(BaseModel):
    question: str
    debug: bool = False

@app.get("/health")
def health():
    return {"status": "ok", "message": "AIRMAN RAG API is running"}

@app.post("/ingest")
def ingest():
    try:
        result = subprocess.run(
            [sys.executable, "ingest.py"],
            capture_output=True, text=True
        )
        return {"status": "success", "output": result.stdout}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/ask")
def ask_question(request: AskRequest):
    result = ask(request.question, debug=request.debug)
    return result
@app.post("/ask/hybrid")
def ask_hybrid_question(request: AskRequest):
    result = ask_hybrid(request.question, debug=request.debug)
    return result