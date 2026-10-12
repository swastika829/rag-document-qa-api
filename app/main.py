import os
import tempfile
import time

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from .rag import ingest_document, answer_question

app = FastAPI(title="RAG Document Q&A API")


class Question(BaseModel):
    question: str


@app.post("/upload")
def upload(file: UploadFile = File(...)):
  
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        path = tmp.name

    num_chunks = ingest_document(path, file.filename)
    os.remove(path)
    return {"filename": file.filename, "chunks_stored": num_chunks}


@app.post("/query")
def query(body: Question):
    """Ask a question. Returns the answer, the sources used, and the time taken."""
    start = time.perf_counter()
    result = answer_question(body.question)
    result["time_ms"] = round((time.perf_counter() - start) * 1000)
    return result
