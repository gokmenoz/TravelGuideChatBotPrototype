import pickle
from typing import Dict, List, Optional

import faiss
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from constants import OPENWEATHER_API_KEY
from utils import (
    build_prompt_with_history,
    build_rag_prompt,
    call_claude_stream,
    extract_location,
    get_weather,
    maybe_enrich_prompt,
    retrieve,
    visa_info,
)

"""
uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload

curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What can I do for 3 days in Lisbon?"}'
"""

app = FastAPI()

# Load FAISS + Chunks
index, chunks = faiss.read_index("faiss_index/index.faiss"), pickle.load(
    open("faiss_index/chunks.pkl", "rb")
)
embedder = SentenceTransformer("BAAI/bge-base-en")


class ChatRequest(BaseModel):
    query: str
    location: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = []  # [{role: "user", content: "..."}]


@app.get("/")
def root():
    return {"message": "✅ Travel Chatbot API is running."}


@app.get("/weather")
def weather(city: str):
    return get_weather(city)


@app.get("/visa")
def visa(country: str):
    return visa_info(country)


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.query
    location = req.location or extract_location(question)
    messages = build_prompt_with_history(req.history, question)

    if not location:
        return {"response": "❗ Please mention a destination in your question."}

    location_chunks = [
        c for c in chunks if c.get("location", "").lower() == location.lower()
    ]

    # If no relevant chunks → fallback to Claude with messages
    if not location_chunks:
        prompt = f"You are a helpful travel assistant. Answer this:\n\n{question}"
        stream = call_claude_stream(prompt, messages_override=messages)
        return StreamingResponse(content=stream, media_type="text/plain")

    docs = retrieve(question, location_chunks, embedder)

    if not docs:
        prompt = f"You are a helpful travel assistant. Answer this:\n\n{question}"
        stream = call_claude_stream(prompt, messages_override=messages)
        return StreamingResponse(content=stream, media_type="text/plain")

    context = "\n---\n".join(docs)
    external_context = maybe_enrich_prompt(question, location)
    full_context = (
        external_context + "\n---\n" + context if external_context else context
    )
    rag_prompt = build_rag_prompt(full_context, question)
    stream = call_claude_stream(rag_prompt, messages_override=messages)

    return StreamingResponse(content=stream, media_type="text/plain")
