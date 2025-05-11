from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.utils import (
    build_prompt_with_history,
    build_rag_prompt,
    call_claude_stream,
    extract_location,
    get_weather,
    maybe_enrich_prompt,
    retrieve,
    visa_info,
    load_index,
    embedder,
)

"""
uvicorn src.api:app --host 0.0.0.0 --port 8001 --reload

curl -X POST http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What can I do for 3 days in Lisbon?"}'
"""

app = FastAPI(title="Travel Guide Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index and chunks
index, chunks = load_index()

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None

class ChatResponse(BaseModel):
    response: str
    location: Optional[str] = None

@app.get("/")
def root():
    return {"message": "✅ Travel Chatbot API is running."}

@app.get("/weather")
def weather(city: str):
    return get_weather(city)

@app.get("/visa")
def visa(country: str):
    return visa_info(country)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Extract location from message
        location = extract_location(request.message)
        
        # Enrich prompt with additional context
        enriched_context = maybe_enrich_prompt(request.message, location) if location else ""
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve(request.message, chunks, embedder)
        context = "\n".join(relevant_chunks)
        
        # Build RAG prompt
        rag_prompt = build_rag_prompt(context, request.message)
        
        # Add enriched context if available
        if enriched_context:
            rag_prompt = f"{enriched_context}\n\n{rag_prompt}"
        
        # Build messages with history
        messages = build_prompt_with_history(request.history or [], rag_prompt)
        
        # Get streaming response from Claude
        response_chunks = []
        for chunk in call_claude_stream(messages_override=messages):
            response_chunks.append(chunk)
        
        return ChatResponse(
            response="".join(response_chunks),
            location=location
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
