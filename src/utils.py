import json
import os
import pickle
import random
import time
from typing import Dict, List, Optional, Union, Generator, Any

import boto3
import botocore.exceptions
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from src.constants import OPENWEATHER_API_KEY

# Constants
FAISS_INDEX_DIR = "faiss_index"
MAX_RETRIES = 5
BASE_DELAY = 2
TOP_K_RETRIEVAL = 5
CHUNK_MAX_TOKENS = 300
CHUNK_OVERLAP = 50

# Global variables
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedder = SentenceTransformer("BAAI/bge-base-en")

# AWS Bedrock setup
session = boto3.Session(profile_name="ogokmen_bedrock")
bedrock = session.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# Global index and chunks
index = None
chunks = []

def initialize_index():
    """Initialize or load the FAISS index and chunks."""
    global index, chunks
    if os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss")):
        index, chunks = load_index(FAISS_INDEX_DIR)
    else:
        index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
        chunks = []

def update_index(location: str) -> None:
    """
    Update the FAISS index with content for a new location.
    
    Args:
        location: Location to add to the index
    """
    global index, chunks
    
    # Initialize index if not exists
    if index is None:
        initialize_index()
    
    # Skip if location already indexed
    if any(c["location"].lower() == location.lower() for c in chunks):
        return
    
    # Get and process content
    content = get_wikivoyage_page(location)
    if not content:
        return
        
    new_chunks = chunk_text(content, location)
    if not new_chunks:
        return
    
    # Build new index
    new_index, _ = build_faiss_index(new_chunks, embedder)
    
    # Update global index and chunks
    if index.ntotal == 0:
        index = new_index
    else:
        index.add(new_index.reconstruct_n(0, new_index.ntotal))
    chunks.extend(new_chunks)
    
    # Save updated index
    save_index(index, new_chunks)


def get_weather(city: str) -> Dict[str, Union[str, float]]:
    """
    Fetch current weather information for a city using OpenWeather API.
    
    Args:
        city: Name of the city to get weather for
        
    Returns:
        Dictionary containing weather information or error message
    """
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}

    try:
        response = requests.get(url, params=params, timeout=10).json()
        if response.get("cod") != 200:
            return {"error": f"Could not fetch weather for {city}."}

        weather = response["weather"][0]["description"]
        temp = response["main"]["temp"]
        feels_like = response["main"]["feels_like"]

        return {
            "city": city,
            "weather": weather,
            "temperature_c": temp,
            "feels_like_c": feels_like,
        }
    except (requests.RequestException, KeyError, ValueError) as e:
        return {"error": f"Error fetching weather for {city}: {str(e)}"}


def visa_info(country: str) -> Dict[str, str]:
    """
    Fetch visa and country information using REST Countries API.
    
    Args:
        country: Name of the country to get visa info for
        
    Returns:
        Dictionary containing country and visa information or error message
    """
    url = f"https://restcountries.com/v3.1/name/{country}"
    
    try:
        response = requests.get(url, timeout=10).json()

        if isinstance(response, list):
            data = response[0]
            name = data.get("name", {}).get("common", country)
            region = data.get("region", "Unknown")
            subregion = data.get("subregion", "Unknown")
            capital = data.get("capital", ["Unknown"])[0]
            population = data.get("population", "Unknown")

            return {
                "country": name,
                "region": region,
                "subregion": subregion,
                "capital": capital,
                "population": population,
                "visa_note": "❗Visa requirements vary by passport. Check https://apply.joinsherpa.com/ or your embassy.",
            }

        return {"error": f"Could not find visa info for {country}"}
    except (requests.RequestException, KeyError, ValueError) as e:
        return {"error": f"Error fetching visa info for {country}: {str(e)}"}


def maybe_enrich_prompt(question: str, location: str) -> str:
    """
    Enrich the prompt with additional context like weather and visa information.
    
    Args:
        question: User's question
        location: Location mentioned in the question
        
    Returns:
        Enriched prompt with additional context
    """
    prompt_parts = []

    # Weather
    if "weather" in question.lower():
        weather = get_weather(location)
        if "error" not in weather:
            prompt_parts.append(
                f"The current weather in {location} is {weather['temperature_c']}°C, "
                f"feels like {weather['feels_like_c']}°C, with {weather['weather']}."
            )

    # Visa
    if "visa" in question.lower():
        visa = visa_info(location)
        if "error" not in visa:
            prompt_parts.append(
                f"{visa['visa_note']} {location} is in {visa['subregion']} with capital {visa['capital']}."
            )

    return "\n".join(prompt_parts) if prompt_parts else ""


def get_wikivoyage_page(title: str, lang: str = "en") -> str:
    """
    Fetch clean text of a Wikivoyage page by title.
    
    Args:
        title: Title of the Wikivoyage page
        lang: Language code (default: "en")
        
    Returns:
        Extracted text content from the page
    """
    url = f"https://{lang}.wikivoyage.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        page = next(iter(data["query"]["pages"].values()))
        return page.get("extract", "")
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Error fetching Wikivoyage page for {title}: {str(e)}")
        return ""


def chunk_text(
    text: str, location: str, max_tokens: int = CHUNK_MAX_TOKENS, overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text: Text to split into chunks
        location: Location associated with the text
        max_tokens: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks with location metadata
    """
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokenizer.convert_tokens_to_string(tokens[start:end])
        chunks.append({"text": chunk, "location": location})
        start += max_tokens - overlap
    return chunks


def build_faiss_index(text_chunks: List[Dict[str, str]], embedder: SentenceTransformer) -> tuple[faiss.Index, List[Dict[str, str]]]:
    """
    Build a FAISS index from text chunks.
    
    Args:
        text_chunks: List of text chunks with metadata
        embedder: Sentence transformer model for encoding
        
    Returns:
        Tuple of (FAISS index, text chunks)
    """
    if not text_chunks:
        raise ValueError("No text chunks provided to build FAISS index.")

    texts = [c["text"] for c in text_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, text_chunks


def save_index(index: faiss.Index, chunks: List[Dict[str, str]], outdir: str = FAISS_INDEX_DIR) -> None:
    """
    Save FAISS index and chunks to disk.
    
    Args:
        index: FAISS index to save
        chunks: Text chunks to save
        outdir: Output directory
    """
    os.makedirs(outdir, exist_ok=True)
    chunks_path = os.path.join(outdir, "chunks.pkl")

    # Load existing chunks if available
    if os.path.exists(chunks_path):
        with open(chunks_path, "rb") as f:
            existing_chunks = pickle.load(f)
    else:
        existing_chunks = []

    # Append new chunks
    all_chunks = existing_chunks + chunks

    # Recompute embeddings and rebuild FAISS index
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # Save updated index and chunks
    faiss.write_index(index, os.path.join(outdir, "index.faiss"))
    with open(chunks_path, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ Appended {len(chunks)} chunks. Total: {len(all_chunks)}")


def load_index(outdir: str = FAISS_INDEX_DIR) -> tuple[faiss.Index, List[Dict[str, str]]]:
    """
    Load FAISS index and chunks from disk.
    
    Args:
        outdir: Directory containing index files
        
    Returns:
        Tuple of (FAISS index, text chunks)
    """
    index = faiss.read_index(f"{outdir}/index.faiss")
    with open(f"{outdir}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def retrieve(
    query: str, chunks: List[Dict[str, str]], embedder: SentenceTransformer, top_k: int = TOP_K_RETRIEVAL
) -> List[str]:
    """
    Retrieve most relevant chunks for a query.
    
    Args:
        query: Search query
        chunks: Available text chunks
        embedder: Sentence transformer model for encoding
        top_k: Number of chunks to retrieve
        
    Returns:
        List of retrieved text chunks
    """
    if not chunks:
        return []

    # Initialize index if not exists
    global index
    if index is None:
        initialize_index()

    q_emb = embedder.encode([query])
    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = embedder.encode(chunk_texts)

    local_index = faiss.IndexFlatL2(chunk_embs.shape[1])
    local_index.add(np.array(chunk_embs))

    D, I = local_index.search(np.array(q_emb), top_k)

    return [chunk_texts[i] for i in I[0] if i < len(chunk_texts)]


def build_rag_prompt(context: str, question: str) -> str:
    """
    Build a RAG prompt with context and question.
    
    Args:
        context: Retrieved context
        question: User's question
        
    Returns:
        Formatted prompt for the LLM
    """
    return f"""
You are a helpful travel assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""


def call_claude_stream(
    prompt: Optional[str] = None,
    messages_override: Optional[List[Dict[str, str]]] = None,
    retries: int = MAX_RETRIES,
    base_delay: int = BASE_DELAY,
) -> Generator[str, None, None]:
    """
    Call Claude API with streaming response.
    
    Args:
        prompt: Text prompt for Claude
        messages_override: Override default message format
        retries: Number of retry attempts
        base_delay: Base delay between retries
        
    Returns:
        Generator yielding response chunks
        
    Raises:
        RuntimeError: If all retries fail
    """
    messages = messages_override or [{"role": "user", "content": prompt}]

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    for attempt in range(retries):
        try:
            response = bedrock.invoke_model_with_response_stream(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            def stream_generator():
                for event in response["body"]:
                    if "chunk" in event:
                        chunk_data = json.loads(event["chunk"]["bytes"])
                        if chunk_data.get("type") == "content_block_delta":
                            delta = chunk_data.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                yield text

            return stream_generator()

        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                wait = base_delay * (2**attempt) + random.uniform(0, 1)
                print(f"⏳ Throttled. Retrying in {wait:.2f}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            time.sleep(1)

    raise RuntimeError("❌ Claude streaming call failed after retries")


def build_prompt_with_history(history: List[Dict[str, str]], question: str) -> List[Dict[str, str]]:
    """
    Build a prompt with conversation history.
    
    Args:
        history: List of previous messages
        question: Current question
        
    Returns:
        List of messages including history and current question
    """
    messages = history + [{"role": "user", "content": question}]
    return messages


def extract_location(text: str) -> Optional[str]:
    """
    Extract location from text using simple rule-based approach.
    
    Args:
        text: Input text
        
    Returns:
        Extracted location or None if not found
    """
    if text.lower() == text:
        text = text.capitalize()
    
    # Simple location detection for now
    # TODO: Consider using a proper NER model if needed
    words = text.split()
    for word in words:
        if word[0].isupper() and len(word) > 1:
            return word.strip()
    return None


def tokenize_dataset(dataset: Any, tokenizer: AutoTokenizer, max_length: int = 1024) -> Any:
    """
    Tokenize a dataset for training.
    
    Args:
        dataset: Dataset to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        Tokenized dataset
    """
    def tokenize(example):
        tokenized = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = tokenized["input_ids"][:]
        return tokenized

    return dataset.map(tokenize, batched=False)
