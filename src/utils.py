import json
import os
import pickle
import random
import threading
import time
from typing import Dict, List

import boto3
import botocore.exceptions
import faiss
import numpy as np
import requests
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer, pipeline)

from constants import OPENWEATHER_API_KEY


def get_weather(city: str):
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}

    response = requests.get(url, params=params).json()
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


def visa_info(country: str):
    url = f"https://restcountries.com/v3.1/name/{country}"
    response = requests.get(url).json()

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


def maybe_enrich_prompt(question: str, location: str) -> str:
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

    # Budget (if added later)
    # ...

    return "\n".join(prompt_parts) if prompt_parts else ""


def get_wikivoyage_page(title: str, lang="en") -> str:
    """Fetch clean text of a Wikivoyage page by title."""
    url = f"https://{lang}.wikivoyage.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
    }

    response = requests.get(url, params=params)
    data = response.json()
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


def cache_wikivoyage(title, directory="travel_docs"):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{title}.json")

    if os.path.exists(path):
        print(f"✅ Cached: {title}")
        return

    content = get_wikivoyage_page(title)
    with open(path, "w") as f:
        json.dump({"title": title, "content": content}, f)
    print(f"⬇️ Saved: {title}")


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # or any tokenizer


def chunk_text(
    text: str, location: str, max_tokens=300, overlap=50
) -> List[Dict[str, str]]:
    tokens = tokenizer.tokenize(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokenizer.convert_tokens_to_string(tokens[start:end])
        chunks.append({"text": chunk, "location": location})
        start += max_tokens - overlap
    return chunks


embedder = SentenceTransformer("BAAI/bge-base-en")  # Or use "all-MiniLM-L6-v2"


def build_faiss_index(text_chunks: List[dict], embedder):
    if not text_chunks:
        raise ValueError("No text chunks provided to build FAISS index.")

    texts = [c["text"] for c in text_chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, text_chunks


def save_index(index, chunks, outdir="faiss_index"):
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


def load_index(outdir="faiss_index"):
    index = faiss.read_index(f"{outdir}/index.faiss")
    with open(f"{outdir}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def retrieve(query, chunks, embedder, top_k=5):
    if not chunks:
        return []

    q_emb = embedder.encode([query])
    chunk_texts = [c["text"] for c in chunks]
    chunk_embs = embedder.encode(chunk_texts)

    local_index = faiss.IndexFlatL2(chunk_embs.shape[1])
    local_index.add(np.array(chunk_embs))

    D, I = local_index.search(np.array(q_emb), top_k)

    return [chunk_texts[i] for i in I[0] if i < len(chunk_texts)]


def build_rag_prompt(context: str, question: str) -> str:
    return f"""
You are a helpful travel assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""


session = boto3.Session(profile_name="ogokmen_bedrock")
bedrock = session.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"


# --- Load LLaMA model and tokenizer once ---
def load_llama_model():
    base_model_name = "NousResearch/Llama-2-7b-hf"
    lora_path = "instruction_tuning_output/llama_lora_adapters"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # Load and merge LoRA
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.merge_adapter()  # merge adapters into base model in memory

    return tokenizer, model


# --- LLaMA inference ---
def llama_inference_stream(prompt):
    tokenizer, model = load_llama_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for token in streamer:
        yield token


def call_claude_stream(prompt=None, messages_override=None, retries=5, base_delay=2):
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


def maybe_enrich_prompt(question: str, location: str) -> str:
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

    # Budget (if added later)
    # ...

    return "\n".join(prompt_parts) if prompt_parts else ""


def build_prompt_with_history(history, question):
    messages = history + [{"role": "user", "content": question}]
    return messages


def rag_qa(question, chunks, embedder):
    retrieved_chunks = retrieve(question, chunks, embedder, top_k=5)
    context = "\n---\n".join(retrieved_chunks)
    prompt = build_rag_prompt(context, question)
    return call_claude_stream(prompt)


def log_training_example(
    question, context, answer, path="training_data/rag_pairs.jsonl"
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(
            json.dumps(
                {
                    "input": f"Question: {question}\n\nContext:\n{context}",
                    "output": answer.strip(),
                }
            )
            + "\n"
        )


def build_extension_prompt(seed_questions, n=20):
    formatted = "\n".join(f"- {q}" for q in seed_questions)
    return f"""
You are a helpful travel assistant trainer. Based on the following travel-related user questions, generate {n} *new* questions that are similar in style, varied in topic, and useful for travel planning.

Original questions:
{formatted}

Return a numbered list of {n} new questions only.
"""


# Load once globally
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)


def extract_location(text):
    if text.lower() == text:
        text = text.capitalize()
    """Extract the first detected location-like entity using NER."""
    entities = ner(text)
    for ent in entities:
        if ent["entity_group"] in ["LOC", "PER", "ORG"]:  # optionally just "LOC"
            return ent["word"].strip()
    return None


def maybe_update_rag(location, index_dir="faiss_index"):
    path = f"travel_docs/{location}.json"
    if os.path.exists(path):
        return  # already in DB

    print(f"🌍 Fetching RAG info for: {location}")
    content = get_wikivoyage_page(location)
    if not content.strip():
        print(f"⚠️ No content found for {location}")
        return

    cache_wikivoyage(location)
    chunks = chunk_text(content, location)
    vectors = embedder.encode(chunks)

    index = faiss.read_index(f"{index_dir}/index.faiss")
    with open(f"{index_dir}/chunks.pkl", "rb") as f:
        all_chunks = pickle.load(f)

    index.add(vectors)
    all_chunks.extend(chunks)

    faiss.write_index(index, f"{index_dir}/index.faiss")
    with open(f"{index_dir}/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"✅ {location} added to index.")


def format_llama_chat_example(example):
    """
    Convert a single input/output pair into LLaMA chat-style prompt:
    <s>[INST] ... [/INST] ...
    """
    system_prompt = "You are a helpful travel assistant."
    instruction = example["input"]
    answer = example["output"]

    formatted = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction} [/INST] {answer} </s>"
    return {"text": formatted}


def tokenize_dataset(dataset, tokenizer, max_length=1024):
    def tokenize(example):
        tokenized = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = tokenized["input_ids"][:]  # or .copy() if you prefer
        return tokenized

    return dataset.map(tokenize, batched=False)
