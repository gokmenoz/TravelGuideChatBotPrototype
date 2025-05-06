import streamlit as st
import boto3
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from utils import (
    log_training_example, extract_location, 
    maybe_update_rag, load_index, retrieve,
    call_claude_stream, build_rag_prompt
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel, merge_and_unload
from transformers import TextIteratorStreamer
import threading


# --- Bedrock client ---
session = boto3.Session(profile_name="ogokmen_bedrock")
bedrock = session.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

# --- Load LLaMA model and tokenizer once ---
@st.cache_resource
def load_llama_model():
    base_model_name = "NousResearch/Llama-2-7b-hf" 
    lora_path = "llama_lora_adapters/checkpoint-279"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load and merge LoRA
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = merge_and_unload(model)  # merge adapters into base model in memory

    return tokenizer, model

# --- LLaMA inference ---
def llama_inference_stream(prompt):
    tokenizer, model = load_llama_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for token in streamer:
        yield token


# --- Load FAISS + chunks ---
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("faiss_index/index.faiss")
    with open("faiss_index/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


# --- Load embedder ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("BAAI/bge-base-en")

# --- RAG chatbot function ---
def answer_question_with_rag(question, location, chunks, embedder):
    if not location:
        msg = "❗ I couldn't find a destination in your message. Please mention the place you're asking about."
        print("⚠️ No location extracted from query.")
        return msg, None

    location_lower = location.lower()
    location_chunks = [c for c in chunks if c.get("location", "").lower() == location_lower]

    if not location_chunks:
        print("⚠️ No matching location context found. Falling back to Claude.")
        fallback_prompt = f"You are a helpful travel assistant. Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
        return call_claude_stream(fallback_prompt), None

    docs = retrieve(question, location_chunks, embedder)
    if not any(docs):
        print("⚠️ No relevant RAG context found. Falling back to Claude without RAG.")
        fallback_prompt = f"You are a helpful travel assistant. Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
        return call_claude_stream(fallback_prompt), None

    context = "\n---\n".join(docs)
    prompt = build_rag_prompt(context, question)

    return llama_inference_stream(prompt), context


# --- Streamlit UI ---
st.title("🧳 Travel Chatbot powered by LLAMA and Claude")
st.write("Ask about a destination. We will use travel docs to answer.")

# Load index + model
index, chunks = load_faiss_index()
embedder = load_embedder()

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Where are you planning to travel?")

if user_input:
    location = extract_location(user_input)
    if location:
        maybe_update_rag(location)

    index, chunks = load_index()  # reload updated index

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result_container = st.empty()
            output = ""
            
            response_gen, context = answer_question_with_rag(
                user_input, location, chunks, embedder, stream=True
            )

            for word in response_gen:
                output += word
                result_container.markdown(output + "▌")

            result_container.markdown(output)
            log_training_example(user_input, context, output)
            st.session_state.messages.append({"role": "assistant", "content": output})