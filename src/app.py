import pickle

import boto3
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

from utils import (
    build_rag_prompt,
    call_claude_stream,
    extract_location,
    load_index,
    maybe_update_rag,
    retrieve,
)

# --- Bedrock client ---
session = boto3.Session(profile_name="ogokmen_bedrock")
bedrock = session.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"


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
    location_chunks = [
        c for c in chunks if c.get("location", "").lower() == location_lower
    ]

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

    return call_claude_stream(prompt), context


# --- Streamlit UI ---
st.title("🧳 Travel Chatbot powered by Claude")
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
                user_input, location, chunks, embedder
            )

            for word in response_gen:
                output += word
                result_container.markdown(output + "▌")

            result_container.markdown(output)
            st.session_state.messages.append({"role": "assistant", "content": output})
