import pickle

import boto3
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

from utils import (
    answer_question_with_rag,
    extract_location,
    load_index,
    log_training_example,
    maybe_update_rag,
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


# --- Streamlit UI ---
st.title("🧳 Claude 3.7 Travel Chatbot (RAG)")
st.write("Ask about a destination. Claude will use travel docs to answer.")

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
            reply, context = answer_question_with_rag(
                user_input, location, chunks, embedder
            )
            log_training_example(user_input, context, reply)
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
