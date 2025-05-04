import os
import random
import re

from tqdm import tqdm

from constants import QUESTIONS  # seed list
from utils import (
    build_extension_prompt,
    build_rag_prompt,
    call_claude,
    embedder,
    load_index,
    log_training_example,
    retrieve,
)


def generate_instruction_data(questions, out_path="training_data/rag_pairs.jsonl"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    index, chunks = load_index()

    for q in tqdm(questions):
        try:
            retrieved_chunks = retrieve(q, index, chunks, embedder, top_k=5)
            context = "\n---\n".join(retrieved_chunks)
            prompt = build_rag_prompt(context, q)
            answer = call_claude(prompt)
            log_training_example(q, context, answer, path=out_path)
        except Exception as e:
            print(f"⚠️ Error generating for: {q}\n{e}")


if __name__ == "__main__":
    # Start with seed list
    question_pool = list(set(QUESTIONS))

    for i in tqdm(range(50)):
        seed_sample = random.sample(question_pool, k=min(10, len(question_pool)))
        new_prompt = build_extension_prompt(seed_sample, n=20)
        response = call_claude(new_prompt)
        generated = re.findall(r"\d+\.\s+(.*)", response)

        # Add only new unique questions
        for q in generated:
            if q not in question_pool:
                question_pool.append(q)

    print(f"🧠 Total unique questions: {len(question_pool)}")

    # Generate training data from the full pool
    generate_instruction_data(question_pool)
    print("✅ Synthetic RAG instruction tuning data saved.")
