# modules/rag_engine.py

import numpy as np
from modules.retriever import retrieve
from modules.llm_interface import generate_response


def format_context(chunks):
    context = ""
    for c in chunks:
        context += f"\n[Source: {c['doc_id']} | Page {c['page']}]\n{c['text']}\n"
    return context

def generate_answer(
    query,
    vector_index,
    documents,
    embedding_model,
    provider,
    model,
    temperature=0.2,
    top_k=5
):

    # 1️⃣ Encode query
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # 2️⃣ Retrieve using sklearn
    distances, indices = vector_index.kneighbors(
        query_embedding,
        n_neighbors=top_k
    )

    retrieved_chunks = [documents[i] for i in indices[0]]
    similarity_scores = distances[0].tolist()

    # 3️⃣ Build context
    context = ""
    for chunk in retrieved_chunks:
        context += f"\n[Source: {chunk['doc_id']} | Page {chunk['page']}]\n"
        context += chunk["text"] + "\n"

    prompt = f"""
You are an ESG analysis assistant.

Use ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    # 4️⃣ Call LLM
    response = generate_response(
        prompt=prompt,
        provider=provider,
        model=model,
        temperature=temperature
    )

    return {
        "response": response,
        "sources": retrieved_chunks,
        "scores": similarity_scores,
        "prompt": prompt
    }
