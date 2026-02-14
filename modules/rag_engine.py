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

# def generate_answer(
#     query,
#     vector_index,
#     documents,
#     embedding_model,
#     provider,
#     model,
#     temperature=0.2,
#     top_k=5
# ):
#     """
#     Main GraphRAG pipeline:
#     1. Encode query
#     2. FAISS retrieval
#     3. Build context
#     4. Call LLM
#     5. Return structured result
#     """

#     # -----------------------------
#     # 1️⃣ Encode Query
#     # -----------------------------
#     query_embedding = embedding_model.encode([query])
#     query_embedding = np.array(query_embedding).astype("float32")

#     # -----------------------------
#     # 2️⃣ Vector Search
#     # -----------------------------
#     D, I = vector_index.search(query_embedding, top_k)

#     retrieved_chunks = [documents[i] for i in I[0]]
#     similarity_scores = D[0].tolist()

#     # -----------------------------
#     # 3️⃣ Build Context
#     # -----------------------------
#     context = "\n\n".join([
#         f"[Document {chunk['doc_id']} | Page {chunk['page']}]\n{chunk['text']}"
#         for chunk in retrieved_chunks
#     ])

#     prompt = f"""
# You are an ESG analysis assistant.

# Use ONLY the provided context to answer the question.

# Context:
# {context}

# Question:
# {query}

# Answer clearly and concisely.
# """

#     # -----------------------------
#     # 4️⃣ LLM Call
#     # -----------------------------
#     response = generate_response(
#         prompt=prompt,
#         provider=provider,
#         model=model,
#         temperature=temperature
#     )

#     # -----------------------------
#     # 5️⃣ Return Structured Output
#     # -----------------------------
#     return {
#         "response": response,
#         "sources": retrieved_chunks,
#         "scores": similarity_scores,
#         "prompt": prompt
#     }


# def generate_answer(
#     query,
#     vector_index,
#     documents,
#     embeddings,
#     provider,
#     model,
#     temperature,
#     top_k=5
# ):

#     import numpy as np

#     query_embedding = embeddings.encode([query])
#     query_embedding = np.array(query_embedding).astype("float32")

#     D, I = vector_index.search(query_embedding, top_k)

#     retrieved_docs = [documents[i] for i in I[0]]
#     similarity_scores = D[0].tolist()

#     context = "\n\n".join([doc["text"] for doc in retrieved_docs])

#     prompt = f"""
# Context:
# {context}

# Question:
# {query}

# Answer strictly based on the context.
# """

#     response = generate_response(
#         prompt=prompt,
#         provider=provider,
#         model=model,
#         temperature=temperature
#     )

#     return {
#         "response": response,
#         "sources": retrieved_docs,
#         "scores": similarity_scores,
#         "prompt": prompt
#     }



# def generate_answer(query, index, embeddings, chunks, graph, provider):

#     if index is None or embeddings is None or not chunks:
#         return "System not ready. No documents loaded.", []

#     retrieved_chunks = retrieve(query, index, embeddings, chunks, graph)

#     context = format_context(retrieved_chunks)

#     prompt = f"""
# You are an edtech assistant.

# Use the context below to answer the question.
# If unsure, say so.

# Context:
# {context}

# Question:
# {query}
# """

#     response = generate_response(prompt, provider)

#     return response, retrieved_chunks


# from modules.retriever import retrieve
# from modules.llm_interface import generate_response
# from modules.graph_utils import compute_centrality

# def format_context(chunks):
#     context = ""
#     for c in chunks:
#         context += f"\n[Source: {c['doc_id']} | Page {c['page']}]\n{c['text']}\n"
#     return context

# def generate_answer(query, graph, vector_index, embeddings, llm, top_k=5):

#     # 1️⃣ Vector Retrieval
#     query_embedding = embeddings.embed_query(query)
#     D, I = vector_index.search(query_embedding, top_k)

#     retrieved_docs = [graph["documents"][i] for i in I[0]]

#     # 2️⃣ Graph Expansion
#     related_nodes = []
#     for doc in retrieved_docs:
#         related_nodes.extend(graph["relations"].get(doc["id"], []))

#     # 3️⃣ Build Context
#     context = "\n\n".join([doc["text"] for doc in retrieved_docs])

#     # 4️⃣ Construct Prompt
#     prompt = f"""
# You are an ESG analyst.

# Context:
# {context}

# Question:
# {query}

# Answer grounded strictly in the context.
# """

#     # 5️⃣ LLM Generation
#     response = llm(prompt)

#     return response, retrieved_docs


# def generate_answer(query, index, chunks, graph, provider):

#     retrieved_chunks = retrieve(query, index, chunks, graph, k=5)

#     context = format_context(retrieved_chunks)

#     prompt = f"""
# You are an ESG research assistant.

# Use the context below to answer the question.
# If unsure, say so.

# Context:
# {context}

# Question:
# {query}
# """

#     response = generate_response(prompt, provider=provider)

#     return response, retrieved_chunks


# def generate_answer(query, vector_index, graph, docs):
    
#     # 1. Vector retrieval
#     retrieved_chunks = retrieve_top_k(query, vector_index)
    
#     # 2. Extract entities from query
#     entities = extract_entities(query)
    
#     # 3. Graph expansion
#     graph_context = expand_graph_context(graph, entities)
    
#     # 4. Merge context
#     final_context = merge_context(retrieved_chunks, graph_context, docs)
    
#     # 5. Call LLM
#     answer = call_llm(query, final_context)
    
#     return answer