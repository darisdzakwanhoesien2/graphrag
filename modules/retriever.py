from sentence_transformers import SentenceTransformer
from modules.graph_utils import extract_entities, graph_proximity_score

model = SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(query, index, embeddings, chunks, graph, k=5):

    query_embedding = model.encode([query]).astype("float32")

    distances, indices = index.kneighbors(query_embedding)

    query_entities = extract_entities(query)

    scored = []

    for rank, idx in enumerate(indices[0]):
        chunk = chunks[idx]

        vector_score = 1 - distances[0][rank]

        chunk_entities = extract_entities(chunk["text"])

        graph_score = graph_proximity_score(
            graph,
            query_entities,
            chunk_entities,
            depth=2
        )

        hybrid_score = 0.7 * vector_score + 0.3 * graph_score

        scored.append((hybrid_score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [c[1] for c in scored[:k]]


# import numpy as np
# from sentence_transformers import SentenceTransformer
# from modules.graph_utils import extract_entities, graph_proximity_score

# model = SentenceTransformer("all-MiniLM-L6-v2")

# def retrieve(query, index, chunks, graph, k=5):

#     query_embedding = model.encode([query]).astype("float32")
#     D, I = index.search(query_embedding, min(20, len(chunks)))

#     query_entities = extract_entities(query)

#     scored_chunks = []

#     for rank, idx in enumerate(I[0]):
#         chunk = chunks[idx]
#         vector_score = 1 / (1 + D[0][rank])

#         chunk_entities = extract_entities(chunk["text"])

#         graph_score = graph_proximity_score(
#             graph,
#             query_entities,
#             chunk_entities,
#             depth=2
#         )

#         hybrid_score = 0.7 * vector_score + 0.3 * graph_score

#         scored_chunks.append((hybrid_score, chunk))

#     scored_chunks.sort(key=lambda x: x[0], reverse=True)

#     return [c[1] for c in scored_chunks[:k]]


# import numpy as np
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")

# def retrieve(query, index, chunks, k=5):

#     query_embedding = model.encode([query]).astype("float32")
#     D, I = index.search(query_embedding, k)

#     return [chunks[i] for i in I[0]]