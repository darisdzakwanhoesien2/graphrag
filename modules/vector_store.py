
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")

# def build_or_update_faiss(existing_index, existing_chunks, new_chunks):

#     texts = [c["text"] for c in new_chunks]
#     embeddings = model.encode(texts)
#     embeddings = np.array(embeddings).astype("float32")

#     if existing_index is None:
#         dim = embeddings.shape[1]
#         index = faiss.IndexFlatL2(dim)
#     else:
#         index = existing_index

#     index.add(embeddings)

#     updated_chunks = existing_chunks + new_chunks

#     return index, updated_chunks

# import faiss

# modules/vector_store.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_or_update_index(index, existing_chunks, new_chunks, embedding_model):

    if not new_chunks:
        return index, existing_chunks

    # Combine chunks
    updated_chunks = existing_chunks + new_chunks

    # Encode all chunk texts
    texts = [chunk["text"] for chunk in updated_chunks]
    vectors = embedding_model.encode(texts)
    vectors = np.array(vectors).astype("float32")

    # Rebuild index (sklearn requires full fit)
    index = NearestNeighbors(
        metric="cosine"
    )
    index.fit(vectors)

    return index, updated_chunks



def build_or_update_faiss(existing_index, existing_chunks, new_chunks):

    if not new_chunks:
        return existing_index, existing_chunks

    texts = [c["text"] for c in new_chunks if c["text"].strip() != ""]

    if not texts:
        return existing_index, existing_chunks

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    if existing_index is None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        all_chunks = new_chunks
    else:
        index = existing_index
        all_chunks = existing_chunks + new_chunks

    index.add(embeddings)

    return index, all_chunks

# def build_or_update_faiss(existing_index, existing_chunks, new_chunks):

#     texts = [c["text"] for c in new_chunks]
#     embeddings = model.encode(texts)
#     embeddings = np.array(embeddings).astype("float32")

#     if existing_index is None:
#         dim = embeddings.shape[1]
#         index = faiss.IndexFlatL2(dim)
#     else:
#         index = existing_index

#     index.add(embeddings)

#     updated_chunks = existing_chunks + new_chunks

#     return index, updated_chunks



# def build_or_update_index(existing_index, existing_embeddings, new_chunks):

#     texts = [c["text"] for c in new_chunks]
#     new_embeddings = model.encode(texts)
#     new_embeddings = np.array(new_embeddings).astype("float32")

#     if existing_embeddings is None:
#         all_embeddings = new_embeddings
#     else:
#         all_embeddings = np.vstack([existing_embeddings, new_embeddings])

#     index = NearestNeighbors(
#         n_neighbors=20,
#         metric="cosine"
#     )

#     index.fit(all_embeddings)

#     return index, all_embeddings




# import numpy as np
# from sklearn.neighbors import NearestNeighbors
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer("all-MiniLM-L6-v2")

# def build_or_update_index(index, existing_chunks, new_chunks, embedding_model):

#     # Encode new chunk texts
#     new_texts = [chunk["text"] for chunk in new_chunks]
#     new_vectors = embedding_model.encode(new_texts)
#     new_vectors = np.array(new_vectors).astype("float32")

#     if index is None:
#         index = NearestNeighbors(
#             n_neighbors=5,
#             metric="cosine"
#         )
#         index.fit(new_vectors)
#         all_vectors = new_vectors

#     else:
#         # Rebuild full index (sklearn requires refit)
#         existing_texts = [chunk["text"] for chunk in existing_chunks]
#         existing_vectors = embedding_model.encode(existing_texts)
#         existing_vectors = np.array(existing_vectors).astype("float32")

#         all_vectors = np.vstack([existing_vectors, new_vectors])

#         index = NearestNeighbors(
#             n_neighbors=5,
#             metric="cosine"
#         )
#         index.fit(all_vectors)

#     updated_chunks = existing_chunks + new_chunks

#     return index, updated_chunks


# def build_or_update_index(index, existing_chunks, new_chunks, embedding_model):

#     # Encode new chunk texts
#     new_texts = [chunk["text"] for chunk in new_chunks]
#     new_vectors = embedding_model.encode(new_texts)
#     new_vectors = np.array(new_vectors).astype("float32")

#     if index is None:
#         index = NearestNeighbors(
#             n_neighbors=5,
#             metric="cosine"
#         )
#         index.fit(new_vectors)
#         all_vectors = new_vectors

#     else:
#         # Rebuild full index (sklearn requires refit)
#         existing_texts = [chunk["text"] for chunk in existing_chunks]
#         existing_vectors = embedding_model.encode(existing_texts)
#         existing_vectors = np.array(existing_vectors).astype("float32")

#         all_vectors = np.vstack([existing_vectors, new_vectors])

#         index = NearestNeighbors(
#             n_neighbors=5,
#             metric="cosine"
#         )
#         index.fit(all_vectors)

#     updated_chunks = existing_chunks + new_chunks

#     return index, updated_chunks


# def build_or_update_index(existing_index, existing_embeddings, existing_chunks, new_chunks):

#     if not new_chunks:
#         return existing_index, existing_embeddings, existing_chunks

#     texts = [c["text"] for c in new_chunks if c["text"].strip() != ""]
#     if not texts:
#         return existing_index, existing_embeddings, existing_chunks

#     new_embeddings = model.encode(texts)
#     new_embeddings = np.array(new_embeddings).astype("float32")

#     if existing_embeddings is None:
#         all_embeddings = new_embeddings
#         all_chunks = new_chunks
#     else:
#         all_embeddings = np.vstack([existing_embeddings, new_embeddings])
#         all_chunks = existing_chunks + new_chunks

#     index = NearestNeighbors(
#         n_neighbors=min(20, len(all_embeddings)),
#         metric="cosine"
#     )

#     index.fit(all_embeddings)

#     return index, all_embeddings, all_chunks



# def build_or_update_index(existing_index, chunks):
#     texts = [c["text"] for c in chunks]
#     embeddings = model.encode(texts)

#     embeddings = np.array(embeddings).astype("float32")

#     if existing_index is None:
#         dim = embeddings.shape[1]
#         index = faiss.IndexFlatL2(dim)
#     else:
#         index = existing_index

#     index.add(embeddings)

#     return index

def create_faiss_index(text_chunks):
    embeddings = model.encode(text_chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, embeddings

