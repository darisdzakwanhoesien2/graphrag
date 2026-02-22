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

def create_faiss_index(text_chunks):
    embeddings = model.encode(text_chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, embeddings

