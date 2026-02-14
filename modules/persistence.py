import pickle
import json
from pathlib import Path

DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_index(index):
    with open(DATA_DIR / "vector_index.pkl", "wb") as f:
        pickle.dump(index, f)


def load_index():
    path = DATA_DIR / "vector_index.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_embeddings(embeddings):
    with open(DATA_DIR / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)


def load_embeddings():
    path = DATA_DIR / "embeddings.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_chunks(chunks):
    with open(DATA_DIR / "chunk_metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_chunks():
    path = DATA_DIR / "chunk_metadata.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


def save_graph(graph):
    with open(DATA_DIR / "graph.gpickle", "wb") as f:
        pickle.dump(graph, f)


def load_graph():
    path = DATA_DIR / "graph.gpickle"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_processed_reports(reports):
    with open(DATA_DIR / "processed_reports.json", "w") as f:
        json.dump(reports, f)


def load_processed_reports():
    path = DATA_DIR / "processed_reports.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


# import pickle
# import json
# # import faiss
# from pathlib import Path

# DATA_DIR = Path("data/processed")
# DATA_DIR.mkdir(parents=True, exist_ok=True)

# # ------------------------
# # FAISS
# # ------------------------

# def save_faiss(index):
#     faiss.write_index(index, str(DATA_DIR / "faiss.index"))

# def load_faiss():
#     path = DATA_DIR / "faiss.index"
#     if path.exists():
#         return faiss.read_index(str(path))
#     return None

# # ------------------------
# # CHUNK METADATA
# # ------------------------

# def save_chunks(chunks):
#     with open(DATA_DIR / "chunk_metadata.pkl", "wb") as f:
#         pickle.dump(chunks, f)

# def load_chunks():
#     path = DATA_DIR / "chunk_metadata.pkl"
#     if path.exists():
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     return []

# # ------------------------
# # GRAPH
# # ------------------------

# def save_graph(graph):
#     with open(DATA_DIR / "graph.gpickle", "wb") as f:
#         pickle.dump(graph, f)

# def load_graph():
#     path = DATA_DIR / "graph.gpickle"
#     if path.exists():
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     return None

# # ------------------------
# # PROCESSED REPORT TRACKING
# # ------------------------

# def save_processed_reports(reports):
#     with open(DATA_DIR / "processed_reports.json", "w") as f:
#         json.dump(reports, f)

# def load_processed_reports():
#     path = DATA_DIR / "processed_reports.json"
#     if path.exists():
#         with open(path, "r") as f:
#             return json.load(f)
#     return []


# import pickle
# import json
# import faiss
# from pathlib import Path

# DATA_DIR = Path("data/processed")

# def save_index(index):
#     faiss.write_index(index, str(DATA_DIR / "faiss.index"))

# def load_index():
#     path = DATA_DIR / "faiss.index"
#     if path.exists():
#         return faiss.read_index(str(path))
#     return None

# def save_graph(graph):
#     with open(DATA_DIR / "graph.gpickle", "wb") as f:
#         pickle.dump(graph, f)

# def load_graph():
#     path = DATA_DIR / "graph.gpickle"
#     if path.exists():
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     return None

# def save_metadata(metadata):
#     with open(DATA_DIR / "metadata.json", "w") as f:
#         json.dump(metadata, f)

# def load_metadata():
#     path = DATA_DIR / "metadata.json"
#     if path.exists():
#         with open(path, "r") as f:
#             return json.load(f)
#     return []