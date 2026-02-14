import pickle
import json
from pathlib import Path

DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# INTERNAL HELPER
# -------------------------------------------------

def _pkl_path(namespace, name):
    return DATA_DIR / f"{namespace}_{name}.pkl"

def _json_path(namespace, name):
    return DATA_DIR / f"{namespace}_{name}.json"


# -------------------------------------------------
# INDEX
# -------------------------------------------------

def save_index(index, namespace):
    with open(_pkl_path(namespace, "index"), "wb") as f:
        pickle.dump(index, f)

def load_index(namespace):
    path = _pkl_path(namespace, "index")
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# -------------------------------------------------
# CHUNKS
# -------------------------------------------------

def save_chunks(chunks, namespace):
    with open(_pkl_path(namespace, "chunks"), "wb") as f:
        pickle.dump(chunks, f)

def load_chunks(namespace):
    path = _pkl_path(namespace, "chunks")
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


# -------------------------------------------------
# GRAPH
# -------------------------------------------------

def save_graph(graph, namespace):
    with open(_pkl_path(namespace, "graph"), "wb") as f:
        pickle.dump(graph, f)

def load_graph(namespace):
    path = _pkl_path(namespace, "graph")
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# -------------------------------------------------
# PROCESSED REPORTS
# -------------------------------------------------

def save_processed_reports(reports, namespace):
    with open(_json_path(namespace, "processed"), "w") as f:
        json.dump(reports, f)

def load_processed_reports(namespace):
    path = _json_path(namespace, "processed")
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


# import pickle
# import json
# from pathlib import Path


# def get_data_dir(namespace: str):
#     base = Path("data/processed") / namespace
#     base.mkdir(parents=True, exist_ok=True)
#     return base


# # -----------------------
# # VECTOR INDEX
# # -----------------------

# def save_index(index, namespace):
#     data_dir = get_data_dir(namespace)
#     with open(data_dir / "vector_index.pkl", "wb") as f:
#         pickle.dump(index, f)


# def load_index(namespace):
#     data_dir = get_data_dir(namespace)
#     path = data_dir / "vector_index.pkl"
#     if path.exists():
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     return None


# # -----------------------
# # EMBEDDINGS
# # -----------------------

# def save_embeddings(embeddings, namespace):
#     data_dir = get_data_dir(namespace)
#     with open(data_dir / "embeddings.pkl", "wb") as f:
#         pickle.dump(embeddings, f)


# def load_embeddings(namespace):
#     data_dir = get_data_dir(namespace)
#     path = data_dir / "embeddings.pkl"
#     if path.exists():
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     return None


# # -----------------------
# # CHUNKS
# # -----------------------

# def save_chunks(chunks, namespace):
#     data_dir = get_data_dir(namespace)
#     with open(data_dir / "chunk_metadata.pkl", "wb") as f:
#         pickle.dump(chunks, f)


# def load_chunks(namespace):
#     data_dir = get_data_dir(namespace)
#     path = data_dir / "chunk_metadata.pkl"
#     if path.exists():
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     return []


# # -----------------------
# # GRAPH
# # -----------------------

# def save_graph(graph, namespace):
#     data_dir = get_data_dir(namespace)
#     with open(data_dir / "graph.gpickle", "wb") as f:
#         pickle.dump(graph, f)


# def load_graph(namespace):
#     data_dir = get_data_dir(namespace)
#     path = data_dir / "graph.gpickle"
#     if path.exists():
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     return None


# # -----------------------
# # PROCESSED TRACKING (OCR only)
# # -----------------------

# def save_processed_reports(reports, namespace):
#     data_dir = get_data_dir(namespace)
#     with open(data_dir / "processed_reports.json", "w") as f:
#         json.dump(reports, f)


# def load_processed_reports(namespace):
#     data_dir = get_data_dir(namespace)
#     path = data_dir / "processed_reports.json"
#     if path.exists():
#         with open(path, "r") as f:
#             return json.load(f)
#     return []


# import pickle
# import json
# from pathlib import Path

# DATA_DIR = Path("data/processed")
# DATA_DIR.mkdir(parents=True, exist_ok=True)


# def save_index(index):
#     with open(DATA_DIR / "vector_index.pkl", "wb") as f:
#         pickle.dump(index, f)


# def load_index():
#     path = DATA_DIR / "vector_index.pkl"
#     if path.exists():
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     return None


# def save_embeddings(embeddings):
#     with open(DATA_DIR / "embeddings.pkl", "wb") as f:
#         pickle.dump(embeddings, f)


# def load_embeddings():
#     path = DATA_DIR / "embeddings.pkl"
#     if path.exists():
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     return None


# def save_chunks(chunks):
#     with open(DATA_DIR / "chunk_metadata.pkl", "wb") as f:
#         pickle.dump(chunks, f)


# def load_chunks():
#     path = DATA_DIR / "chunk_metadata.pkl"
#     if path.exists():
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     return []


# def save_graph(graph):
#     with open(DATA_DIR / "graph.gpickle", "wb") as f:
#         pickle.dump(graph, f)


# def load_graph():
#     path = DATA_DIR / "graph.gpickle"
#     if path.exists():
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     return None


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