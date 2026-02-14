Okay. Now we’re building a **real system**, not a toy demo. I love this direction. 🔥

You want:

* ✅ Persistent FAISS (properly done, not hacky)
* ✅ Incremental ingestion
* ✅ Both OpenRouter (cloud) + LMStudio (local)
* ✅ Still visualizable graph

We’ll design this clean, modular, scalable.

---

# 🧠 FINAL ARCHITECTURE (Production-Ready)

```
OCR (.md pages)
        ↓
Incremental Loader
        ↓
Chunking
        ↓
Embeddings (MiniLM)
        ↓
Persistent FAISS Index
        ↓
Persistent Metadata (chunk mapping)
        ↓
Persistent Knowledge Graph
        ↓
Retriever
        ↓
LLM Layer
     ↙        ↘
OpenRouter   LMStudio
```

---

# 📂 UPDATED PROJECT STRUCTURE

```
graphrag_streamlit/
│
├── app.py
├── config.py
│
├── data/
│   ├── ocr_output/
│   └── processed/
│        ├── faiss.index
│        ├── chunk_metadata.pkl
│        ├── graph.gpickle
│        └── processed_reports.json
│
├── modules/
│   ├── data_loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── graph_builder.py
│   ├── retriever.py
│   ├── persistence.py
│   └── llm_interface.py
│
└── pages/
    ├── 1_Chatbot.py
    └── 2_Graph_Viewer.py
```

---

# 🔥 PART 1 — Proper FAISS Persistence

We must persist:

1. FAISS index
2. Chunk metadata mapping (index → chunk info)

Because FAISS stores only vectors, not metadata.

---

## 📁 modules/persistence.py (FULL VERSION)

```python
import pickle
import json
import faiss
from pathlib import Path

DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# FAISS
# ------------------------

def save_faiss(index):
    faiss.write_index(index, str(DATA_DIR / "faiss.index"))

def load_faiss():
    path = DATA_DIR / "faiss.index"
    if path.exists():
        return faiss.read_index(str(path))
    return None

# ------------------------
# CHUNK METADATA
# ------------------------

def save_chunks(chunks):
    with open(DATA_DIR / "chunk_metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_chunks():
    path = DATA_DIR / "chunk_metadata.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return []

# ------------------------
# GRAPH
# ------------------------

def save_graph(graph):
    with open(DATA_DIR / "graph.gpickle", "wb") as f:
        pickle.dump(graph, f)

def load_graph():
    path = DATA_DIR / "graph.gpickle"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# ------------------------
# PROCESSED REPORT TRACKING
# ------------------------

def save_processed_reports(reports):
    with open(DATA_DIR / "processed_reports.json", "w") as f:
        json.dump(reports, f)

def load_processed_reports():
    path = DATA_DIR / "processed_reports.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []
```

---

# 🔥 PART 2 — FAISS Incremental Update (Correctly Done)

### 📁 modules/vector_store.py

```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def build_or_update_faiss(existing_index, existing_chunks, new_chunks):

    texts = [c["text"] for c in new_chunks]
    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    if existing_index is None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
    else:
        index = existing_index

    index.add(embeddings)

    updated_chunks = existing_chunks + new_chunks

    return index, updated_chunks
```

Now:

* Old vectors remain
* New ones append
* Metadata grows consistently

No reprocessing required.

---

# 🔥 PART 3 — LLM Interface (OpenRouter + LMStudio)

We make it selectable.

---

## 📁 modules/llm_interface.py

```python
import requests
import os

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# -----------------------------------
# OpenRouter
# -----------------------------------

def call_openrouter(prompt, model="mistralai/mistral-7b-instruct"):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]


# -----------------------------------
# LMStudio (Local)
# -----------------------------------

def call_lmstudio(prompt, model="local-model"):
    url = "http://localhost:1234/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, json=payload)
    return response.json()["choices"][0]["message"]["content"]


# -----------------------------------
# Unified Interface
# -----------------------------------

def generate_response(prompt, provider="openrouter"):

    if provider == "openrouter":
        return call_openrouter(prompt)

    elif provider == "lmstudio":
        return call_lmstudio(prompt)

    else:
        raise ValueError("Unsupported LLM provider")
```

Now in Streamlit, you can select:

```python
provider = st.selectbox("Select LLM", ["openrouter", "lmstudio"])
```

---

# 🔥 PART 4 — Retriever

### 📁 modules/retriever.py

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query, index, chunks, k=5):

    query_embedding = model.encode([query]).astype("float32")
    D, I = index.search(query_embedding, k)

    return [chunks[i] for i in I[0]]
```

---

# 🔥 PART 5 — Graph Visualization

### 📁 pages/2_Graph_Viewer.py

```python
import streamlit as st
from pyvis.network import Network
from modules.persistence import load_graph

st.title("Knowledge Graph")

graph = load_graph()

if graph:
    net = Network(height="700px", width="100%")

    for node in graph.nodes():
        net.add_node(node)

    for edge in graph.edges():
        net.add_edge(edge[0], edge[1])

    net.save_graph("graph.html")

    with open("graph.html", "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=700)
else:
    st.write("Graph not available.")
```

---

# 💡 Why Persist FAISS?

Without persistence:

* 500 pages × embedding every restart
* 30–60 seconds startup
* Wasteful compute

With persistence:

* Load instantly
* Only embed new reports
* Stable index state

---

# 💡 Why Persist Chunk Metadata?

FAISS returns only vector IDs.
You must map:

```
vector_id → chunk → doc_id → page
```

Without storing metadata, retrieval breaks.

---

# 💡 Why Persist Graph?

Because:

* Graph grows across reports
* Cross-report entity relationships matter
* Centrality measures depend on global structure
* You want visualization stability

Rebuilding graph every time = unnecessary.

---

# 🚀 System Capability After This

You now have:

* Scalable ingestion
* Incremental updates
* Persistent vector DB
* Persistent knowledge graph
* Multi-LLM switching
* Graph visualization
* Citation-ready retrieval

This is **enterprise-level GraphRAG**.

---

Now I want to level this up one more time.

Do you want:

1. 🔬 Graph-aware retrieval (boost chunks connected to query entities)?
2. 📊 Centrality-based ranking?
3. 🧠 Hybrid scoring (vector score + graph proximity)?

Because that’s where your system becomes *research-paper level*, not just engineering.

Tell me how deep you want to go.
# graphrag
