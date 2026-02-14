import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
from pathlib import Path

from modules.chunker import chunk_documents
from modules.vector_store import build_or_update_index
from modules.graph_builder import build_or_update_graph
from modules.persistence import (
    load_index,
    save_index,
    load_chunks,
    save_chunks,
    load_graph,
    save_graph
)
from modules.rag_engine import generate_answer


# =================================================
# CONFIG
# =================================================

NAMESPACE = "abstract"

HISTORY_DIR = Path("data/processed/abstract")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_FILE = HISTORY_DIR / "query_history.json"


st.title("📄 Abstract Journal GraphRAG (CSV)")


# =================================================
# LOAD EMBEDDING MODEL
# =================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()


# =================================================
# SAFE SESSION INIT
# =================================================

if "abstract_index" not in st.session_state:
    st.session_state.abstract_index = load_index(NAMESPACE)

if "abstract_chunks" not in st.session_state:
    st.session_state.abstract_chunks = load_chunks(NAMESPACE)

if "abstract_graph" not in st.session_state:
    st.session_state.abstract_graph = load_graph(NAMESPACE)


# =================================================
# CSV UPLOAD
# =================================================

uploaded_file = st.file_uploader("Upload CSV with Abstracts", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    if "Abstract" not in df.columns:
        st.error("CSV must contain an 'Abstract' column.")
        st.stop()

    documents = []

    for _, row in df.iterrows():

        abstract_text = row["Abstract"]

        if pd.isna(abstract_text) or abstract_text == "(missing abstract)":
            continue

        documents.append({
            "doc_id": row.get("DOI", "Unknown DOI"),
            "page": row.get("Year", 0),
            "text": abstract_text,
            "title": row.get("Title", ""),
            "authors": row.get("Authors", ""),
            "journal": row.get("Journal", "")
        })

    if len(documents) == 0:
        st.warning("No valid abstracts found in CSV.")
    else:

        new_chunks = chunk_documents(documents)

        index, chunks = build_or_update_index(
            index=st.session_state.abstract_index,
            existing_chunks=st.session_state.abstract_chunks or [],
            new_chunks=new_chunks,
            embedding_model=embedding_model
        )

        graph = build_or_update_graph(
            st.session_state.abstract_graph,
            new_chunks
        )

        save_index(index, NAMESPACE)
        save_chunks(chunks, NAMESPACE)
        save_graph(graph, NAMESPACE)

        st.session_state.abstract_index = index
        st.session_state.abstract_chunks = chunks
        st.session_state.abstract_graph = graph

        st.success(f"✅ Indexed {len(documents)} abstracts successfully!")


# =================================================
# QUERY SECTION
# =================================================

query = st.text_input("Ask about the uploaded abstracts")

if query:

    if st.session_state.abstract_index is None:
        st.warning("⚠️ No abstracts indexed yet.")
    else:

        provider = "lmstudio"
        model = "mistralai/ministral-3-3b"
        temperature = 0.2
        top_k = 5

        result = generate_answer(
            query=query,
            vector_index=st.session_state.abstract_index,
            documents=st.session_state.abstract_chunks,
            embedding_model=embedding_model,
            provider=provider,
            model=model,
            temperature=temperature,
            top_k=top_k
        )

        response = result["response"]
        sources = result["sources"]
        scores = result["scores"]

        st.write("### 🧠 Answer")
        st.write(response)

        with st.expander("📚 Sources Used"):

            for i, s in enumerate(sources):

                st.markdown(f"""
**DOI:** {s['doc_id']}  
**Year:** {s['page']}  
**Score:** {scores[i]:.4f}  
""")

        # =================================================
        # STORE QUERY HISTORY
        # =================================================

        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "top_k": top_k,
            "response": response,
            "sources": [
                {
                    "doi": s["doc_id"],
                    "year": s["page"],
                    "score": scores[i]
                }
                for i, s in enumerate(sources)
            ]
        }

        # Append to history file
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        else:
            history = []

        history.append(history_entry)

        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

        st.success("📁 Query stored in history.")


# import streamlit as st
# import pandas as pd
# from sentence_transformers import SentenceTransformer

# from modules.chunker import chunk_documents
# from modules.vector_store import build_or_update_index
# from modules.graph_builder import build_or_update_graph
# from modules.persistence import (
#     load_index,
#     save_index,
#     load_chunks,
#     save_chunks,
#     load_graph,
#     save_graph
# )
# from modules.rag_engine import generate_answer


# # -------------------------------------------------
# # CONFIG
# # -------------------------------------------------

# NAMESPACE = "abstract"

# st.title("📄 Abstract Journal GraphRAG (CSV Version)")


# # -------------------------------------------------
# # LOAD EMBEDDING MODEL
# # -------------------------------------------------

# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# embedding_model = load_embedding_model()


# # -------------------------------------------------
# # SAFE SESSION INIT
# # -------------------------------------------------

# if "abstract_index" not in st.session_state:
#     st.session_state.abstract_index = load_index(NAMESPACE)

# if "abstract_chunks" not in st.session_state:
#     st.session_state.abstract_chunks = load_chunks(NAMESPACE)

# if "abstract_graph" not in st.session_state:
#     st.session_state.abstract_graph = load_graph(NAMESPACE)


# # -------------------------------------------------
# # CSV UPLOAD
# # -------------------------------------------------

# uploaded_file = st.file_uploader("Upload CSV with Abstracts", type=["csv"])

# if uploaded_file:

#     df = pd.read_csv(uploaded_file)

#     required_column = "Abstract"

#     if required_column not in df.columns:
#         st.error("CSV must contain an 'Abstract' column.")
#         st.stop()

#     documents = []

#     for _, row in df.iterrows():

#         abstract_text = row["Abstract"]

#         # Skip missing abstracts
#         if pd.isna(abstract_text) or abstract_text == "(missing abstract)":
#             continue

#         documents.append({
#             "doc_id": row.get("DOI", "Unknown DOI"),
#             "page": row.get("Year", 0),
#             "text": abstract_text,
#             "title": row.get("Title", ""),
#             "authors": row.get("Authors", ""),
#             "journal": row.get("Journal", "")
#         })

#     new_chunks = chunk_documents(documents)

#     index, chunks = build_or_update_index(
#         index=st.session_state.abstract_index,
#         existing_chunks=st.session_state.abstract_chunks or [],
#         new_chunks=new_chunks,
#         embedding_model=embedding_model
#     )

#     graph = build_or_update_graph(
#         st.session_state.abstract_graph,
#         new_chunks
#     )

#     save_index(index, NAMESPACE)
#     save_chunks(chunks, NAMESPACE)
#     save_graph(graph, NAMESPACE)

#     st.session_state.abstract_index = index
#     st.session_state.abstract_chunks = chunks
#     st.session_state.abstract_graph = graph

#     st.success(f"✅ Indexed {len(documents)} abstracts successfully!")


# # -------------------------------------------------
# # QUERY
# # -------------------------------------------------

# query = st.text_input("Ask about the uploaded abstracts")

# if query:

#     if st.session_state.abstract_index is None:
#         st.warning("⚠️ No abstracts indexed yet.")
#     else:

#         result = generate_answer(
#             query=query,
#             vector_index=st.session_state.abstract_index,
#             documents=st.session_state.abstract_chunks,
#             embedding_model=embedding_model,
#             provider="lmstudio",
#             model="mistralai/ministral-3-3b",
#             temperature=0.2,
#             top_k=5
#         )

#         st.write("### 🧠 Answer")
#         st.write(result["response"])

#         with st.expander("📚 Sources Used"):

#             for i, s in enumerate(result["sources"]):

#                 # find metadata from original chunk
#                 st.markdown(f"""
# **DOI:** {s['doc_id']}  
# **Year:** {s['page']}  
# **Score:** {result['scores'][i]:.4f}  
# """)


# import streamlit as st
# from sentence_transformers import SentenceTransformer

# from modules.chunker import chunk_documents
# from modules.vector_store import build_or_update_index
# from modules.graph_builder import build_or_update_graph
# from modules.persistence import (
#     load_index,
#     save_index,
#     load_chunks,
#     save_chunks,
#     load_graph,
#     save_graph
# )
# from modules.rag_engine import generate_answer


# # -------------------------------------------------
# # NAMESPACE
# # -------------------------------------------------

# NAMESPACE = "abstract"

# st.title("📄 Abstract Journal GraphRAG")


# # -------------------------------------------------
# # LOAD EMBEDDING MODEL
# # -------------------------------------------------

# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# embedding_model = load_embedding_model()


# # -------------------------------------------------
# # SAFE SESSION INITIALIZATION
# # -------------------------------------------------

# if "abstract_index" not in st.session_state:
#     st.session_state.abstract_index = load_index(NAMESPACE)

# if "abstract_chunks" not in st.session_state:
#     st.session_state.abstract_chunks = load_chunks(NAMESPACE)

# if "abstract_graph" not in st.session_state:
#     st.session_state.abstract_graph = load_graph(NAMESPACE)


# # -------------------------------------------------
# # UPLOAD ABSTRACT
# # -------------------------------------------------

# uploaded_file = st.file_uploader("Upload Abstract (.txt)", type=["txt"])

# if uploaded_file:

#     text = uploaded_file.read().decode("utf-8")

#     docs = [{
#         "doc_id": uploaded_file.name,
#         "page": 1,
#         "text": text
#     }]

#     new_chunks = chunk_documents(docs)

#     index, chunks = build_or_update_index(
#         index=st.session_state.abstract_index,
#         existing_chunks=st.session_state.abstract_chunks or [],
#         new_chunks=new_chunks,
#         embedding_model=embedding_model
#     )

#     graph = build_or_update_graph(
#         st.session_state.abstract_graph,
#         new_chunks
#     )

#     save_index(index, NAMESPACE)
#     save_chunks(chunks, NAMESPACE)
#     save_graph(graph, NAMESPACE)

#     st.session_state.abstract_index = index
#     st.session_state.abstract_chunks = chunks
#     st.session_state.abstract_graph = graph

#     st.success("✅ Abstract indexed successfully!")


# # -------------------------------------------------
# # QUERY ABSTRACTS
# # -------------------------------------------------

# query = st.text_input("Ask about uploaded abstracts")

# if query:

#     if st.session_state.abstract_index is None:
#         st.warning("⚠️ No abstracts indexed yet.")
#     else:

#         result = generate_answer(
#             query=query,
#             vector_index=st.session_state.abstract_index,
#             documents=st.session_state.abstract_chunks,
#             embedding_model=embedding_model,
#             provider="lmstudio",
#             model="mistralai/ministral-3-3b",
#             temperature=0.2,
#             top_k=5
#         )

#         st.write("### 🧠 Answer")
#         st.write(result["response"])

#         with st.expander("📚 Sources Used"):
#             for i, s in enumerate(result["sources"]):
#                 st.write(
#                     f"{s['doc_id']} | Page {s['page']} | Score {result['scores'][i]:.4f}"
#                 )


# import streamlit as st
# from sentence_transformers import SentenceTransformer

# from modules.chunker import chunk_documents
# from modules.vector_store import build_or_update_index
# from modules.graph_builder import build_or_update_graph
# from modules.persistence import (
#     load_index,
#     save_index,
#     load_chunks,
#     save_chunks,
#     load_graph,
#     save_graph
# )
# from modules.rag_engine import generate_answer


# # -------------------------------------------------
# # NAMESPACE
# # -------------------------------------------------

# NAMESPACE = "abstract"


# st.title("📄 Abstract Journal GraphRAG")


# # -------------------------------------------------
# # LOAD EMBEDDING MODEL
# # -------------------------------------------------

# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# embedding_model = load_embedding_model()


# # -------------------------------------------------
# # INITIALIZE SYSTEM
# # -------------------------------------------------

# def initialize_system():
#     index = load_index(NAMESPACE)
#     chunks = load_chunks(NAMESPACE)
#     graph = load_graph(NAMESPACE)

#     if chunks is None:
#         chunks = []

#     return index, chunks, graph


# if "abstract_ready" not in st.session_state:
#     idx, ch, gr = initialize_system()
#     st.session_state.abstract_index = idx
#     st.session_state.abstract_chunks = ch
#     st.session_state.abstract_graph = gr
#     st.session_state.abstract_ready = True


# # -------------------------------------------------
# # UPLOAD ABSTRACT
# # -------------------------------------------------

# uploaded_file = st.file_uploader("Upload Abstract (.txt)", type=["txt"])

# if uploaded_file:

#     text = uploaded_file.read().decode("utf-8")

#     docs = [{
#         "doc_id": uploaded_file.name,
#         "page": 1,
#         "text": text
#     }]

#     new_chunks = chunk_documents(docs)

#     index, chunks = build_or_update_index(
#         index=st.session_state.abstract_index,
#         existing_chunks=st.session_state.abstract_chunks,
#         new_chunks=new_chunks,
#         embedding_model=embedding_model
#     )

#     graph = build_or_update_graph(
#         st.session_state.abstract_graph,
#         new_chunks
#     )

#     save_index(index, NAMESPACE)
#     save_chunks(chunks, NAMESPACE)
#     save_graph(graph, NAMESPACE)

#     st.session_state.abstract_index = index
#     st.session_state.abstract_chunks = chunks
#     st.session_state.abstract_graph = graph

#     st.success("✅ Abstract indexed successfully!")


# # -------------------------------------------------
# # QUERY ABSTRACTS
# # -------------------------------------------------

# query = st.text_input("Ask about uploaded abstracts")

# if query and st.session_state.abstract_index:

#     result = generate_answer(
#         query=query,
#         vector_index=st.session_state.abstract_index,
#         documents=st.session_state.abstract_chunks,
#         embedding_model=embedding_model,
#         provider="lmstudio",  # you can make this dynamic later
#         model="mistralai/ministral-3-3b",
#         temperature=0.2,
#         top_k=5
#     )

#     st.write("### 🧠 Answer")
#     st.write(result["response"])

#     with st.expander("📚 Sources Used"):
#         for i, s in enumerate(result["sources"]):
#             st.write(
#                 f"{s['doc_id']} | Page {s['page']} | Score {result['scores'][i]:.4f}"
#             )


# import streamlit as st

# from modules.abstract_loader import get_abstract_folders, load_abstract_folder
# from modules.abstract_chunker import chunk_abstracts
# from modules.vector_store import build_or_update_index
# from modules.graph_builder import build_or_update_graph
# from modules.persistence import (
#     load_index,
#     save_index,
#     load_embeddings,
#     save_embeddings,
#     load_chunks,
#     save_chunks,
#     load_graph,
#     save_graph
# )
# from modules.rag_engine import generate_answer


# NAMESPACE = "abstract"


# def initialize_abstract_system():

#     index = load_index(NAMESPACE)
#     embeddings = load_embeddings(NAMESPACE)
#     chunks = load_chunks(NAMESPACE)
#     graph = load_graph(NAMESPACE)

#     if chunks is None:
#         chunks = []

#     folders = get_abstract_folders()

#     st.write(f"Found abstract folders: {folders}")

#     for folder in folders:

#         docs = load_abstract_folder(folder)
#         new_chunks = chunk_abstracts(docs)

#         index, embeddings, chunks = build_or_update_index(
#             index,
#             embeddings,
#             chunks,
#             new_chunks
#         )

#         graph = build_or_update_graph(graph, new_chunks)

#     save_index(index, NAMESPACE)
#     save_embeddings(embeddings, NAMESPACE)
#     save_chunks(chunks, NAMESPACE)
#     save_graph(graph, NAMESPACE)

#     return index, embeddings, chunks, graph


# # -------------------------
# # STREAMLIT UI
# # -------------------------

# st.set_page_config(layout="wide")
# st.title("📚 Abstract Journal GraphRAG")

# provider = st.selectbox("LLM Provider", ["openrouter", "lmstudio"])

# if "abstract_ready" not in st.session_state:
#     index, embeddings, chunks, graph = initialize_abstract_system()

#     st.session_state.index = index
#     st.session_state.embeddings = embeddings
#     st.session_state.chunks = chunks
#     st.session_state.graph = graph
#     st.session_state.abstract_ready = True


# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])


# user_input = st.chat_input("Ask something about these research papers...")

# if user_input and st.session_state.index is not None:

#     st.session_state.messages.append({"role": "user", "content": user_input})

#     response, sources = generate_answer(
#         user_input,
#         st.session_state.index,
#         st.session_state.embeddings,
#         st.session_state.chunks,
#         st.session_state.graph,
#         provider
#     )

#     st.session_state.messages.append({"role": "assistant", "content": response})
#     st.chat_message("assistant").write(response)

#     with st.expander("📑 Sources"):
#         for s in sources:
#             st.markdown(f"""
# **{s.get('title','')}**  
# Year: {s.get('year','')}  
# Journal: {s.get('journal','')}  
# DOI: {s.get('doi','')}  
# Cited By: {s.get('cited_by','')}  
# Tags: {s.get('tags','')}
# """)


# import streamlit as st
# from sentence_transformers import SentenceTransformer

# from modules.chunker import chunk_documents
# from modules.vector_store import build_or_update_index
# from modules.graph_builder import build_or_update_graph
# from modules.persistence import (
#     load_index,
#     save_index,
#     load_chunks,
#     save_chunks,
#     load_graph,
#     save_graph
# )
# from modules.rag_engine import generate_answer


# NAMESPACE = "abstract"

# st.title("📄 Abstract Journal GraphRAG")


# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# embedding_model = load_embedding_model()


# def initialize_system():
#     index = load_index(NAMESPACE)
#     chunks = load_chunks(NAMESPACE)
#     graph = load_graph(NAMESPACE)
#     return index, chunks, graph


# if "abstract_ready" not in st.session_state:
#     index, chunks, graph = initialize_system()
#     st.session_state.index = index
#     st.session_state.chunks = chunks
#     st.session_state.graph = graph
#     st.session_state.abstract_ready = True


# uploaded_file = st.file_uploader("Upload Abstract (.txt)", type=["txt"])

# if uploaded_file:

#     text = uploaded_file.read().decode("utf-8")

#     docs = [{
#         "doc_id": uploaded_file.name,
#         "page": 1,
#         "text": text
#     }]

#     new_chunks = chunk_documents(docs)

#     index, chunks = build_or_update_index(
#         index=st.session_state.index,
#         existing_chunks=st.session_state.chunks,
#         new_chunks=new_chunks,
#         embedding_model=embedding_model
#     )

#     graph = build_or_update_graph(
#         st.session_state.graph,
#         new_chunks
#     )

#     save_index(index, NAMESPACE)
#     save_chunks(chunks, NAMESPACE)
#     save_graph(graph, NAMESPACE)

#     st.session_state.index = index
#     st.session_state.chunks = chunks
#     st.session_state.graph = graph

#     st.success("Abstract indexed successfully!")


# query = st.text_input("Ask about abstracts")

# if query and st.session_state.index:

#     result = generate_answer(
#         query=query,
#         vector_index=st.session_state.index,
#         documents=st.session_state.chunks,
#         embedding_model=embedding_model,
#         provider="lmstudio",
#         model="mistralai/ministral-3-3b",
#         temperature=0.2,
#         top_k=5
#     )

#     st.write("### Answer")
#     st.write(result["response"])

#     with st.expander("Sources"):
#         for i, s in enumerate(result["sources"]):
#             st.write(
#                 f"{s['doc_id']} | Page {s['page']} | Score {result['scores'][i]:.4f}"
#             )


# import streamlit as st
# from sentence_transformers import SentenceTransformer

# from modules.chunker import chunk_documents
# from modules.vector_store import build_or_update_index
# from modules.graph_builder import build_or_update_graph
# from modules.persistence import (
#     load_index,
#     save_index,
#     load_chunks,
#     save_chunks,
#     load_graph,
#     save_graph
# )
# from modules.rag_engine import generate_answer


# st.title("📄 Abstract Journal GraphRAG")


# # -------------------------------------------------
# # LOAD EMBEDDING MODEL
# # -------------------------------------------------

# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# embedding_model = load_embedding_model()


# # -------------------------------------------------
# # INITIALIZATION
# # -------------------------------------------------

# def initialize_abstract_system():

#     index = load_index()
#     chunks = load_chunks()
#     graph = load_graph()

#     if chunks is None:
#         chunks = []

#     return index, chunks, graph


# if "abstract_system_ready" not in st.session_state:
#     index, chunks, graph = initialize_abstract_system()

#     st.session_state.abstract_index = index
#     st.session_state.abstract_chunks = chunks
#     st.session_state.abstract_graph = graph
#     st.session_state.abstract_system_ready = True


# # -------------------------------------------------
# # UPLOAD ABSTRACTS
# # -------------------------------------------------

# uploaded_file = st.file_uploader("Upload Abstract (txt)", type=["txt"])

# if uploaded_file:

#     text = uploaded_file.read().decode("utf-8")

#     docs = [{
#         "doc_id": uploaded_file.name,
#         "page": 1,
#         "text": text
#     }]

#     new_chunks = chunk_documents(docs)

#     index, chunks = build_or_update_index(
#         index=st.session_state.abstract_index,
#         existing_chunks=st.session_state.abstract_chunks,
#         new_chunks=new_chunks,
#         embedding_model=embedding_model
#     )

#     graph = build_or_update_graph(
#         st.session_state.abstract_graph,
#         new_chunks
#     )

#     save_index(index)
#     save_chunks(chunks)
#     save_graph(graph)

#     st.session_state.abstract_index = index
#     st.session_state.abstract_chunks = chunks
#     st.session_state.abstract_graph = graph

#     st.success("Abstract indexed successfully!")


# # -------------------------------------------------
# # QUERY ABSTRACTS
# # -------------------------------------------------

# query = st.text_input("Ask about uploaded abstracts")

# if query and st.session_state.abstract_index is not None:

#     result = generate_answer(
#         query=query,
#         vector_index=st.session_state.abstract_index,
#         documents=st.session_state.abstract_chunks,
#         embedding_model=embedding_model,
#         provider="lmstudio",   # you can extend to dynamic later
#         model="mistralai/ministral-3-3b",
#         temperature=0.2,
#         top_k=5
#     )

#     st.write("### Answer")
#     st.write(result["response"])

#     with st.expander("Sources"):
#         for i, s in enumerate(result["sources"]):
#             st.write(
#                 f"{s['doc_id']} | Page {s['page']} | Score: {result['scores'][i]:.4f}"
#             )


# import streamlit as st

# from modules.abstract_loader import (
#     get_abstract_folders,
#     load_abstract_folder
# )
# from modules.abstract_chunker import chunk_abstracts
# from modules.vector_store import build_or_update_index
# from modules.graph_builder import build_or_update_graph
# from modules.persistence import (
#     load_index,
#     save_index,
#     load_embeddings,
#     save_embeddings,
#     load_chunks,
#     save_chunks,
#     load_graph,
#     save_graph
# )
# from modules.rag_engine import generate_answer


# def initialize_abstract_system():

#     index = load_index()
#     embeddings = load_embeddings()
#     chunks = load_chunks()
#     graph = load_graph()

#     if chunks is None:
#         chunks = []

#     folders = get_abstract_folders()

#     for folder in folders:
#         docs = load_abstract_folder(folder)
#         new_chunks = chunk_abstracts(docs)

#         index, embeddings, chunks = build_or_update_index(
#             index,
#             embeddings,
#             chunks,
#             new_chunks
#         )

#         graph = build_or_update_graph(graph, new_chunks)

#     save_index(index)
#     save_embeddings(embeddings)
#     save_chunks(chunks)
#     save_graph(graph)

#     return index, embeddings, chunks, graph


# # -------------------------
# # STREAMLIT UI
# # -------------------------

# st.set_page_config(layout="wide")
# st.title("📚 Abstract Journal GraphRAG")

# provider = st.selectbox("LLM Provider", ["openrouter", "lmstudio"])

# if "abstract_ready" not in st.session_state:
#     index, embeddings, chunks, graph = initialize_abstract_system()

#     st.session_state.index = index
#     st.session_state.embeddings = embeddings
#     st.session_state.chunks = chunks
#     st.session_state.graph = graph
#     st.session_state.abstract_ready = True


# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])


# user_input = st.chat_input("Ask something about these papers...")

# if user_input and st.session_state.index is not None:

#     st.session_state.messages.append({"role": "user", "content": user_input})

#     response, sources = generate_answer(
#         user_input,
#         st.session_state.index,
#         st.session_state.embeddings,
#         st.session_state.chunks,
#         st.session_state.graph,
#         provider
#     )

#     st.session_state.messages.append({"role": "assistant", "content": response})
#     st.chat_message("assistant").write(response)

#     with st.expander("Sources"):
#         for s in sources:
#             st.write(
#                 f"{s['title']} ({s['year']})\n"
#                 f"Journal: {s['journal']}\n"
#                 f"DOI: {s['doi']}"
#             )
