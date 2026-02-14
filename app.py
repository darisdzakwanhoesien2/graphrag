import streamlit as st
from sentence_transformers import SentenceTransformer
from datetime import datetime

from modules.data_loader import get_report_folders, load_report
from modules.chunker import chunk_documents
from modules.vector_store import build_or_update_index
from modules.graph_builder import build_or_update_graph
from modules.persistence import (
    load_index,
    save_index,
    load_chunks,
    save_chunks,
    load_graph,
    save_graph,
    load_processed_reports,
    save_processed_reports
)
from modules.rag_engine import generate_answer
from modules.history_store import append_history


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(layout="wide")
st.title("🌱 GraphRAG ESG Chatbot")


# -------------------------------------------------
# LOAD EMBEDDING MODEL (ONLY ONCE)
# -------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()


# -------------------------------------------------
# SIDEBAR SETTINGS
# -------------------------------------------------

st.sidebar.header("⚙️ LLM Settings")

provider = st.sidebar.selectbox(
    "Provider",
    ["lmstudio", "openrouter"]
)

if provider == "lmstudio":
    model = st.sidebar.selectbox(
        "LMStudio Model",
        ["mistralai/ministral-3-3b"]
    )
else:
    model = st.sidebar.selectbox(
        "OpenRouter Model",
        [
            "mistralai/devstral-2512:free",
            "tngtech/deepseek-r1t2-chimera:free",
            "qwen/qwen3-coder:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "openai/gpt-oss-120b:free",
            "google/gemini-2.0-flash-exp:free",
            "openai/gpt-oss-20b:free"
        ]
    )

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
top_k = st.sidebar.slider("Top K Retrieval", 1, 10, 5)


# -------------------------------------------------
# SYSTEM INITIALIZATION
# -------------------------------------------------

def initialize_system():

    processed = load_processed_reports()
    index = load_index()
    chunks = load_chunks()
    graph = load_graph()

    if chunks is None:
        chunks = []

    all_folders = get_report_folders()
    new_folders = [f for f in all_folders if f not in processed]

    if new_folders:
        st.write(f"📂 Processing new reports: {new_folders}")

        for folder in new_folders:
            docs = load_report(folder)
            new_chunks = chunk_documents(docs)

            index, chunks = build_or_update_index(
                index=index,
                existing_chunks=chunks,
                new_chunks=new_chunks,
                embedding_model=embedding_model
            )

            graph = build_or_update_graph(graph, new_chunks)
            processed.append(folder)

        save_index(index)
        save_chunks(chunks)
        save_graph(graph)
        save_processed_reports(processed)

    return index, chunks, graph


# -------------------------------------------------
# LOAD SYSTEM ONCE
# -------------------------------------------------

if "system_ready" not in st.session_state:

    index, chunks, graph = initialize_system()

    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.graph = graph
    st.session_state.system_ready = True


# -------------------------------------------------
# CHAT HISTORY (SESSION ONLY)
# -------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------------------------------------
# USER INPUT
# -------------------------------------------------

user_input = st.chat_input("Ask something about ESG...")

if user_input and st.session_state.index is not None:

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # Generate answer
    result = generate_answer(
        query=user_input,
        vector_index=st.session_state.index,
        documents=st.session_state.chunks,
        embedding_model=embedding_model,
        provider=provider,
        model=model,
        temperature=temperature,
        top_k=top_k
    )

    response = result["response"]
    sources = result["sources"]
    scores = result["scores"]

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    # Display response
    with st.chat_message("assistant"):
        st.write(response)

    # Display sources
    with st.expander("📚 Sources Used"):
        for i, s in enumerate(sources):
            st.write(
                f"{s['doc_id']} | Page {s['page']} | Score: {scores[i]:.4f}"
            )

    # -------------------------------------------------
    # STORE HISTORY (PERSISTENT)
    # -------------------------------------------------

    history_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": user_input,
        "response": response,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "top_k": top_k,
        "sources": [
            {
                "doc_id": s["doc_id"],
                "page": s["page"],
                "chunk_id": s["chunk_id"],
                "score": scores[i]
            }
            for i, s in enumerate(sources)
        ]
    }

    append_history(history_entry)


# import streamlit as st
# from sentence_transformers import SentenceTransformer

# from modules.data_loader import get_report_folders, load_report
# from modules.chunker import chunk_documents
# from modules.vector_store import build_or_update_index
# from modules.graph_builder import build_or_update_graph
# from modules.persistence import (
#     load_index,
#     save_index,
#     load_chunks,
#     save_chunks,
#     load_graph,
#     save_graph,
#     load_processed_reports,
#     save_processed_reports
# )
# from modules.rag_engine import generate_answer


# # -------------------------------------------------
# # PAGE CONFIG
# # -------------------------------------------------

# st.set_page_config(layout="wide")
# st.title("🌱 GraphRAG ESG Chatbot")


# # -------------------------------------------------
# # LOAD EMBEDDING MODEL (ONLY ONCE)
# # -------------------------------------------------

# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# embedding_model = load_embedding_model()


# # -------------------------------------------------
# # SIDEBAR SETTINGS
# # -------------------------------------------------

# st.sidebar.header("⚙️ LLM Settings")

# provider = st.sidebar.selectbox(
#     "Provider",
#     ["lmstudio", "openrouter"]
# )

# if provider == "lmstudio":
#     model = st.sidebar.selectbox(
#         "LMStudio Model",
#         ["mistralai/ministral-3-3b"]
#     )
# else:
#     model = st.sidebar.selectbox(
#         "OpenRouter Model",
#         [
#             "mistralai/devstral-2512:free",
#             "tngtech/deepseek-r1t2-chimera:free",
#             "qwen/qwen3-coder:free",
#             "nvidia/nemotron-3-nano-30b-a3b:free",
#             "meta-llama/llama-3.3-70b-instruct:free",
#             "openai/gpt-oss-120b:free",
#             "google/gemini-2.0-flash-exp:free",
#             "openai/gpt-oss-20b:free"
#         ]
#     )

# temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
# top_k = st.sidebar.slider("Top K Retrieval", 1, 10, 5)


# # -------------------------------------------------
# # SYSTEM INITIALIZATION
# # -------------------------------------------------

# def initialize_system():

#     processed = load_processed_reports()
#     index = load_index()
#     chunks = load_chunks()
#     graph = load_graph()

#     if chunks is None:
#         chunks = []

#     all_folders = get_report_folders()
#     new_folders = [f for f in all_folders if f not in processed]

#     if new_folders:
#         st.write(f"📂 Processing new reports: {new_folders}")

#         for folder in new_folders:
#             docs = load_report(folder)
#             new_chunks = chunk_documents(docs)

#             # Update vector index
#             index, chunks = build_or_update_index(
#                 index=index,
#                 existing_chunks=chunks,
#                 new_chunks=new_chunks,
#                 embedding_model=embedding_model
#             )

#             # Update graph
#             graph = build_or_update_graph(graph, new_chunks)

#             processed.append(folder)

#         # Persist updated state
#         save_index(index)
#         save_chunks(chunks)
#         save_graph(graph)
#         save_processed_reports(processed)

#     return index, chunks, graph


# # -------------------------------------------------
# # LOAD SYSTEM ONCE
# # -------------------------------------------------

# if "system_ready" not in st.session_state:

#     index, chunks, graph = initialize_system()

#     st.session_state.index = index
#     st.session_state.chunks = chunks
#     st.session_state.graph = graph
#     st.session_state.system_ready = True


# # -------------------------------------------------
# # CHAT HISTORY
# # -------------------------------------------------

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])


# # -------------------------------------------------
# # USER INPUT
# # -------------------------------------------------

# user_input = st.chat_input("Ask something about ESG...")

# if user_input and st.session_state.index is not None:

#     # Save user message
#     st.session_state.messages.append(
#         {"role": "user", "content": user_input}
#     )

#     # Generate answer
#     result = generate_answer(
#         query=user_input,
#         vector_index=st.session_state.index,
#         documents=st.session_state.chunks,
#         embedding_model=embedding_model,
#         provider=provider,
#         model=model,
#         temperature=temperature,
#         top_k=top_k
#     )

#     response = result["response"]
#     sources = result["sources"]

#     # Save assistant message
#     st.session_state.messages.append(
#         {"role": "assistant", "content": response}
#     )

#     # Display assistant response
#     with st.chat_message("assistant"):
#         st.write(response)

#     # Show sources
#     with st.expander("📚 Sources Used"):
#         for s in sources:
#             st.write(f"{s['doc_id']} | Page {s['page']}")


# import streamlit as st
# from sentence_transformers import SentenceTransformer

# from modules.data_loader import get_report_folders, load_report
# from modules.chunker import chunk_documents
# from modules.vector_store import build_or_update_index
# from modules.graph_builder import build_or_update_graph
# from modules.persistence import (
#     load_index,
#     save_index,
#     load_chunks,
#     save_chunks,
#     load_graph,
#     save_graph,
#     load_processed_reports,
#     save_processed_reports
# )
# from modules.rag_engine import generate_answer


# # -------------------------------------------------
# # PAGE CONFIG
# # -------------------------------------------------

# st.set_page_config(layout="wide")
# st.title("🌱 GraphRAG ESG Chatbot")


# # -------------------------------------------------
# # EMBEDDING MODEL (SEPARATE FROM STORED VECTORS)
# # -------------------------------------------------

# @st.cache_resource
# def load_embedding_model():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# embedding_model = load_embedding_model()


# # -------------------------------------------------
# # SIDEBAR SETTINGS
# # -------------------------------------------------

# st.sidebar.header("⚙️ LLM Settings")

# provider = st.sidebar.selectbox(
#     "Provider",
#     ["lmstudio", "openrouter"]
# )

# if provider == "lmstudio":
#     model = st.sidebar.selectbox(
#         "LMStudio Model",
#         ["mistralai/ministral-3-3b"]
#     )
# else:
#     model = st.sidebar.selectbox(
#         "OpenRouter Model",
#         [
#             "mistralai/devstral-2512:free",
#             "tngtech/deepseek-r1t2-chimera:free",
#             "qwen/qwen3-coder:free",
#             "nvidia/nemotron-3-nano-30b-a3b:free",
#             "meta-llama/llama-3.3-70b-instruct:free",
#             "openai/gpt-oss-120b:free",
#             "google/gemini-2.0-flash-exp:free",
#             "openai/gpt-oss-20b:free"
#         ]
#     )

# temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
# top_k = st.sidebar.slider("Top K Retrieval", 1, 10, 5)


# # -------------------------------------------------
# # SYSTEM INITIALIZATION
# # -------------------------------------------------

# def initialize_system():

#     processed = load_processed_reports()
#     index = load_index()
#     chunks = load_chunks()
#     graph = load_graph()

#     if chunks is None:
#         chunks = []

#     all_folders = get_report_folders()
#     new_folders = [f for f in all_folders if f not in processed]

#     if new_folders:
#         st.write(f"Processing new reports: {new_folders}")

#         for folder in new_folders:
#             docs = load_report(folder)
#             new_chunks = chunk_documents(docs)

#             index, chunks = build_or_update_index(
#                 index,
#                 chunks,
#                 new_chunks,
#                 embedding_model
#             )

#             graph = build_or_update_graph(graph, new_chunks)
#             processed.append(folder)

#         save_index(index)
#         save_chunks(chunks)
#         save_graph(graph)
#         save_processed_reports(processed)

#     return index, chunks, graph


# if "system_ready" not in st.session_state:
#     index, chunks, graph = initialize_system()

#     st.session_state.index = index
#     st.session_state.chunks = chunks
#     st.session_state.graph = graph
#     st.session_state.system_ready = True


# # -------------------------------------------------
# # CHAT HISTORY
# # -------------------------------------------------

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])


# # -------------------------------------------------
# # USER INPUT
# # -------------------------------------------------

# user_input = st.chat_input("Ask something about ESG...")

# if user_input and st.session_state.index is not None:

#     st.session_state.messages.append(
#         {"role": "user", "content": user_input}
#     )

#     result = generate_answer(
#         query=user_input,
#         vector_index=st.session_state.index,
#         documents=st.session_state.chunks,
#         embedding_model=embedding_model,
#         provider=provider,
#         model=model,
#         temperature=temperature,
#         top_k=top_k
#     )

#     response = result["response"]
#     sources = result["sources"]

#     st.session_state.messages.append(
#         {"role": "assistant", "content": response}
#     )

#     st.chat_message("assistant").write(response)

#     with st.expander("📚 Sources"):
#         for s in sources:
#             st.write(f"{s['doc_id']} | Page {s['page']}")


# import streamlit as st

# from modules.data_loader import get_report_folders, load_report
# from modules.chunker import chunk_documents
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
#     save_graph,
#     load_processed_reports,
#     save_processed_reports
# )
# from modules.rag_engine import generate_answer


# # -------------------------
# # PAGE CONFIG
# # -------------------------

# st.set_page_config(layout="wide")
# st.title("🌱 GraphRAG ESG Chatbot")


# # -------------------------
# # SIDEBAR SETTINGS
# # -------------------------

# st.sidebar.header("⚙️ LLM Settings")

# provider = st.sidebar.selectbox(
#     "Provider",
#     ["lmstudio", "openrouter"]
# )

# if provider == "lmstudio":
#     model = st.sidebar.selectbox(
#         "LMStudio Model",
#         ["mistralai/ministral-3-3b"]
#     )

# elif provider == "openrouter":
#     model = st.sidebar.selectbox(
#         "OpenRouter Model",
#         [
#             "mistralai/devstral-2512:free",
#             "tngtech/deepseek-r1t2-chimera:free",
#             "qwen/qwen3-coder:free",
#             "nvidia/nemotron-3-nano-30b-a3b:free",
#             "meta-llama/llama-3.3-70b-instruct:free",
#             "openai/gpt-oss-120b:free",
#             "google/gemini-2.0-flash-exp:free",
#             "openai/gpt-oss-20b:free"
#         ]
#     )

# temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
# top_k = st.sidebar.slider("Top K Retrieval", 1, 10, 5)


# # -------------------------
# # SYSTEM INITIALIZATION
# # -------------------------

# def initialize_system():

#     processed = load_processed_reports()
#     index = load_index()
#     embeddings = load_embeddings()
#     chunks = load_chunks()
#     graph = load_graph()

#     if chunks is None:
#         chunks = []

#     all_folders = get_report_folders()
#     new_folders = [f for f in all_folders if f not in processed]

#     if new_folders:
#         st.write(f"Processing new reports: {new_folders}")

#         for folder in new_folders:
#             docs = load_report(folder)
#             new_chunks = chunk_documents(docs)

#             index, embeddings, chunks = build_or_update_index(
#                 index,
#                 embeddings,
#                 chunks,
#                 new_chunks
#             )

#             graph = build_or_update_graph(graph, new_chunks)

#             processed.append(folder)

#         save_index(index)
#         save_embeddings(embeddings)
#         save_chunks(chunks)
#         save_graph(graph)
#         save_processed_reports(processed)

#     return index, embeddings, chunks, graph


# # Initialize only once
# if "system_ready" not in st.session_state:
#     index, embeddings, chunks, graph = initialize_system()

#     st.session_state.index = index
#     st.session_state.embeddings = embeddings
#     st.session_state.chunks = chunks
#     st.session_state.graph = graph
#     st.session_state.system_ready = True


# # -------------------------
# # CHAT HISTORY
# # -------------------------

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])


# # -------------------------
# # USER INPUT
# # -------------------------

# user_input = st.chat_input("Ask something about ESG...")

# if user_input and st.session_state.index is not None:

#     st.session_state.messages.append(
#         {"role": "user", "content": user_input}
#     )

#     result = generate_answer(
#         query=user_input,
#         vector_index=st.session_state.index,
#         documents=st.session_state.chunks,
#         embeddings=st.session_state.embeddings,
#         provider=provider,
#         model=model,
#         temperature=temperature,
#         top_k=top_k
#     )

#     response = result["response"]
#     sources = result["sources"]

#     st.session_state.messages.append(
#         {"role": "assistant", "content": response}
#     )

#     st.chat_message("assistant").write(response)

#     with st.expander("📚 Sources"):
#         for s in sources:
#             st.write(f"{s['doc_id']} | Page {s['page']}")


# import streamlit as st

# from modules.data_loader import get_report_folders, load_report
# from modules.chunker import chunk_documents
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
#     save_graph,
#     load_processed_reports,
#     save_processed_reports
# )
# from modules.rag_engine import generate_answer


# # -------------------------
# # SYSTEM INITIALIZATION
# # -------------------------

# def initialize_system():

#     processed = load_processed_reports()
#     index = load_index()
#     embeddings = load_embeddings()
#     chunks = load_chunks()
#     graph = load_graph()

#     if chunks is None:
#         chunks = []

#     all_folders = get_report_folders()
#     new_folders = [f for f in all_folders if f not in processed]

#     if new_folders:
#         st.write(f"Processing new reports: {new_folders}")

#         for folder in new_folders:
#             docs = load_report(folder)
#             new_chunks = chunk_documents(docs)

#             index, embeddings, chunks = build_or_update_index(
#                 index,
#                 embeddings,
#                 chunks,
#                 new_chunks
#             )

#             graph = build_or_update_graph(graph, new_chunks)

#             processed.append(folder)

#         save_index(index)
#         save_embeddings(embeddings)
#         save_chunks(chunks)
#         save_graph(graph)
#         save_processed_reports(processed)

#     return index, embeddings, chunks, graph


# # -------------------------
# # STREAMLIT UI
# # -------------------------

# st.set_page_config(layout="wide")
# st.title("GraphRAG ESG Chatbot")

# provider = st.selectbox("LLM Provider", ["openrouter", "lmstudio"])

# # Initialize once
# if "system_ready" not in st.session_state:
#     index, embeddings, chunks, graph = initialize_system()

#     st.session_state.index = index
#     st.session_state.embeddings = embeddings
#     st.session_state.chunks = chunks
#     st.session_state.graph = graph
#     st.session_state.system_ready = True


# # Chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])


# # User input
# user_input = st.chat_input("Ask something about ESG...")

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
#             st.write(f"{s['doc_id']} | Page {s['page']}")


# import streamlit as st

# from modules.data_loader import get_report_folders, load_report
# from modules.chunker import chunk_documents
# from modules.vector_store import build_or_update_faiss
# from modules.graph_builder import build_or_update_graph
# from modules.persistence import (
#     load_faiss,
#     save_faiss,
#     load_chunks,
#     save_chunks,
#     load_graph,
#     save_graph,
#     load_processed_reports,
#     save_processed_reports
# )
# from modules.rag_engine import generate_answer


# def initialize_system():

#     processed = load_processed_reports()
#     index = load_faiss()
#     chunks = load_chunks()
#     graph = load_graph()

#     all_folders = get_report_folders()
#     new_folders = [f for f in all_folders if f not in processed]

#     if new_folders:
#         st.write(f"Processing new reports: {new_folders}")

#         for folder in new_folders:
#             docs = load_report(folder)
#             new_chunks = chunk_documents(docs)

#             index, chunks = build_or_update_faiss(index, chunks, new_chunks)
#             graph = build_or_update_graph(graph, new_chunks)

#             processed.append(folder)

#         save_faiss(index)
#         save_chunks(chunks)
#         save_graph(graph)
#         save_processed_reports(processed)

#     return index, chunks, graph


# # -------------------------
# # STREAMLIT UI
# # -------------------------

# st.set_page_config(layout="wide")
# st.title("GraphRAG ESG Chatbot")

# provider = st.selectbox("LLM Provider", ["openrouter", "lmstudio"])

# if "system_ready" not in st.session_state:
#     index, chunks, graph = initialize_system()
#     st.session_state.index = index
#     st.session_state.chunks = chunks
#     st.session_state.graph = graph
#     st.session_state.system_ready = True

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# user_input = st.chat_input("Ask something about ESG...")

# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     response, sources = generate_answer(
#         user_input,
#         st.session_state.index,
#         st.session_state.chunks,
#         st.session_state.graph,
#         provider
#     )

#     st.session_state.messages.append({"role": "assistant", "content": response})
#     st.chat_message("assistant").write(response)

#     with st.expander("Sources"):
#         for s in sources:
#             st.write(f"{s['doc_id']} | Page {s['page']}")


# from modules.data_loader import get_report_folders, load_report
# from modules.chunker import chunk_documents
# from modules.vector_store import build_or_update_index
# from modules.graph_builder import build_or_update_graph
# from modules.persistence import *

# def initialize_system():

#     processed = load_metadata()
#     existing_index = load_index()
#     existing_graph = load_graph()

#     all_folders = get_report_folders()
#     new_folders = [f for f in all_folders if f not in processed]

#     if new_folders:
#         print("Processing new folders:", new_folders)

#         for folder in new_folders:
#             docs = load_report(folder)
#             chunks = chunk_documents(docs)

#             existing_index = build_or_update_index(existing_index, chunks)
#             existing_graph = build_or_update_graph(existing_graph, chunks)

#             processed.append(folder)

#         save_index(existing_index)
#         save_graph(existing_graph)
#         save_metadata(processed)

#     return existing_index, existing_graph


# import streamlit as st

# st.title("GraphRAG ESG Chatbot")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# user_input = st.chat_input("Ask something about ESG...")

# if user_input:
#     st.session_state.messages.append({"role": "user", "content": user_input})
    
#     response = generate_answer(user_input)
    
#     st.session_state.messages.append({"role": "assistant", "content": response})
#     st.chat_message("assistant").write(response)