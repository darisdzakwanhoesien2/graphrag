import streamlit as st
from modules.history_store import load_history

st.title("📜 Query History")

history = load_history()

if history:
    for entry in reversed(history):
        with st.expander(f"🕒 {entry['timestamp']}"):
            st.write("**Query:**", entry["query"])
            st.write("**Response:**", entry["response"])
            st.write("**Model:**", entry["model"])
            st.write("**Provider:**", entry["provider"])
            st.write("**Sources Used:**")
            for s in entry["sources"]:
                st.write(f"- {s['doc_id']} | Page {s['page']} | Score {s['score']:.4f}")
else:
    st.write("No history stored yet.")
