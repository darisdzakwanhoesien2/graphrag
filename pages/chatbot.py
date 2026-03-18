import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# paths
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATASET = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "new_dataset", "set3", "openalex_raw.jsonl"))
CHATLOG_PATH = os.path.normpath(os.path.join(PROJECT_ROOT, "data", "chatbot.jsonl"))
PAGES_DIR = os.path.dirname(__file__)
GRAPHRAG_DIR = os.path.normpath(os.path.join(PAGES_DIR, ".."))         # .../graphrag
REPO_ROOT = os.path.normpath(os.path.join(PAGES_DIR, "..", ".."))      # .../graphrag_system
DEFAULT_DATASET = os.path.normpath(os.path.join(REPO_ROOT, "new_dataset", "set3", "openalex_raw.jsonl"))
# fallback: older layout where new_dataset sits next to graphrag
if not os.path.exists(DEFAULT_DATASET):
    DEFAULT_DATASET = os.path.normpath(os.path.join(GRAPHRAG_DIR, "..", "new_dataset", "set3", "openalex_raw.jsonl"))
CHATLOG_PATH = os.path.normpath(os.path.join(GRAPHRAG_DIR, "data", "chatbot.jsonl"))


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("//") or ln.startswith("<!--"):
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            records.append(rec)
    return records


def reconstruct_abstract(rec: Dict[str, Any]) -> Optional[str]:
    aix = rec.get("abstract_inverted_index")
    if not aix:
        return None
    try:
        max_pos = 0
        for token, positions in aix.items():
            if positions:
                max_pos = max(max_pos, max(positions))
        tokens = [""] * (max_pos + 1)
        for token, positions in aix.items():
            for p in positions:
                tokens[p] = token
        text = " ".join(t for t in tokens if t)
        return text.strip()
    except Exception:
        return None


def make_corpus(records: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    corpus = []
    for rec in records:
        title = rec.get("display_name") or rec.get("title") or ""
        abstract = rec.get("abstract") or reconstruct_abstract(rec) or ""
        keywords = []
        for kw in rec.get("keywords", []):
            if isinstance(kw, dict):
                keywords.append(kw.get("display_name", ""))
            else:
                keywords.append(str(kw))
        text = " ".join([title, abstract, " ".join(keywords)])
        corpus.append({
            "id": rec.get("id"),
            "doi": rec.get("doi"),
            "title": title,
            "year": rec.get("publication_year"),
            "text": text.strip() or title
        })
    return corpus


class SimpleRetrievalBot:
    def __init__(self, docs: List[Dict[str, str]]):
        self.docs = docs
        texts = [d["text"] for d in docs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
        if texts:
            self.X = self.vectorizer.fit_transform(texts)
        else:
            self.X = None

    def query(self, q: str, top_k: int = 3):
        if self.X is None:
            return []
        v = self.vectorizer.transform([q])
        sims = cosine_similarity(v, self.X).ravel()
        idx = np.argsort(-sims)[:top_k]
        results = []
        for i in idx:
            results.append({"score": float(sims[i]), **self.docs[i]})
        return results


def append_chatlog(path: str, entry: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@st.cache_data
def cached_load_records(path: str) -> List[Dict[str, Any]]:
    return load_jsonl(path)


@st.cache_data
def cached_build_bot(records: List[Dict[str, Any]]):
    docs = make_corpus(records)
    bot = SimpleRetrievalBot(docs)
    return docs, bot


def format_results(results):
    if not results:
        return "No relevant documents found."
    out = []
    for r in results:
        snippet = (r.get("text") or "")[:400].replace("\n", " ").strip()
        out.append({
            "title": r.get("title"),
            "doi": r.get("doi"),
            "year": r.get("year"),
            "score": round(r.get("score", 0), 4),
            "snippet": snippet,
            "id": r.get("id"),
        })
    return out


def main():
    st.title("Graphrag — OpenAlex retrieval chatbot")
    st.sidebar.header("Settings")
    dataset_path = st.sidebar.text_input("Dataset path", value=DEFAULT_DATASET)
    top_k = st.sidebar.slider("Top K results", min_value=1, max_value=10, value=3)
    st.sidebar.markdown("Chats are appended to: `%s`" % CHATLOG_PATH)

    if not os.path.exists(dataset_path):
        st.error(f"Dataset not found: {dataset_path}")
        return

    records = cached_load_records(dataset_path)
    docs, bot = cached_build_bot(records)

    st.info(f"Loaded {len(docs)} documents. Enter a query below.")
    q = st.text_input("Query", key="query_input")
    submit = st.button("Search")

    if submit and q:
        results = bot.query(q, top_k=top_k)
        formatted = format_results(results)
        if isinstance(formatted, str):
            st.write(formatted)
        else:
            for r in formatted:
                st.markdown(f"**{r['title']}**  ")
                st.write(f"DOI: {r['doi']}  | Year: {r['year']}  | Score: {r['score']}")
                st.write(r["snippet"])
                st.markdown("---")
        # save chat turn
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": q,
            "results": formatted,
            "top_ids": [r.get("id") for r in results],
            "top_dois": [r.get("doi") for r in results],
        }
        append_chatlog(CHATLOG_PATH, entry)
        st.success("Chat saved to log.")

    if st.checkbox("Show recent chatlog"):
        if os.path.exists(CHATLOG_PATH):
            with open(CHATLOG_PATH, "r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            st.write(lines[-50:])  # show last 50
        else:
            st.info("No chatlog found.")


if __name__ == "__main__":
    main()