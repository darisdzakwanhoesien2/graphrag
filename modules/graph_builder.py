import spacy
import networkx as nx

nlp = spacy.load("en_core_web_sm")

def build_or_update_graph(existing_graph, chunks):

    G = existing_graph if existing_graph else nx.Graph()

    for chunk in chunks:
        doc = nlp(chunk["text"])
        entities = [ent.text for ent in doc.ents]

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                G.add_edge(
                    entities[i],
                    entities[j],
                    doc_id=chunk["doc_id"],
                    page=chunk["page"]
                )

    return G

def build_graph_from_documents(docs):
    G = nx.Graph()

    for doc_id, text in docs.items():
        parsed = nlp(text)
        entities = [ent.text for ent in parsed.ents]

        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                G.add_edge(entities[i], entities[j], source=doc_id)

    return G
def build_global_graph(chunks):
    G = nx.Graph()

    for chunk in chunks:
        doc = nlp(chunk["text"])
        entities = [ent.text for ent in doc.ents]

        for i in range(len(entities)):
            for j in range(i+1, len(entities)):
                G.add_edge(
                    entities[i],
                    entities[j],
                    doc_id=chunk["doc_id"],
                    page=chunk["page"]
                )

    return G