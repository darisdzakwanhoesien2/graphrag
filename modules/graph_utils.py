import networkx as nx
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def compute_centrality(graph):
    if graph is None:
        return {}
    return nx.degree_centrality(graph)

def graph_proximity_score(graph, query_entities, chunk_entities, depth=2):
    score = 0
    for qe in query_entities:
        if qe in graph:
            lengths = nx.single_source_shortest_path_length(graph, qe, cutoff=depth)
            for ce in chunk_entities:
                if ce in lengths:
                    score += 1 / (lengths[ce] + 1)
    return score
