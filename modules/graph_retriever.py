def expand_graph_context(G, entities, depth=1):
    expanded = set()

    for ent in entities:
        if ent in G:
            neighbors = nx.single_source_shortest_path_length(G, ent, cutoff=depth)
            expanded.update(neighbors.keys())

    return list(expanded)