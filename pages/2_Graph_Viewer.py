import streamlit as st
from pyvis.network import Network
from modules.persistence import load_graph

st.title("Knowledge Graph Viewer")

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
