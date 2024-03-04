import networkx as nx


def test_graph():
    G = nx.Graph()
    for i in range(6):
        G.add_node(i)
    G.add_edge(0,1)
    G.add_edge(0,2)
    G.add_edge(1,3)
    G.add_edge(1,4)
    G.add_edge(2,4)
    G.add_edge(2,5)
    return G


def line_graph(n):
    G = nx.Graph()
    G.add_node(0)
    for i in range(1,n):
        G.add_node(i)
        G.add_edge(i-1, i)

    return G