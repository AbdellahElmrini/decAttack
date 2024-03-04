import networkx as nx
import matplotlib.pyplot as plt

# Create the graphs
G1 = nx.Graph()
G1.add_edges_from([(0, 1), (1, 2), (1, 3)])

G2 = nx.Graph()
G2.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2,4)])

G3 = nx.Graph()
G3.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4)])

G4 = nx.Graph()
G4.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)])

# Manually define the positions for each graph based on the figure
pos_G1 = {
    0: (0.5, 1),
    1: (0.5, 0.6),
    2: (0.3, 0.2),
    3: (0.7, 0.2)
}

pos_G2 = {
    0: (1, 2),
    1: (0.5, 1.5),
    2: (1.5, 1.5),
    3: (0, 1),
    4: (1, 1),
    5: (2, 1)
}

pos_G3 = {
    0: (0.5, 1),
    1: (0.5, 0.6),
    2: (0.3, 0.2),
    3: (0.7, 0.2),
    4: (0.5, 0.2),
    5: (0.9, -0.2)
}

pos_G4 = {
    0: (0.5, 1),
    1: (0.5, 0.6),
    2: (0.3, 0.2),
    3: (0.7, 0.2),
    4: (0.3, -0.2),
    5: (0.7, -0.2)
}

# Draw the graphs
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Graph G1
nx.draw(G1, pos=pos_G4, with_labels=True, ax=axes[0])
axes[0].title.set_text('G1')

# Graph G2
nx.draw(G2, pos=pos_G2, with_labels=True, ax=axes[1])
axes[1].title.set_text('G2')

# Graph G3
nx.draw(G3, pos=pos_G4, with_labels=True, ax=axes[2])
axes[2].title.set_text('G3')

# Graph G4
nx.draw(G4, pos=pos_G4, with_labels=True, ax=axes[3])
axes[3].title.set_text('G4')

plt.tight_layout()
plt.show()
