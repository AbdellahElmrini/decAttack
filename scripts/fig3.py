import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Load the saved 'props' array
props = np.load('props.npy')

# Load the single version
colors = np.load('colors.npy')
attackers = np.load('attackers.npy')
attackers = ['red' if i==attackers else 'xkcd:silver' for i in range(len(colors))]

# Load the graph
ego = 414
name_edgelist = "facebook/"+str(ego)+".edges"
my_graph = nx.read_edgelist(name_edgelist)
my_graph = nx.relabel_nodes(my_graph, lambda x: int(x))
Gcc = sorted(nx.connected_components(my_graph), key=len, reverse=True)
G0 = my_graph.subgraph(Gcc[0]).copy()
G0 = nx.convert_node_labels_to_integers(G0, label_attribute="fb_id")

# compute the layout that will be used for the two graphs
pos = nx.spring_layout(G0)

# Plot the graph with a different colormap
plt.figure(figsize=(6, 8))
plt.subplot(211)

ax = plt.gca()  # Get current axes

# Use a different colormap, e.g., 'plasma'
nodes = nx.draw(G0, ax=ax,pos=pos, node_color=props, node_size=40, alpha=0.7, 
                edgecolors='xkcd:silver', edge_color='xkcd:silver', 
                width=.5, cmap=plt.cm.Spectral_r)
# Create a ScalarMappable with the same normalization and colormap
norm = plt.Normalize(vmin=min(props), vmax=max(props))
sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r, norm=norm)
sm.set_array([])

# Draw the colorbar, associated with the current Axes
plt.colorbar(sm, ax=ax)

plt.subplot(212)
nx.draw(G0, pos=pos, node_color=colors, node_size=40, alpha=0.7,  edgecolors=attackers,edge_color='xkcd:silver', width=.5, cmap=plt.cm.viridis_r)

plt.savefig("both_color_ego.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
