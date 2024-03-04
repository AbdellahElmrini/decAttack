
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors as colors
from random import seed
import sys
sys.path.append("src")
from consensus import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 28})

seed(0)
np.random.seed(1)

ego = 414

# Same preprocessing as in muffliato
name_edgelist = "facebook/"+str(ego)+".edges"
my_graph = nx.read_edgelist(name_edgelist)
my_graph = nx.relabel_nodes(my_graph, lambda x:int(x))
Gcc = sorted(nx.connected_components(my_graph), key=len, reverse=True)
G0 = my_graph.subgraph(Gcc[0]).copy()
n = G0.number_of_nodes()
print(n)
G0 = nx.convert_node_labels_to_integers(G0, label_attribute="fb_id")


props = np.zeros(n)
alltoall = np.zeros((n,n))
for attacker in range(n):
    # chose an attacker
    print(attacker)
    # run the reconstruction
    try:
        R = Reconstruct(G0, 10, [attacker])
        U, x, x_hat = R.reconstruction()
    except Exception:
        print('should skip 1')
        try:
            R = Reconstruct(G0, 4, [attacker])
            U, x, x_hat = R.reconstruction()
        except Exception:
            print("really skip")
            continue
    reconstructible = np.where(np.isnan(x_hat), 0, 1)
    alltoall[attacker] = reconstructible
    props[attacker] = np.sum(reconstructible)


plt.figure(figsize=(6,8))
print(alltoall)

np.save('props.npy', props)
np.save('alltoall.npy', alltoall)

G0.remove_edges_from(nx.selfloop_edges(G0))
nx.draw(G0, node_color=props, node_size=20, alpha=0.5, edgecolors='xkcd:silver', edge_color='xkcd:silver', width=.5, cmap=plt.cm.viridis_r)
plt.colorbar(plt.cm.ScalarMappable(norm=None, cmap = plt.cm.Spectral_r))


plt.savefig("alltoallego.pdf", bbox_inches='tight', pad_inches=0)
plt.show()