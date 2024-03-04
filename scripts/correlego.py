
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors as colors
from random import seed
import sys
sys.path.append("src")
from consensus import *
import scipy.stats as stats

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 28})


# usual graph pre-processing
seed(0)
np.random.seed(1)
ego = 414

name_edgelist = "facebook/"+str(ego)+".edges"
my_graph = nx.read_edgelist(name_edgelist)
my_graph = nx.relabel_nodes(my_graph, lambda x:int(x))
Gcc = sorted(nx.connected_components(my_graph), key=len, reverse=True)
G = my_graph.subgraph(Gcc[0]).copy()
n = G.number_of_nodes()
G = nx.convert_node_labels_to_integers(G, label_attribute="fb_id")
G.remove_edges_from(nx.selfloop_edges(G))


# load who is reconstructible
reconstruction = np.load('alltoall.npy')
proportions = np.load('props.npy')
print(reconstruction)
# compute the metric needed
betweenness = [nx.betweenness_centrality(G)[node] for node in range(n)]
eigenvector = [nx.eigenvector_centrality(G)[node] for node in range(n)]
degree = [nx.degree_centrality(G)[node] for node in range(n)]

# Step 4: Compute similarity vectors
all_shortest_paths = dict(nx.shortest_path_length(G))
shortest_path = np.array([[all_shortest_paths[i][j] for i in range(n)] for j in range(n)])
com_scores = nx.communicability(G)

communicability_scores = np.array([[com_scores[i][j] for i in range(n)] for j in range(n)])

# Compute the statistics
k_sp = np.zeros(n)
k_com = np.zeros(n)
for node in range(n):
    k_sp[node] =  stats.kendalltau(reconstruction[node], shortest_path[node]).correlation
    k_com[node] = stats.kendalltau(reconstruction[node], communicability_scores[node]).correlation

# Calculate mean, standard deviation, and NaN count for Kendall Tau correlations
mean_k_sp = np.nanmean(k_sp)
std_k_sp = np.nanstd(k_sp)
nan_count_shortest_path = np.count_nonzero(np.isnan(k_sp))

mean_k_com = np.nanmean(k_com)
std_k_com = np.nanstd(k_com)
nan_count_communicability = np.count_nonzero(np.isnan(k_com))

# Print results
print(f"Kendall Tau Correlation with Shortest Path: Mean = {mean_k_sp}, STD = {std_k_sp}, NaN count = {nan_count_shortest_path}")
print(f"Kendall Tau Correlation with Communicability: Mean = {mean_k_com}, STD = {std_k_com}, NaN count = {nan_count_communicability}")

# for the props
# Compute Spearman Correlations for centralities and proportions
spearman_betweenness = stats.spearmanr(proportions, betweenness).correlation
spearman_eigenvector = stats.spearmanr(proportions, eigenvector).correlation
spearman_degree = stats.spearmanr(proportions, degree).correlation

print(f"Spearman Correlation with Betweenness Centrality: {spearman_betweenness}")
print(f"Spearman Correlation with Eigenvector Centrality: {spearman_eigenvector}")
print(f"Spearman Correlation with Degree Centrality: {spearman_degree}")