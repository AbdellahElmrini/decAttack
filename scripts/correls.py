import sys
sys.path.append("src")
from consensus import *
import networkx as nx
import numpy as np
import scipy.stats as stats

def run_experiment(seed):
    np.random.seed(seed)

    # Step 1: Generate a fully connected Erdős-Rényi graph
    while True:
        G = nx.erdos_renyi_graph(50, 0.08)
        if nx.is_connected(G):
            try:
                eigenvector = nx.eigenvector_centrality(G)[0] # because sometimes the computation diverge
                # Step 2: Run reconstruction process
    
                attacker_nodes = [0]
                R = Reconstruct(G, 10, attacker_nodes) # because of numerical instability
                U, x, x_hat = R.reconstruction()
                break
            except _:
                continue  # Skip this graph and generate a new one
        

    # Step 3: Compute centralities for node 0
    betweenness = nx.betweenness_centrality(G)[0]
    eigenvector = nx.eigenvector_centrality(G)[0]
    degree = nx.degree_centrality(G)[0]

    # Step 4: Compute similarity vectors
    shortest_path = np.array([nx.shortest_path_length(G, source=0, target=i) for i in range(G.number_of_nodes())])
    communicability_scores = nx.communicability(G)
    communicability = np.array([communicability_scores[0][i] for i in range(G.number_of_nodes())])

    # Step 5: Create reconstructible vector
    reconstructible = np.where(np.isnan(x_hat), 0, 1)

    return shortest_path, communicability, reconstructible, betweenness, eigenvector, degree, np.mean(reconstructible)



def main():
    num_runs = 500
    all_shortest_paths, all_communicability, all_reconstructible = [], [], []
    betweennesses, eigenvectors, degrees, proportions = [], [], [], []
    kendall_shortest_path, kendall_communicability = [], []

    for seed in range(num_runs):
        sp, com, rec, bet, eig, deg, prop = run_experiment(seed)
        all_shortest_paths.append(sp)
        all_communicability.append(com)
        all_reconstructible.append(rec)
        betweennesses.append(bet)
        eigenvectors.append(eig)
        degrees.append(deg)
        proportions.append(prop)

        # Compute and append Kendall Tau correlations for each run
        kendall_shortest_path.append(stats.kendalltau(rec, sp).correlation)
        kendall_communicability.append(stats.kendalltau(rec, com).correlation)

    # Calculate mean, standard deviation, and NaN count for Kendall Tau correlations
    mean_kendall_shortest_path = np.nanmean(kendall_shortest_path)
    std_kendall_shortest_path = np.nanstd(kendall_shortest_path)
    nan_count_shortest_path = np.count_nonzero(np.isnan(kendall_shortest_path))

    mean_kendall_communicability = np.nanmean(kendall_communicability)
    std_kendall_communicability = np.nanstd(kendall_communicability)
    nan_count_communicability = np.count_nonzero(np.isnan(kendall_communicability))

    # Compute Spearman Correlations for centralities and proportions
    spearman_betweenness = stats.spearmanr(proportions, betweennesses).correlation
    spearman_eigenvector = stats.spearmanr(proportions, eigenvectors).correlation
    spearman_degree = stats.spearmanr(proportions, degrees).correlation

    # Print results
    print(f"Kendall Tau Correlation with Shortest Path: Mean = {mean_kendall_shortest_path}, STD = {std_kendall_shortest_path}, NaN count = {nan_count_shortest_path}")
    print(f"Kendall Tau Correlation with Communicability: Mean = {mean_kendall_communicability}, STD = {std_kendall_communicability}, NaN count = {nan_count_communicability}")
    print(f"Spearman Correlation with Betweenness Centrality: {spearman_betweenness}")
    print(f"Spearman Correlation with Eigenvector Centrality: {spearman_eigenvector}")
    print(f"Spearman Correlation with Degree Centrality: {spearman_degree}")

if __name__ == "__main__":
    main()