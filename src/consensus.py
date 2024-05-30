import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import sys

from fractions import Fraction
    
try:
    from src.rref import *
    from src.utils import *
except:
    from rref import *
    from utils import *


class Reconstruct():

    def __init__(self, G, n_iter, attacker_nodes, gossip_matrix = HamiltonGossipMatrix, use_fractions=False):
        """
        A class to reconstruct the intial values used in a consensus averaging algorithm
        G: networkx graph, we require the nodes to be indexed from 0 to n-1
        n_iter: number of gossip iterations n_iter >= 1
        attacker_nodes: indices of the attacker nodes
        gossip_matrix: function that returns the gossip matrix of the graph
        """
        self.G = G 
        self.n_iter = n_iter
        self.attacker_nodes = attacker_nodes 
        self.W = gossip_matrix(self.G)
        if use_fractions:
            self.W= gossip_matrix(self.G, use_fractions=True)
        self.build_knowledge_matrix()
        self.use_fractions = use_fractions

    def build_knowledge_matrix(self):
        W  = self.W
        att_matrix = []
        for node in self.attacker_nodes:
            # The attackers know their own values
            att_matrix.append(np.eye(1,len(W),node)[0])
        for neighbor in get_neighbors(self.G, self.attacker_nodes):
                att_matrix.append(np.eye(1,len(W),neighbor)[0])
        for _ in range(1, self.n_iter):
            for neighbor in get_neighbors(self.G, self.attacker_nodes):
                att_matrix.append(W[neighbor])
            W = self.W @ W
        self.knowledge_matrix = np.array(att_matrix)
        return self.knowledge_matrix
    
    def rank(self):
        return np.linalg.matrix_rank(self.knowledge_matrix)
    
    def indistinguishable_nodes(self):
        #TODO : add the general case when a submatrix is not of full rank, rename otherwise 
        # Returns pairs of nodes the attacker cannot distinguish
        corrMatrix = np.corrcoef(np.array(self.knowledge_matrix).T)-np.eye(len(self.G))
        indistinct = np.transpose(np.nonzero(corrMatrix==1))
        indistinct = [tuple(e) for e in indistinct if e[0]<e[1]]
        # Adding constant columns
        constant  = self.same_columns()
        indistinct = indistinct + constant
        return indistinguiable_groups(indistinct)
    
    def indistinguishable_nodes_set(self):
        # TODO Add this in utils
        res = set()
        for a in self.indistinguishable_nodes():
            res.add(tuple(sorted(a)))
        return res
    
    def same_columns(self):
    # nodes are indistinguishable with respect to the attacker if they have the same columns in the knowledge matrix
        indistinct = []
        for node1 in self.G.nodes():
            for node2 in self.G.nodes():
                if node1 < node2:
                    if np.isclose(self.knowledge_matrix[:,node1],self.knowledge_matrix[:,node2]).all():
                        indistinct.append((node1,node2))
        return indistinct

    def LU_reconstruction_example(self, X=None):
        """
        A reconstruction example using the LU method.
        If X is provided, it is used as the initial input vector for gossip.
        Otherwise, a random X is generated.

        Returns:
        x : an input vector used in gossip
        v : the vector to which attackers have access at the end of the gossip
        x_hat : an estimation of the input vector, non-computed entries denoted by 'nc'
        combs : a list of tuples, each tuple contains three elements:
            idx : the indices of input elements 
            coefs : the mixing coefficients
            val : the value of the aggregate of the inputs x[idx] mixed with coefs 
        """


        P1, L1, U1 = sp.linalg.lu(self.knowledge_matrix) # U is the Row Echelon Form of the knowledge matrix
        print('the know')
        print(self.knowledge_matrix)


        print("LU shape", P1.shape, L1.shape, U1.shape)
        P, L, U = rref_transform(P1, L1, U1) # Now U is the Reduced Row Echelon Form of the knowledge matrix
        print("RREF shape", P.shape, L.shape, U.shape)
        


        if True:
            assert np.allclose(P @ L @ U,  P1 @ L1 @U1)
            print("Us", np.linalg.matrix_rank(U1), np.linalg.matrix_rank(U))
            print("Ls", np.linalg.matrix_rank(L1), np.linalg.matrix_rank(L))



        x, v = self.generate_gossip_example(X)

        assert np.allclose(self.knowledge_matrix @ x, v), "Construction gossip"
        z = np.linalg.pinv(P @ L) @ v
        x_hat = [np.NaN for i in range(len(x))]

        combs = []
        for i in range(len(U)):
            if count_non_zero_elements(U[i]) == 0:
                continue
            if count_non_zero_elements(U[i]) == 1:
                j = first_nonzero(U[i])
                x_hat[j] = z[i]/U[i][j] #i or j
            else: 
                idx = nonzero_indices(U[i])
                coefs = U[i][idx]
                val = z[i]
                combs.append((idx, coefs, val))
        return P, L, U, x, v, x_hat, combs


    def reconstruction(self, X=None):
        T, R, U = compute_rref(self.knowledge_matrix, use_fractions=self.use_fractions)
        #U = U.astype(float)
        print(U.shape)
        U[np.abs(U) < 1e-6] = 0
        T[np.abs(T) < 1e-6] = 0

        x, y = self.generate_gossip_example(X)
        if self.use_fractions:
            assert np.allclose(matrix_multiply(self.knowledge_matrix, x), y), "Construction gossip"
        else: 
            assert np.allclose(self.knowledge_matrix @ x, y), "Construction gossip"
        x_hat = [np.NaN for i in range(len(x))]
        z = T @ y
        for i in range(len(U)):
            if count_non_zero_elements(U[i]) == 0:
                continue
            if count_non_zero_elements(U[i]) == 1:
                j = first_nonzero(U[i])
                x_hat[j] = z[i]/U[i][j]
                assert np.isclose(x[j], x_hat[j], atol=.05), str(x_hat[j])+' ' + str(z[i]) +' ' + str(U[i][j]) +' ' + str(x[j])

        return U, x, x_hat

    def generate_gossip_example(self, X0=None):
        """
        Generate a random example of gossip.
        If X0 is provided, it is used as the initial input vector for gossip.
        Otherwise, a random X0 is generated.
        
        Returns:
        X0 : an input vector used in gossip
        V : the vector to which attackers have access at the end of the gossip
        """
        N = self.G.number_of_nodes()
        if X0 is None:
            X0 = np.random.randint(2, size=N)
        
        X = X0
        # Gossip
        V = []
        
        # Initialize X as an array of Fractions if use_fractions is True
        if self.use_fractions:
            X = np.array([Fraction(x) for x in X0], dtype=object)  # Replace initial_X_values with your initial values for X
            X0 = X
        else:
            X = np.array(X0)  # Replace initial_X_values with your initial values for X

        for node in self.attacker_nodes:
            V.append(X[node])

        for neighbor in get_neighbors(self.G, self.attacker_nodes):
            V.append(X[neighbor])

        for i in range(1, self.n_iter):
            if self.use_fractions:
                X = matrix_multiply(self.W, X)
            else:
                X = self.W @ X
            
            for neighbor in get_neighbors(self.G, self.attacker_nodes):
                V.append(X[neighbor])

        V = np.array(V, dtype=object if self.use_fractions else None)

        
        return X0, V


    def print_reconstruction_results(self, verbose=True):
        
        #self.build_knowledge_matrix()
        if verbose:
            print("Attacker knowledge matrix")
            print(self.knowledge_matrix)
        rank = self.rank()
        print("Rank of the attacker knowledge matrix :", rank)
        indistinct = self.indistinguishable_nodes()
        print("Indistinguishable nodes :", indistinct)
        


    def visualize_results(self, X=None, save_to_pdf=False, prefix='', pos=None):
        # Perform the reconstruction and get necessary variables
        U, x, x_hat = self.reconstruction(X)

        # Print matrices and vectors
        print("Knowledge Matrix (K):", "\\begin{bmatrix}"+"\\\\\n".join("&".join(map(str,v)) for v in np.round(self.knowledge_matrix,3))+"\n\\end{bmatrix}")
        print("RREF (U):", "\\begin{bmatrix}"+"\\\\\n".join("&".join(map(str,v)) for v in np.round(U,3))+"\n\\end{bmatrix}")
        print("Input Vector (X):", np.round(x,3))
        print("Estimated Input Vector (x_hat):", x_hat)

        #print("Final check")
        print(np.round(x - x_hat, 3))
        num_nans = np.sum(np.isnan(x_hat))
        print(num_nans, num_nans/len(x))

        
        
        if pos is None:
            # Graph visualization
            pos = nx.spring_layout(self.G)  # Positioning of the nodes

        # Adjust label positions to be to the right of the nodes
        #label_pos = {node: (pos[node][0] + 0.1, pos[node][1]) for node in self.G.nodes()}

        # Function to draw and optionally save the graph
        def draw_graph(node_labels, title, filename):
            plt.figure(figsize=(4, 3))
            G.remove_edges_from(nx.selfloop_edges(G))

            node_colors = ['red' if n in self.attacker_nodes else 'xkcd:purple' for n in self.G.nodes()]
            for node in self.G.nodes():
                if node_labels[node] =='nc':
                    node_colors[node] = 'xkcd:green'
            nx.draw(self.G, pos, with_labels=True, node_color='white', edgecolors=node_colors, node_size=500, edge_color='xkcd:silver')
            #nx.draw_networkx_labels(self.G, label_pos, labels=node_labels, font_size=12)
            #plt.title(title)
            if save_to_pdf:
                plt.savefig(prefix + filename, bbox_inches='tight', pad_inches=0)
            #plt.show()

        # Figure 1: Graph with X values
        filename_1 = "original_values_graph.pdf" if prefix == '' else prefix + "_original_values_graph.pdf"
        print(self.G.nodes())
        print(len(x), x)
        draw_graph({n: f"{x[n]:.2f}" for n in self.G.nodes()}, "Graph with Original Values (X)", filename_1)

        # Figure 2: Graph with x_hat values
        filename_2 = "reconstructed_values_graph.pdf" if prefix == '' else prefix + "_reconstructed_values_graph.pdf"
        draw_graph({n: f"{x_hat[n]:.2f}" if x_hat[n] is not np.NaN else 'nc' for n in self.G.nodes()}, "Graph with Reconstructed Values (x_hat)", filename_2)



        
if __name__ == "__main__":
    # Defining a graph
    np.random.seed(0)
    """
    G = nx.Graph()
    for i in range(6):
        G.add_node(i)
    G.add_edge(0,1)
    G.add_edge(0,4)
    G.add_edge(1,2)
    G.add_edge(1,3)
    G.add_edge(3,4)
    G.add_edge(4,5)
    
    n=7
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)
    G.add_edge(0,1)
    G.add_edge(0,2)
    G.add_edge(0,3)
    G.add_edge(1,4)
    G.add_edge(4,5)
    G.add_edge(4,6)

    
    for _ in range(100):
        n= 10
        G = nx.erdos_renyi_graph(n, .3, seed=np.random.randint(1000000))
        # Defining the attacker nodes
        attacker_nodes = [0]
        n_iter = 5

        try:
            # Reconstructing the knowledge matrix
            R = Reconstruct(G, n_iter, attacker_nodes)
            R.visualize_results(np.random.random(n), save_to_pdf=True)
        except e:
            print(seed)
            raise e
    

    n = 50  # Number of nodes
    iterations = [1, 4, 8]  # Different numbers of iterations for reconstruction
    attacker_nodes = [0,1]  # Defining the attacker nodes

    np.random.seed(0)
    # Generate a geometric random graph
    pos = {i: (np.random.random(), np.random.random()) for i in range(n)}
    G = nx.random_geometric_graph(n, 0.2, pos=pos)
    """
    import matplotlib.cm
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rcParams.update({'font.size': 100})
    
    iterations = [3]
    attacker_nodes = [0, 1]
    
    #G = nx.erdos_renyi_graph(n, 0.3)
    G = nx.Graph()
    for i in range(11):
        G.add_node(i)
    G.add_edge(0,2)
    G.add_edge(1,2)
    G.add_edge(2,3)
    G.add_edge(2,4)
    G.add_edge(1,5)
    G.add_edge(1,7)
    G.add_edge(1,8)    
    G.add_edge(5,9)
    G.add_edge(9,10)
    G.add_edge(9,11)
    G.add_edge(11,6)
    G.add_edge(7,8)
    G.add_edge(7,9)
    G.add_edge(7,11)
    n = G.number_of_nodes()

    pos = nx.spring_layout(G)

    start_colors = ['red', 'red', 'xkcd:orange', 'xkcd:grey','xkcd:grey', 'xkcd:orange', 'xkcd:grey', 'xkcd:orange', 'xkcd:orange', 'xkcd:grey','xkcd:grey','xkcd:grey' ]
    
    plt.figure(figsize=(4, 3))
    nx.draw(G, pos, with_labels=True, node_color='white', edgecolors=start_colors, node_size=500, edge_color='xkcd:silver')


    plt.savefig('essai.pdf', bbox_inches='tight', pad_inches=0)

    for n_iter in iterations:
        # Reconstructing the knowledge matrix
        R = Reconstruct(G, n_iter, attacker_nodes, use_fractions=False)

        # Run visualization and save results
        prefix = f"iter_{n_iter}"
        R.visualize_results(save_to_pdf=True, prefix=prefix, pos=pos)
    