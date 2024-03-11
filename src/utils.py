import numpy as np
import hashlib
import networkx as nx
import collections
import torch
from matplotlib import pyplot as plt
try:
    from src.constants import *
except:
    from constants import *
import socket


from fractions import Fraction
tol = 1e-7

def to_frac(v):
    return Fraction(v).limit_denominator(int(1/tol))

def uniformGossipMatrix(G):
    """
    Generates the uniform gossip matrix, converges to a weighted average of the initial values
    with weights proportional to the degree of the nodes
    G : networkx graph
    returns the uniform gossip matrix
    """
    W = nx.adjacency_matrix(G).toarray() + np.eye(G.number_of_nodes())
    for i in range(len(W)):
        W[i] /= sum(W[i])
    return W

def LaplacianGossipMatrix(G):
    max_degree = max([G.degree(node) for node in G.nodes()])
    W = np.eye(G.number_of_nodes()) - 1/max_degree * nx.laplacian_matrix(G).toarray()
    return W

def HamiltonGossipMatrix(graph, use_fractions=False):
    "Alternative Gossip matrix taken from Muffliato"
    degree = nx.degree(graph)
    for u in nx.nodes(graph):
            graph.add_edge(u, u)
    if use_fractions:
        # Calculate and set the weights
        for u in nx.nodes(graph):
            out_w = Fraction(0)
            for v in nx.neighbors(graph, u):
                if v != u:
                    w = Fraction(1, max(degree[u], degree[v]) - 1)
                    out_w += w
                    graph[u][v]['weight'] = w
            graph[u][u]['weight'] = Fraction(1) - out_w

        # Convert to a numpy array
        # Since numpy does not support Fraction, we'll use object type
        W = nx.to_numpy_array(graph, dtype=object)
    else:
        for u in nx.nodes(graph):
            out_w = 0
            for v in nx.neighbors(graph, u):
                if v != u:
                    w = 1 / (max(degree[u],degree[v]) - 1)
                    out_w += w
                    graph[u][v]['weight'] = w
            graph[u][u]['weight'] = 1 - out_w
        W = nx.to_numpy_array(graph)
    return W

def genHash(s):
    s = "_".join(s)
    hash_object = hashlib.md5(s.encode())
    hashing = hash_object.hexdigest()
    return hashing

def WL(graph, iterations, special_initialize = False, attackers = None):
    """
    graph : networkx graph
    iterations : number of iterations of the WL algorithm
    special_initialize : boolean, if True, the attackers are initialized with a special value
    attackers : list of nodes, if special_initialize is True, the attackers and their 
    neighbors are initialized with distinct values
    returns the features dictionary and a boolean indicating if the algorithm terminated
    """
    features = {node:"0" for node in graph.nodes()}
    if special_initialize:
        c = 0
        if attackers is not None:
            for node in attackers:
                c+=1
                features[node] = str(c)
            neighbors = get_neighbors(graph, attackers)
            for node in neighbors:
                c+=1
                features[node] = str(c)
    terminate = False
    for _ in range(iterations):
        new_feats={}
        for node in graph.nodes():
            neighbors = graph.neighbors(node)
            feats = [features[node]]+ sorted([features[neighbor] for neighbor in neighbors])
            new_feats[node] = genHash(feats)
        if set(integerHash(features).values()) == set(integerHash(new_feats).values()):
            terminate = True
            break
        features = new_feats
    return features, terminate

def integerHash(features):
    """
    Return a simpler hash using integers 
    each hash receives an integer value from 0 to n_unique_hash-1
    """
    new_dict = {}
    S = list(set(features.values()))
    for k, h in features.items():
        new_dict[k] = S.index(h)
    return new_dict

def get_same_hash_nodes(integer_hash):
    """
    integer_hash : a dictionary of the form {node: hash}
    returns a dictionary of the form {hash: [list of nodes with that hash]}
    """
    unique_hashes = set(integer_hash.values())
    res = []
    for h in unique_hashes:
        res.append([k for k, v in integer_hash.items() if v == h])
    return res

def indistinguishable_nodes_from_hash(f):
    hash_groups = get_same_hash_nodes(integerHash(f))
    res = set()
    for group in hash_groups:
        if len(group)>1:
            res.add(tuple(sorted(group)))
    return res

def get_neighbors(G, attackers):
    """
    G : networkx graph
    attackers : list of the nodes considered as attackers
    returns : non repetetive list of the neighbors of the attackers
    """
    return sorted(set(n for attacker in attackers for n in G.neighbors(attacker)))

def get_non_attackers_neighbors(G, attackers):
    """
    G : networkx graph
    attackers : list of the nodes considered as attackers
    returns : non repetetive list of the neighbors of the attackers
    """
    return sorted(set(n for attacker in attackers for n in G.neighbors(attacker)).difference(set(attackers)))
    #return list(set(sum([list(G.neighbors(attacker)) for attacker in attackers], [])).difference(set(attackers)))

def indistinguiable_groups(A):
    """
    A helper function to create inditinguishable groups of nodes
    A : A list of pairs (u, v) of indistinguishable nodes
    returns : The list of the groups of indistinguishable nodes
    """
    
    def get_parent(P, v):
        # P a list containing the parent of each node
        if P[v]==v:
            return v
        else:
            return get_parent(P, P[v])

    # Considering A are the edges of a graph, construct a list of connected components 
    try:
        V = list(set(sum(A, ()))) # TODO : raise exception in some cases
    except Exception as e:
        return []
    P = {v : v for v in V}
    for a in A:
        u,v = a
        P[get_parent(P, u)] = get_parent(P, v)
    fA = collections.defaultdict(list)
    for v in V:
        fA[get_parent(P, v)].append(v)
    return list(fA.values())
        
def count_non_zero_elements(arr):
    """ 
    return the number of non zero element in array arr
    """
    return arr.size - np.count_nonzero(np.isclose(arr, 0))

def first_nonzero(arr):
    barr = np.isclose(arr, 0)
    assert barr.ndim ==1, "this should be a 1D array"
    for i in range(len(barr)):
        if not(barr[i]):
            break
    return i

def nonzero_indices(arr):
    """
    returns the indices of nonzero elements 
    """
    barr = np.isclose(arr, 0)
    idx = []
    for i in range(len(barr)):
        if not(barr[i]):
            idx.append(i)
    return idx

def GLS(X, y, cov):
    """
    Returns the generalized least squares estimator b, such as 
    Xb = y + e
    e being a noise of covariance matrix cov
    """
    X_n, X_m = X.shape
    y_m = len(y)
    s_n = len(cov)
    assert s_n == X_n, "Dimension mismatch"
    try:
        inv_cov = np.linalg.inv(cov)
    except Exception as e:
        print("WARNING : The covariance matrix is not invertible, using pseudo inverse instead")
        inv_cov = np.linalg.pinv(cov)
    return np.linalg.inv(X.T@inv_cov@X)@ X.T@inv_cov@y


dm = torch.as_tensor(cifar10_mean)[:, None, None]
ds = torch.as_tensor(cifar10_std)[:, None, None]

def plot_tensors(tensor, save=False, save_folder=None, suffix="", title=None, data="cifar10", multiline=False, row_size=10):
    if data == "cifar10":
        dm = torch.as_tensor(cifar10_mean)[:, None, None]
        ds = torch.as_tensor(cifar10_std)[:, None, None]
    elif data == "mnist":
        dm = torch.as_tensor((0.1307,))[:, None, None]
        ds = torch.as_tensor((0.3081,))[:, None, None]

    if save and save_folder is None:
        raise ValueError("Please specify a save folder")
    plt.clf()
    tensor = tensor.clone().detach().cpu()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        if multiline:
            n_rows = (tensor.shape[0] + 9) // row_size
            fig, axes = plt.subplots(n_rows, row_size, figsize=(10, n_rows))
            #fig.subplots_adjust(hspace=0.1, wspace=0.1)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            for i, im in enumerate(tensor):
                row = i // row_size
                col = i % row_size
                axes[row, col].axis("off")
                axes[row, col].imshow(im.permute(1, 2, 0).cpu())
            # axis("off") for the rest 
            for i in range(tensor.shape[0], n_rows*row_size):
                row = i // row_size
                col = i % row_size
                axes[row, col].axis("off")
        else:
            fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0] * 12))
            for i, im in enumerate(tensor):
                axes[i].axis("off")
                axes[i].imshow(im.permute(1, 2, 0).cpu())
    if save:
        plt.savefig(save_folder + '/image{}.png'.format(suffix))
    

def count_model_params(model):
    n_params = 0
    for param in model.parameters():
        n_params+=1
    return n_params

def system_startup(args=None, defs=None):
    """
    Code copied from invertinggradient 
    Print useful system information.
    """
    # Choose GPU device and print status information:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    setup = dict(device=device, dtype=torch.double) 
    print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.')
    if args is not None:
        print(args)
    if defs is not None:
        print(repr(defs))
    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')
    return setup

def matrix_multiply(A, B):

    # Check if A or B is a vector (1D array) and reshape it to matrix form if necessary
    if A.ndim == 1:
        A = A.reshape((1, A.shape[0]))
    if B.ndim == 1:
        B = B.reshape((B.shape[0], 1))
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must equal number of rows in B")

    # Initialize the result matrix with zeros (as Fractions)
    result = np.array([[Fraction(0) for _ in range(cols_B)] for _ in range(rows_A)], dtype=object)
    print('result', result, A, B)
    # Perform the multiplication
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i, j] += A[i, k] * B[k, j]
    print('result 2', result)
    return result
