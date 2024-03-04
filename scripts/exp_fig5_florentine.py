import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("../src")

from decentralized import *
from graphs import *
from models import *
from utils import *
from metrics import *
from consensus import *
from tqdm import tqdm, trange

import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42

matplotlib.rcParams['ps.fonttype'] = 42


from PIL import Image, ImageOps

# Defining the attack parameters
setup = system_startup()

G=nx.florentine_families_graph()
G = nx.convert_node_labels_to_integers(G)
G = nx.relabel_nodes(G, {0:5, 5:0})
N = G.number_of_nodes()
n_iter = 7
attackers = [0]
W = LaplacianGossipMatrix(G)
Wt = torch.tensor(W).float()
R = ReconstructOptim(G, n_iter, attackers)

np.linalg.matrix_rank(R.target_knowledge_matrix)==N-1

# Defining the model
torch.manual_seed(0)
model = Model(FC_Model(3072, 10), invert_gradient_FC_Model, setup)

# Defining and running the decentralized protocol
D = Decentralized(R, model, setup, targets_only = True)

lr = 1e-5
D.run(lr);

x_hat = R.reconstruct_LS_target_only(D.sent_params, D.attackers_params)

outputs = []
for j in range(len(x_hat)):
    outputs.append(model.invert( x_hat[j], lr))

target_inputs = torch.cat(D.data)[1:] 
plot_tensors(target_inputs, title = 'inputs')

plot_tensors(torch.cat(outputs), title = 'outputs')




#### For the drawing 

np.random.seed(0)
pos = nx.spring_layout(G)
fig, ax = plt.subplots(1,2, figsize = (13,5))
nx.draw_networkx_edges(
    G,
    pos=pos,
    ax=ax[0],
    arrows=True,
    arrowstyle="-",
    min_source_margin=15,
    min_target_margin=15,
)
tr_figure = ax[0].transData.transform
tr_axes = fig.transFigure.inverted().transform
icon_size = (ax[0].get_xlim()[1] - ax[0].get_xlim()[0]) * 0.030
icon_center = icon_size / 2.0
target_inputs = torch.cat(D.data) 
for n in range(N):
    xf, yf = tr_figure(pos[n])
    xa, ya = tr_axes((xf, yf))
    # get overlapped axes and plot icon
    a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
    #tensor_np = outputs[n-1].mul_(ds).add_(dm).clamp_(0, 1)
    tensor_np = torch.clone(target_inputs[n])
    tensor_np = tensor_np.mul_(ds).add_(dm).clamp_(0, 1)
    tensor_np = tensor_np.squeeze().numpy()  # Assuming you're using PyTorch
    
    # Step 2: Reshape the NumPy array to the appropriate shape for an image
    # Reshape to (3, 32, 32) from (1, 3, 32, 32)
    tensor_np = tensor_np.transpose((1, 2, 0))
    
    # Step 3: Convert the NumPy array to a PIL image
    pil_image = Image.fromarray((tensor_np * 255).astype(np.uint8))
    if n==0:
        pil_image = ImageOps.expand(pil_image, border=4, fill='blue')
    a.imshow(pil_image)
    a.axis("off")

ax[0].set_title("Inputs")

# OUTPUTS

np.random.seed(0)
pos = nx.spring_layout(G)

nx.draw_networkx_edges(
    G,
    pos=pos,
    ax=ax[1],
    arrows=True,
    arrowstyle="-",
    min_source_margin=15,
    min_target_margin=15,
)
tr_figure = ax[1].transData.transform
tr_axes = fig.transFigure.inverted().transform
icon_size = (ax[1].get_xlim()[1] - ax[1].get_xlim()[0]) * 0.030
icon_center = icon_size / 2.0
# Attacker data 
xf, yf = tr_figure(pos[0])
xa, ya = tr_axes((xf, yf))
# get overlapped axes and plot icon
a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
#tensor_np = outputs[n-1].mul_(ds).add_(dm).clamp_(0, 1)
tensor_np = torch.clone(target_inputs[0])
tensor_np = tensor_np.mul_(ds).add_(dm).clamp_(0, 1)
tensor_np = tensor_np.squeeze().numpy()  # Assuming you're using PyTorch

# Step 2: Reshape the NumPy array to the appropriate shape for an image
# Reshape to (3, 32, 32) from (1, 3, 32, 32)
tensor_np = tensor_np.transpose((1, 2, 0))

# Step 3: Convert the NumPy array to a PIL image
pil_image = Image.fromarray((tensor_np * 255).astype(np.uint8))
pil_image = ImageOps.expand(pil_image, border=4, fill='blue')
a.imshow(pil_image)
a.axis("off")
for n in range(N-1):
    xf, yf = tr_figure(pos[n+1])
    xa, ya = tr_axes((xf, yf))
    # get overlapped axes and plot iconL
    a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])
    tensor_np = torch.clone(outputs[n])
    tensor_np = tensor_np.mul_(ds).add_(dm).clamp_(0, 1)
    #tensor_np = target_inputs[n].mul_(ds).add_(dm).clamp_(0, 1)
    tensor_np = tensor_np.squeeze().numpy()  # Assuming you're using PyTorch
    
    # Step 2: Reshape the NumPy array to the appropriate shape for an image
    # Reshape to (3, 32, 32) from (1, 3, 32, 32)
    tensor_np = tensor_np.transpose((1, 2, 0))
    
    # Step 3: Convert the NumPy array to a PIL image
    pil_image = Image.fromarray((tensor_np * 255).astype(np.uint8))
    if reconstructed[n]:
        pil_image = ImageOps.expand(pil_image, border=4, fill='green')
    else:
        pil_image = ImageOps.expand(pil_image, border=4, fill='red')
    a.imshow(pil_image)
    a.axis("off")
ax[1].set_title("Outputs")
plt.savefig("../experiments/gd/florentine/inputs_outputs_on_graph.pdf")

extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig("../experiments/gd/florentine/inputs_outputs_on_graph_0.pdf", bbox_inches=extent)
extent = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig("../experiments/gd/florentine/inputs_outputs_on_graph_1.pdf", bbox_inches=extent)

plt.show()

