import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("src")
sys.path.append("invertinggradients")
from decentralized import *
from graphs import *
from models import *
from utils import *
from metrics import *
from consensus import *
from resnet_model import *
from tqdm import tqdm, trange
import torch
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 42

matplotlib.rcParams['ps.fonttype'] = 42
torch.set_default_dtype(torch.float64)

from PIL import Image, ImageOps

# Defining the attack parameters
setup = system_startup()

G=nx.florentine_families_graph()
G = nx.convert_node_labels_to_integers(G)
#G = nx.relabel_nodes(G, {0:5, 5:0})
N = G.number_of_nodes()
n_iter = 7
attackers = [0]
W = LaplacianGossipMatrix(G)
Wt = torch.tensor(W).float()
R = ReconstructOptim(G, n_iter, attackers)


# Inversion parameters
config = dict(signed=True,
                    boxed=True,
                    cost_fn='sim',
                    indices='def',
                    weights='equal',
                    lr=0.1,
                    optim='adam',
                    restarts=1,
                    max_iterations=500,
                    total_variation=1e-1,
                    init='randn',
                    filter='none',
                    lr_decay=True,
                    scoring_choice='loss')

# Defining the model
torch.manual_seed(0)
cnn_model = CNN(10)

model = Model(cnn_model, invert_gradient_MNIST, flatten = False, setup = setup)

# Defining and running the decentralized protocol
D = Decentralized(R, model, setup, data_name= "MNIST",  targets_only = True)


lr = 1e-5
D.run(lr);

x_hat = R.reconstruct_LS_target_only(D.sent_params, D.attackers_params)


print('Absolute errors')
print(square_loss( D.gradients[1:N], -1/lr*torch.tensor(x_hat) ))
print('Relative errors')
print(relative_square_loss(D.gradients[1:N] , -1/lr*torch.tensor(x_hat)))

save_folder = "outputs/"

outputs = []
for j in range(N-1): #
    ground_truth = D.data[j+1]
    output, stats = model.invert(x_hat[j], lr, D.labels[j+1], D.models[j+1].pytorch_model, config)
    outputs.append(output)
    test_mse = (output.detach() - ground_truth).pow(2).mean().item()
    feat_mse = (D.models[j+1](output.detach())- D.models[j+1](ground_truth)).pow(2).mean().item()

    test_psnr = psnr(output, ground_truth, factor=1.0, batched = True)

    print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
          f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |");


    plot_tensors(output, True, save_folder = save_folder, suffix = str(j), data ='mnist')


target_inputs = torch.cat(D.data)[1:] 


plot_tensors(torch.cat(outputs), True, save_folder = save_folder, suffix = '_outputs', data ='mnist', multiline = False)
plot_tensors(target_inputs, save= True, save_folder =save_folder, data = 'mnist', suffix = '_inputs')



exit()

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

