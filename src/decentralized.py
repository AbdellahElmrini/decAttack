import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import copy
import torchvision
from torchvision import transforms

from optim import ReconstructOptim
from utils import *
from constants import *
from graphs import *
from models import *
from data import *
import torch
torch.set_default_dtype(torch.float64)

class Decentralized():
    def __init__(self, R, model, setup, data_name="Cifar10", seed = 999, targets_only = False, data_root = "images"):
        """
        Class to run the decentralized algorithm and get the parameters received by the attackers
        R : ReconstructOptim object containing the graph, the attackers and the reconstruction methods
        model : Model object containing the model architecture and the inversion function
        setup : dictionary containing the device and the dtype
        data_name : name of the dataset used (only Cifar10 for the moment)
        targets_only : if True, the attackers parameters are used in the computation and only the parameters of the targets are returned
        """
         # GPU if available, otherwise CPU 
        self.R = R # Reconstruction Class containing parameters of the graph and the attackers
        self.N = self.R.G.number_of_nodes()
        self.setup = setup 
        self.attackers = self.R.attackers  # Attacker nodes
        self.model = model # model architecture used for each node
        self.data_name = data_name  # Dataset name
        self.seed = seed 
        self.data_root = data_root
        self.n_params = model.n_params 
        self.W = R.W # Gossip matrix
        self.Wt = torch.tensor(self.W, **self.setup) # Gossip matrix as a torch tensor
        self.targets_only = targets_only
        self.build() # Building the models and the dataset
        
    def build(self):
        """
        Function building the models and the data, and storing them as attributes
        """
        # Building the models 
        models = []
        np.random.seed(self.seed)
        idxx = np.random.choice(10000,self.N)
        for i, idx in enumerate(idxx):
            models.append(copy.deepcopy(self.model))
        self.models = models
        self.idxx = idxx
        # Fetching the dataset
        try:
            trainset = datasets[self.data_name](self.setup, self.data_root)
            print("Data Ok")
        except:
            raise Exception(f"{self.data_name} is not supported")
        # Building the dataset for each node
        data = []
        labels = []
        for i, idx in enumerate(self.idxx):
            img, label = trainset[idx]
            labels.append(torch.as_tensor((label,), device=self.setup['device']))
            ground_truth = img.to(**self.setup).unsqueeze(0)

            data.append(ground_truth)
        self.data = data
        self.labels = labels
        
        
    def get_model_params(self):
        """
        get the current parameters of all the models
        """
        params = [[] for j in range(self.n_params)]
        for i in range(self.N):
            for j, param in enumerate(self.models[i].parameters()):
                params[j].append(param)
        for j in range(len(params)):
            params[j] = torch.stack(params[j])
        return params
    

    def pretrain(self, train_steps, lr_train):
        """
        Pretraining the models for train_steps steps with learning rate lr_train
        return the norm of the gradients at each step
        """
        gr_norms = np.zeros((train_steps,self.N))
        for it in range(train_steps):
            for i in range(self.N):
                ground_truth = self.data[i]#.flatten().unsqueeze(0)
                label = self.labels[i]

                input_gradient = self.models[i].step(lr_train, ground_truth, label )
                gradient_norm = sum([torch.norm(gr) for gr in input_gradient])
                #gr_norms.append(gradient_norm.cpu().item())
                gr_norms[it,i] = gradient_norm.cpu().item()

            # Adding sent parameters            
            params = self.get_model_params()
            

            # Aggregating 
            for j in range(self.n_params):
                params[j] = torch.einsum('mn,n...->m...', self.Wt, params[j])
            
            for i in range(self.N):
                self.models[i].update([params[j][i] for j in range(self.n_params)])


        return gr_norms
     
    def run(self, lr ):
        
        gradients = []
        sent_params = []
        params = self.get_model_params()
        params0 = self.get_model_params()
        attackers_params = []
        gr_norms = np.zeros((self.R.n_iter, self.N))
        
        for it in range(self.R.n_iter): 
            for i in range(self.N):
                
                ground_truth = self.data[i]
                label = self.labels[i]
                self.models[i].update([params[j][i] for j in range(self.n_params)])
            
                input_gradient = self.models[i].step(lr, ground_truth, label )

                # Adding gradients
                gradients.append(input_gradient)
                gradient_norm = sum([torch.norm(gr) for gr in input_gradient])
                gr_norms[it,i] = gradient_norm.cpu().item()
            # Adding sent parameters            
            params = self.get_model_params()

            if it == 0:
                for node in self.R.attackers:
                    sent_params.append(torch.cat([(params[j][node] - params0[j][node]).flatten().detach() for j in range(self.n_params)]).cpu())
                        
            for node in self.R.attackers:
                attackers_params.append(torch.cat([(params[j][node] - params0[j][node]).flatten().detach() for j in range(self.n_params)]).cpu())

            for neighbor in get_non_attackers_neighbors(self.R.G, self.attackers):
                sent_params.append(torch.cat([(params[j][neighbor] - params0[j][neighbor]).flatten().detach() for j in range(self.n_params)]).cpu())
                    
                    
            # Aggregating 
            for j in range(self.n_params):
                params[j] = torch.einsum('mn,n...->m...', self.Wt, params[j])
            for j in range(self.n_params):
                params0[j] = torch.einsum('mn,n...->m...', self.Wt, params0[j])
        self.sent_params = torch.stack(sent_params)
        self.attackers_params = torch.stack(attackers_params)
        self.gradients =  [torch.cat([gradients[i][j].flatten().cpu() for j in range(len(gradients[0]))]) for i in range(self.N)]
        self.gr_norms = gr_norms
       
        if self.targets_only:
            return sent_params, attackers_params
        else:
            return sent_params

    
    def consensus_distance(self):

        params = self.get_model_params()
        num_clients = self.N
        total_distance = 0
        
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                diff = [(params[k][i] - params[k][j] ).flatten().detach() for k in range(len(params))]
                distance = np.linalg.norm(torch.cat(diff).cpu().numpy())
                total_distance += distance
        
        average_distance = total_distance / (num_clients * (num_clients - 1) / 2)
        return average_distance
        
    def plot_inputs(self,  save = False, save_folder = None, suffix = 0, title = None, data = 'cifar10', multiline = False):
        n_attackers = len(self.attackers)
        plot_tensors(torch.cat(self.data[n_attackers:]),  save = save, save_folder = save_folder, suffix = suffix, title= title, data = data, multiline = multiline)