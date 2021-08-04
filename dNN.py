import numpy as np
import scipy
from neighborhood import neighbor_graph, laplacian
#from correspondence import Correspondence
from stiefel import *
import itertools
import torch
import torch.nn as nn
#import torch.nn.functional as F
cuda = torch.device('cuda') 


"""Defines the neural network"""

class Net(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        h1_sigmoid = self.linear1(x).sigmoid()
        h2_sigmoid = self.linear2(h1_sigmoid).sigmoid()
        y_pred = self.linear3(h2_sigmoid)
        return y_pred

def train_and_project(x1_np, x2_np, w, d = 2, n=3000, lr=0.01, layers=None):

    if not ((isinstance(x1_np, np.ndarray)) & (isinstance(x2_np, np.ndarray))):
        raise TypeError('Input numpy arrays with genes by cells')

    if layers is None:
        a = 4
        n1 = scipy.stats.gmean([x1_np.shape[1], d]).astype(int)
        n2 = scipy.stats.gmean([x2_np.shape[1], d]).astype(int)
        layers1 = [a*n1, n1, d]
        layers2 = [a*n2, n2, d]
    elif len(layers) != 3:
        raise ValueError('Input node numbers of three hidden layers')
    else:
        layer1 = layer2 = layers

    losses = [] 
    torch.manual_seed(0)

    model_1 = Net(x1_np.shape[1], layers1[0], layers1[1], layers1[2])
    model_2 = Net(x2_np.shape[1], layers2[0], layers2[1], layers2[2])
    print(model_1)
    print(model_2)

    x1 = torch.from_numpy(x1_np.astype(np.float32))
    x2 = torch.from_numpy(x2_np.astype(np.float32))

    L_np = laplacian(w, normed=False)  #csgraph.laplacian(w, normed=False)
    L = torch.from_numpy(L_np.astype(np.float32))
    
    #params = list(model_1.parameters()) + list(model_2.parameters())
    params = [model_1.parameters(), model_2.parameters()]

    optimizer = torch.optim.Adam(itertools.chain(*params), lr = lr)
    
    for t in range(n):
        # Forward pass: Compute predicted y by passing x to the model
        y1_pred = model_1(x1)
        y2_pred = model_2(x2)
        #print('y1, y2', y1_pred.shape, y2_pred.shape)

        outputs = torch.cat((y1_pred, y2_pred), 0)
        #print('outputs', outputs.shape)
        
        # Project the output onto Stiefel Manifold
        u, s, v = torch.svd(outputs, some=True)
        proj_outputs = u@v.t()
        
        # Compute and print loss
        loss = torch.trace(proj_outputs.t()@L@proj_outputs)
        
        if t == 0 or t%100 == 99:
            print(t, loss.item())
            losses.append(loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        proj_outputs.retain_grad() #et

        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        # Project the (Euclidean) gradient onto the tangent space of Stiefel Manifold (to get Rimannian gradient)
        rgrad = proj_stiefel(proj_outputs, proj_outputs.grad) #pt

        optimizer.zero_grad()
        # Backpropogate the Rimannian gradient w.r.t proj_outputs
        proj_outputs.backward(rgrad) #backprop(pt)

        optimizer.step()

    proj_outputs_np = proj_outputs.detach().numpy()
    
    return proj_outputs_np, losses
  
  
  
  
