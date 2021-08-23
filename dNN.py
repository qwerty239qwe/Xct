import numpy as np
import scipy
from stiefel import *
import itertools
import torch
import torch.nn as nn
#import torch.nn.functional as F
cuda = torch.device('cuda') 
        
class Net(nn.Module):
    """Define the neural network"""
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

def train_and_project(counts_np, w, dim = 2, steps = 1000, lr = 0.001, layers = None):
    '''manifold alignment by neural network
        counts_np: list of counts in numpy array, gene by cell;
        w: correspondence'''
    if not all(isinstance(x_np, np.ndarray) for x_np in counts_np):
        raise TypeError('input a list of counts in numpy arrays with genes by cells')
    if not sum([x_np.shape[0] for x_np in counts_np]) == w.shape[0]:
        raise ValueError('input sequence of counts consistent with correspondence')

    n = len(counts_np)
    d = {}
    if layers is None:
        a = 4
        for i in range(1, n+1):
            d[f'n_{i}'] = scipy.stats.gmean([counts_np[i-1].shape[1], dim]).astype(int)
            d[f'layers_{i}'] = [a*d[f'n_{i}'], d[f'n_{i}'], dim]
    elif len(layers) != 3:
        raise ValueError('input node numbers of three hidden layers')
    else:
        for i in range(1, n+1):
            d[f'layers_{i}'] = layers

    losses = [] 
    torch.manual_seed(0)

    for i in range(1, n+1):
        d[f'model_{i}'] = Net(counts_np[i-1].shape[1], d[f'layers_{i}'][0], d[f'layers_{i}'][1], d[f'layers_{i}'][2])
        print(d[f'model_{i}'])
        d[f'x_{i}'] = torch.from_numpy(counts_np[i-1].astype(np.float32))

    L_np = scipy.sparse.csgraph.laplacian(w, normed = False) 
    L = torch.from_numpy(L_np.astype(np.float32))
    
    params = [d[f'model_{i}'].parameters() for i in range(1, n+1)]
    optimizer = torch.optim.Adam(itertools.chain(*params), lr = lr)
    
    for t in range(steps):
        # Forward pass: Compute predicted y by passing x to the model
        y_preds = []
        for i in range(1, n+1):
            y_preds.append(d[f'model_{i}'](d[f'x_{i}']))

        outputs = torch.cat(y_preds[:], 0) #vertical concat
        #print('outputs', outputs.shape)
        
        # Project the output onto Stiefel Manifold
        u, s, v = torch.svd(outputs, some=True)
        proj_outputs = u@v.t()
        
        # Compute and print loss
        loss = torch.trace(proj_outputs.t()@L@proj_outputs)
        
        if t == 0 or t%100 == 99:
            print(t+1, loss.item())
            losses.append(loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        proj_outputs.retain_grad() #et

        optimizer.zero_grad()
        loss.backward(retain_graph = True)

        # Project the (Euclidean) gradient onto the tangent space of Stiefel Manifold (to get Rimannian gradient)
        rgrad = proj_stiefel(proj_outputs, proj_outputs.grad) #pt

        optimizer.zero_grad()
        # Backpropogate the Rimannian gradient w.r.t proj_outputs
        proj_outputs.backward(rgrad) #backprop(pt)

        optimizer.step()

    proj_outputs_np = proj_outputs.detach().numpy()
    
    return proj_outputs_np, losses
