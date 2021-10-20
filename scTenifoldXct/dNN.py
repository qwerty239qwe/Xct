import numpy as np
import scipy
from scTenifoldXct.stiefel import *
import itertools
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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


class ManifoldAlignmentNet:
    def __init__(self,
                 counts_arr,
                 w,
                 n_dim,
                 layers,
                 verbose):
        self.n_models, self.counts_arr, self.w = self._check_data(counts_arr, w=w)
        self.model_dic = self.create_models(layers, n_dim, verbose)

    def _check_data(self, counts_arr, w):
        n_models = len(counts_arr)
        if not all(isinstance(x_np, np.ndarray) for x_np in counts_arr):
            raise TypeError('input a list of counts in numpy arrays with genes by cells')
        if not sum([x_np.shape[0] for x_np in counts_arr]) == w.shape[0]:
            raise ValueError('input sequence of counts consistent with correspondence')
        return n_models, counts_arr, w

    def create_models(self, layers, n_dim, verbose=True):
        layer_dic = {}
        if layers is None:
            a = 4
            for i in range(1, self.n_models + 1):
                n_h = scipy.stats.gmean([self.counts_arr[i - 1].shape[1], n_dim]).astype(int)
                layer_dic[i] = [a * n_h, n_h, n_dim]
        elif len(layers) != 3:
            raise ValueError('input node numbers of three hidden layers')
        else:
            for i in range(1, self.n_models + 1):
                layer_dic[i] = layers

        model_dic = {}
        torch.manual_seed(0)
        for i in range(1, self.n_models + 1):
            model_dic[f'model_{i}'] = Net(self.counts_arr[i - 1].shape[1], *layer_dic[i])
            if verbose:
                print(model_dic[f'model_{i}'])
            self.counts_arr[i - 1] = torch.from_numpy(self.counts_arr[i - 1].astype(np.float32))

        return model_dic

    def save_model_states(self, file_dir):
        for i in range(1, self.n_models + 1):
            torch.save(self.model_dic[f'model_{i}'].state_dict(), f"{file_dir}/model_{i}")

    def load_model_states(self, file_dir):
        for i in range(1, self.n_models + 1):
            self.model_dic[f'model_{i}'].load_state_dict(torch.load(f"{file_dir}/model_{i}"))
            self.model_dic[f'model_{i}'].eval()

    def train(self,
              n_steps = 1000,
              lr = 0.01,
              verbose = True,
              **optim_kwargs):

        self.losses = []
        L_np = scipy.sparse.csgraph.laplacian(self.w, normed=False)
        L = torch.from_numpy(L_np.astype(np.float32))
        params = [self.model_dic[f'model_{i}'].parameters() for i in range(1, self.n_models + 1)]
        optimizer = torch.optim.Adam(itertools.chain(*params), lr=lr, **optim_kwargs)

        for i in range(1, self.n_models + 1):
            self.model_dic[f'model_{i}'].train()

        for t in range(n_steps):
            # Forward pass: Compute predicted y by passing x to the model
            y_preds = []
            for i in range(1, self.n_models + 1):
                y_preds.append(self.model_dic[f'model_{i}'](self.counts_arr[i - 1]))

            outputs = torch.cat(y_preds[:], 0)  # vertical concat
            # print('outputs', outputs.shape)

            # Project the output onto Stiefel Manifold
            u, _, v = torch.svd(outputs, some=True)
            proj_outputs = u @ v.t()

            # Compute loss
            loss = torch.trace(proj_outputs.t() @ L @ proj_outputs)

            if t == 0 or t % 10 == 9:
                if verbose:
                    print(t + 1, loss.item())
                self.losses.append(loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            proj_outputs.retain_grad()  # et

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # Project the (Euclidean) gradient onto the tangent space of Stiefel Manifold (to get Rimannian gradient)
            rgrad = proj_stiefel(proj_outputs, proj_outputs.grad)  # pt

            optimizer.zero_grad()

            # Backpropogate the Rimannian gradient w.r.t proj_outputs
            proj_outputs.backward(rgrad)  # backprop(pt)
            optimizer.step()

        self.proj_outputs_np = proj_outputs.detach().numpy()
        return self.proj_outputs_np, self.losses

    def plot_losses(self, file_name=None):
        '''plot loss every 100 steps'''
        plt.figure(figsize=(6, 5), dpi=80)
        plt.plot(np.arange(len(self.losses)) * 100, self.losses)
        if file_name is not None:
            plt.savefig(file_name, dpi=80)
        plt.show()


def train_and_project(counts_np: np.ndarray,
                      w,
                      dim: int = 3,
                      steps: int = 1000,
                      lr: float = 0.01,
                      layers = None,
                      verbose: bool = True):
    """
    manifold alignment by neural network

    Args:
        counts_np (np.ndarray): list of counts in numpy array, gene by cell
        w (np.ndarray): correspondence
        dim (int): dimension of latent space projected to
        steps (int): number of training steps
        lr (float): learning rate
        layers: node numbers of three layers
        verbose (bool):

    Returns:

    """

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
        d[f'model_{i}'] = Net(counts_np[i-1].shape[1], *d[f'layers_{i}'])
        if verbose:
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
        u, _, v = torch.svd(outputs, some=True)
        proj_outputs = u@v.t()
        
        # Compute loss
        loss = torch.trace(proj_outputs.t()@L@proj_outputs)
        
        if t == 0 or t % 10 == 9:
            if verbose:
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
