import torch
import gpytorch
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch import kernels
from gpytorch.kernels import MultitaskKernel, ScaleKernel, Kernel
from typing import Optional
from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
import tqdm
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

class MTGPRegressor(gpytorch.models.ExactGP):
    def __init__(self, X, y, kernel, mean=ConstantMean(), likelihood=None):
        if likelihood is None:
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
        
        super().__init__(X, y, likelihood)
        self.mean = gpytorch.means.MultitaskMean(mean, num_tasks=2)
        self.covar_module = MultitaskKernel(kernel, num_tasks=2, rank=1)
        self.likelihood = likelihood
    
    def forward(self, x):
        mean_x = self.mean(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    def predict(self, X):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self(X))
            mean = predictions.mean
            self.lower, self.upper = predictions.confidence_region()
            self.sigma1_l = predictions.mean[:, 0] - predictions.stddev[:, 0]
            self.sigma1_u = predictions.mean[:, 0] + predictions.stddev[:, 0]
            self.sigma2_l = predictions.mean[:, 1] - predictions.stddev[:, 1]
            self.sigma2_u = predictions.mean[:, 1] + predictions.stddev[:, 1]
        return predictions, mean
    
    def plot_preds(self, mean, date_train, date_test, y_train, y_test, 
                   label1 = 'training data', label2 = 'predictions', figsize = (10, 5)):
        plot_df = pd.DataFrame({'mean_lat': mean[:,0],
                        'mean_long': mean[:,1],
                        'lower_lat': self.lower[:,0],
                        'lower_long': self.lower[:,1],
                        'upper_lat': self.upper[:,0],
                        'upper_long': self.upper[:,1],
                        'sigma1_l': self.sigma1_l,
                        'sigma1_u': self.sigma1_u,
                        'sigma2_l': self.sigma2_l,
                        'sigma2_u': self.sigma2_u,
                       'datetime': date_test},
                       columns=['mean_lat', 'mean_long', 'lower_lat', 
                                'lower_long', 'upper_lat', 'upper_long', 
                                'sigma1_l', 'sigma1_u', 'sigma2_l', 'sigma2_u', 'datetime'])

        # Initialize plots
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        f, (y1_ax, y2_ax) = plt.subplots(2, 1, constrained_layout = True)

        y1_ax.plot(date_train, y_train[:,0].numpy(), '.', c = 'blue', label = label1)
        y1_ax.plot(date_test, plot_df['mean_lat'], '.', c='red', label = label2)
        y1_ax.scatter(date_test, y_test[:,0].numpy(), marker='.', c='blue')
        y1_ax.fill_between(date_test, 0, 1, where=date_test, 
                           color='pink', alpha=0.5, label = 'Testing period', 
                           transform=y1_ax.get_xaxis_transform())
        y1_ax.set_title('Latitude')
        y1_ax.set_xticks([])

        y2_ax.plot(date_train, y_train[:,1].numpy(), '.', c = 'blue', label = label1)
        y2_ax.plot(date_test, plot_df['mean_long'], '.', c='red', label = label2)
        y2_ax.scatter(date_test, y_test[:,1].numpy(), marker='.', c='blue')
        y2_ax.fill_between(date_test, 0, 1, where=date_test, 
                           color='pink', alpha=0.5, label = 'Testing period', 
                           transform=y2_ax.get_xaxis_transform())
        y2_ax.set_title('Longitude')
        #y2_ax.set_xticks([])

        plt.legend()
        plt.xticks(rotation = 45)
        plt.xlabel('Date', fontsize=10)

        plt.show()

class WhiteNoiseKernel(kernels.Kernel):
    def __init__(self, noise=1):
        super().__init__()
        self.noise = noise
    
    def forward(self, x1, x2, **params):
        if self.training and torch.equal(x1, x2):
            return DiagLazyTensor(torch.ones(x1.shape[0]).to(x1) * self.noise)
        elif x1.size(-2) == x2.size(-2) and torch.equal(x1, x2):
            return DiagLazyTensor(torch.ones(x1.shape[0]).to(x1) * self.noise)
        else:
            return torch.zeros(x1.shape[0], x2.shape[0]).to(x1)
        


class DiffusionKernel(Kernel):
    r"""
        Computes diffusion kernel over discrete spaces with arbitrary number of categories. 
        Input type: n dimensional discrete input with c_i possible categories/choices for each dimension i 
        As an example, binary {0,1} combinatorial space corresponds to c_i = 2 for each dimension i
        References:
        - https://www.ml.cmu.edu/research/dap-papers/kondor-diffusion-kernels.pdf (Section 4.4)
        - https://arxiv.org/abs/1902.00448
        - https://arxiv.org/abs/2012.07762
        
        Args:
        :attr:`categories`(tensor, list):
            array with number of possible categories in each dimension            
    """
    has_lengthscale = True
    def __init__(self, categories, **kwargs):
        if categories is None:
            raise RunTimeError("Can't create a diffusion kernel without number of categories. Please define them!")
        super().__init__(**kwargs)
        self.cats = categories

    def forward(self, x1, x2, diag: Optional[bool] = False, last_dim_is_batch: Optional[bool] = False, **params):
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)        

        if diag:
            res = 1.
            for i in range(x1.shape[-1]):
                res *= ((1 - torch.exp(-self.lengthscale[..., i] * self.cats[i]))/(1 + (self.cats[i] - 1) * torch.exp(-self.lengthscale[..., i]*self.cats[i]))).unsqueeze(-1) ** ((x1[..., i] != x2[..., i])[:, 0, ...])
            return res

        res = 1.
        for i in range(x1.shape[-1]): 
            res *= ((1 - torch.exp(-self.lengthscale[..., i] * self.cats[i]))/(1 + (self.cats[i] - 1) * torch.exp(-self.lengthscale[..., i]*self.cats[i]))).unsqueeze(-1) ** ((x1[..., i].unsqueeze(-2)[..., None] != x2[..., i].unsqueeze(-2))[0, ...])
        return res             
    
def training(model, X_train, y_train, n_epochs=200, lr=0.3, loss_threshold=0.00001, fix_noise_variance=None, verbose=True):
    '''
    Training function for GPs. 

    Parameters
    ----------
    model : gpytorch.models.ExactGP
        GP model to be trained.
    X_train : torch.tensor
        Training data.
    y_train : torch.tensor
        Training labels.
    n_epochs : int, optional
        Number of epochs to train for. The default is 200.
    lr : float, optional
        Learning rate. The default is 0.3.
    loss_threshold : float, optional
        Threshold for loss. The default is 0.00001.
    fix_noise_variance : float, optional
        If not None, fix the noise variance to this value. The default is None.
    verbose : bool, optional
        If True, print loss at each epoch. The default is True.
    
    Returns
    -------
    ls : list
        List of losses at each epoch.
    mll : gpytorch.mlls.ExactMarginalLogLikelihood
        Marginal log likelihood.
    '''
    model.train()
    model.likelihood.train()
    
    try:
        n_comp = len([m for m in model.covar_module.data_covar_module.kernels])
        for i in range(n_comp):
            model.covar_module.data_covar_module.kernels[i].outputscale = (1 / n_comp)
    except AttributeError:
        n_comp = 1

    # Use the adam optimizer
    if fix_noise_variance is not None:
        model.likelihood.noise = fix_noise_variance
        training_parameters = [p for name, p in model.named_parameters()
                               if not name.startswith('likelihood')]
    else:
        training_parameters = model.parameters()
        
    optimizer = torch.optim.Adam(training_parameters, lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    model = model.double()
    counter = 0
    ls = list()
    with tqdm.trange(n_epochs, disable=not verbose) as bar:
        for i in bar:
    
            optimizer.zero_grad()
            
            output = model(X_train.double())
            loss = -mll(output, y_train)
            if hasattr(model.covar_module.data_covar_module, 'kernels'):
                with torch.no_grad():
                    for j in range(n_comp):
                        model.covar_module.data_covar_module.kernels[j].outputscale =  \
                        model.covar_module.data_covar_module.kernels[j].outputscale /  \
                        sum([model.covar_module.data_covar_module.kernels[i].outputscale for i in range(n_comp)])
            else:
                pass
            loss.backward()
            ls.append(loss.item())
            optimizer.step()
            if (i > 0):
                if abs(ls[counter - 1] - ls[i]) < loss_threshold:
                    break
            counter = counter + 1
                        
            # display progress bar
            postfix = dict(Loss=f"{loss.item():.3f}",
                           noise=f"{model.likelihood.noise.item():.3}")
            
            if (hasattr(model.covar_module, 'base_kernel') and
                hasattr(model.covar_module.base_kernel, 'lengthscale')):
                lengthscale = model.covar_module.base_kernel.lengthscale
                if lengthscale is not None:
                    lengthscale = lengthscale.squeeze(0).detach().cpu().numpy()
            else:
                lengthscale = model.covar_module.lengthscale

            if lengthscale is not None:
                if len(lengthscale) > 1:
                    lengthscale_repr = [f"{l:.3f}" for l in lengthscale]
                    postfix['lengthscale'] = f"{lengthscale_repr}"
                else:
                    postfix['lengthscale'] = f"{lengthscale[0]:.3f}"
                
            bar.set_postfix(postfix)
            
    return ls, mll