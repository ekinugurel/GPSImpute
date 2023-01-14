import torch
import matplotlib
import gpytorch
import copy
import numpy as np
import pandas as pd
import tqdm

def plot_kernel(kernel, xlim=None, ax=None):
    if xlim is None:
        xlim = [-3, 5]
    x = torch.linspace(xlim[0], xlim[1], 100)
    with torch.no_grad():
        K = kernel(x, torch.ones((1))).evaluate().reshape(-1, 1)
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.plot(x.numpy(), K.cpu().numpy())
    
# Find optimal model hyperparameters
def training(model, X_train, y_train, n_epochs=200, lr=0.3, loss_threshold=0.00001, fix_noise_variance=None, verbose=True):
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
    
    counter = 0
    ls = list()
    with tqdm.trange(n_epochs, disable=not verbose) as bar:
        for i in bar:
    
            optimizer.zero_grad()
            
            output = model(X_train)
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
            
def train_model_get_bic(X_train, y_train, kernel, n_epochs=300):
    """
    Train GP model and calculate Bayesian Information Criterion (BIC)
    
    Parameters
    ----------
    X_train : torch.tensor
        Array of train features, n*d (d>=1)
    
    y_train : torch.tensor
        Array of target values
        
    kernel : gpytorch.kernels.Kernel
        Kernel object
        
    n_epochs : int
        Number of epochs to train GP model
        
    Returns
    -------
    bic : float
        BIC value
    """
    kernel = copy.deepcopy(kernel)
    
    model = MTGPRegressor(X_train, y_train, kernel)
    
    try:
        n_comp = len([m for m in model.covar_module.data_covar_module.kernels])
        for i in range(n_comp):
            model.covar_module.data_covar_module.kernels[i].outputscale = (1 / n_comp)
    except AttributeError:
        n_comp = 1
        
    ls, mll = training(model, X_train, y_train, n_epochs=n_epochs, verbose=False)
    
    with torch.no_grad():
        log_ll = mll(model(X_train), y_train) * X_train.shape[0]
        
    N = X_train.shape[0]
    m = sum(p.numel() for p in model.hyperparameters())
    bic = -2 * log_ll + m * np.log(N)

    return bic, ls 
    
def _get_all_product_kernels(op_list, kernel_list):
    """
    Find product pairs and calculate them.
    For example, if we are given expression:
        K = k1 * k2 + k3 * k4 * k5
    the function will calculate all the product kernels
        k_mul_1 = k1 * k2
        k_mul_2 = k3 * k4 * k5
    and return list [k_mul_1, k_mul_2].
    """
    product_index = np.where(np.array(op_list) == '*')[0]
    if len(product_index) == 0:
        return kernel_list

    product_index = product_index[0]
    product_kernel = kernel_list[product_index] * kernel_list[product_index + 1]
    
    if len(op_list) == product_index + 1:
        kernel_list_copy = kernel_list[:product_index] + [product_kernel]
        op_list_copy = op_list[:product_index]
    else:
        kernel_list_copy = kernel_list[:product_index] + [product_kernel] + kernel_list[product_index + 2:]
        op_list_copy = op_list[:product_index] + op_list[product_index + 1:]
        
    return _get_all_product_kernels(op_list_copy, kernel_list_copy)