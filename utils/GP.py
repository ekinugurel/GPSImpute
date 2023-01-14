import torch
import gpytorch
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch import kernels
from gpytorch.kernels import MultitaskKernel, ScaleKernel, Kernel
from typing import Optional
from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior

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
            self.predictions = self.likelihood(self(X))
            self.mean = self.predictions.mean
            self.lower, self.upper = self.predictions.confidence_region()
            self.sigma1_l = self.predictions.mean[:, 0] - self.predictions.stddev[:, 0]
            self.sigma1_u = self.predictions.mean[:, 0] + self.predictions.stddev[:, 0]
            self.sigma2_l = self.predictions.mean[:, 1] - self.predictions.stddev[:, 1]
            self.sigma2_u = self.predictions.mean[:, 1] + self.predictions.stddev[:, 1]

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