[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ekinugurel/GPSImpute/master)
# Correcting missingness in passively-generated mobile data with Multi-Task Gaussian Processes
### Ekin Uğurel, Xiangyang Guan, Yanchao Wang, Qi Ryan Wang, Shuai Huang, Cynthia Chen
### Read full paper <a href="https://www.sciencedirect.com/science/article/pii/S0968090X24000445?dgcid=author">here</a>
 
GPSImpute is a library to impute missing values for mobile datasets, largely in the context of human mobility. It uses multi-task Gaussian processes to infer missing GPS traces. In practice, for short gaps in continuous coverage, our method is comparable to existing smoothing techniques. For long gaps, however, our method cleverly takes advantage of an individual's longitudinal data to predict missing segments.

![End-to-end Workflow.](img/endtoendworkflow_2.png)

## Descriptive Data Analysis
<!-- <img src="img/locacc.png" alt="" title="Cumulative distribution of location accuracy in Spectus data" width="600"/> -->

<!-- <img src="img/timeofday.png" alt="" title="Observation distribution over the time of day" width="400"/> -->

![Observation distribution over the time of day](img/timeofday.png)

![Histograms of missingness levels in Spectus data](img/missingness1.png)

![Cumulative distribution of location accuracy in Spectus data](img/locacc.png)
## Installation
```
$ git clone https://github.com/ekinugurel/GPSImpute.git
$ cd ./GPSImpute
$ pip install -e .
```
## Model Inputs
![Variables used in model](img/table1.png)

## Usage
#### Initialization Strategy & Parameter Optimization
The optimization problem of a GP is non-convex. Therefore, kernel parameter initialization can help avoid bad local optima, leading to better model estimation and more accurate prediction results. We achieve this in two ways. First, if the gap is short enough, we take advantage of training points near the gap to infer trip characteristics. These include metrics like average velocity, total distance traveled, trip duration, and stop rate, among others. Our first set of experiments suggest that trips with differing characteristics tend to land at different optimal parameters. Therefore, we suggest initializing the lengthscale parameter of continuous temporal dimensions (Unix time in our case) according to the characteristics of the trip with missing data. 

If using a monotonically-increasing temporal dimension as an input, one alternative we have identified is to initialize the lengthscale parameter with the average length of gap between observations in the training data. Ak et al. (2018) employed this strategy without any optimization. We however do further optimize the parameter beyond this initial heuristic using the Adaptive Moment Estimation algorithm (Kingma and Ba, 2017), a preferred choice among various machine learning frameworks due to its computational efficiency and ability to manage sparse gradients on noisy data.

We initialize binary variables (i.e., one-of-k encoded categorical variables) with a lengthscale of 1. This is largely done to avoid model misspecification during the optimization stage as lengthscale parameters are constrained to be nonnegative. In our implementation, a small amount of noise (called jitter) is added to the diagonal of the covariance matrix. This is done to ensure numerical stability when performing matrix operations like inversion and decomposition. Simultaneously, however, jitter can lead to parameter values dipping below 0 at the very first iteration of the optimization algorithm---if the learning rate is small, the first iteration of the optimization algorithm may result in a value that is very close to 0, and the associated jitter may cause the parameter to flip signs.

![Example imputation](img/ex1.png)

## Results
![1 week of missingness - Relationship between sparsity and the derived mobility metrics](img/1week.png)
![1 day of missingness - Relationship between sparsity and the derived mobility metrics](img/1day.png)
![6 hours of missingness - Relationship between sparsity and the derived mobility metrics](img/6hours.png)
![5 minutes of missingness - Relationship between sparsity and the derived mobility metrics](img/5mins.png)

## Licensing
See the [LICENSE](LICENSE) file for licensing information as it pertains to files in this repository.

## Contact
Ekin Uğurel (ugurel [at] uw.edu)

## Citation
```
@article{UGUREL2024104523,
title = {Correcting missingness in passively-generated mobile data with Multi-Task Gaussian Processes},
journal = {Transportation Research Part C: Emerging Technologies},
volume = {161},
pages = {104523},
year = {2024},
issn = {0968-090X},
doi = {https://doi.org/10.1016/j.trc.2024.104523},
url = {https://www.sciencedirect.com/science/article/pii/S0968090X24000445},
author = {Ekin Uğurel and Xiangyang Guan and Yanchao Wang and Shuai Huang and Qi Wang and Cynthia Chen},
keywords = {Mobile data, Human mobility, Travel behavior, Gaussian processes, Multi-task learning, Missing data}
```
