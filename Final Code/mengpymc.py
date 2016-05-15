import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from pymc3 import Model, Normal, HalfNormal, Binomial
from pymc3 import find_MAP
from scipy import optimize


# Intialize random number generator
np.random.seed(123)

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.linspace(0, 1, size)
X2 = np.linspace(0,.2, size)

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma



fig, axes = subplots(1, 2, sharex=True, figsize=(10,4))
axes[0].scatter(X1, Y)
axes[1].scatter(X2, Y)
axes[0].set_ylabel('Y'); axes[0].set_xlabel('X1'); axes[1].set_xlabel('X2');

plt.show()




basic_model = Model()

with basic_model:

    # Priors for unknown model parameters
    alpha = Normal('alpha', mu=0, sd=10)
    beta = Normal('beta', mu=0, sd=10, shape=2)
    sigma = HalfNormal('sigma', sd=1)

    # Expected value of outcome
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

# map_estimate = find_MAP(model=basic_model)

# print(map_estimate)

map_estimate = find_MAP(model=basic_model, fmin=optimize.fmin_powell)

print(map_estimate)


from pymc3 import NUTS, sample
from pymc3 import traceplot

with basic_model:

    # obtain starting values via MAP
    start = find_MAP(fmin=optimize.fmin_powell)

    # instantiate sampler
    step = NUTS(scaling=start)

    # draw 2000 posterior samples
    trace = sample(2000, step, start=start)
    trace['alpha'][-5:]
    traceplot(trace)
    plt.show()




from pymc3 import summary
summary(trace)

n = 500
p = 0.3
with Model():
	x = Normal('alpha', mu=0, sd=10)
	print type(x)