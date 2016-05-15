import pymc3 as pm
import scipy as sp
from IPython.display import Image
import prettyplotlib as ppl
from prettyplotlib import plt
import seaborn as sns
sns.set_context('talk')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size': 22})
rc('xtick', labelsize=12) 
rc('ytick', labelsize=12)
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

from IPython.html import widgets # Widget definitions
from IPython.display import display # Used to display widgets in the notebook
from IPython.html.widgets import interact, interactive
from IPython.display import clear_output, display, HTML

import pymc3 as pm
model = pm.Model()

np.random.seed(9)

# Change this to reflect the Iphone points
algo_a = sp.stats.bernoulli(.5).rvs(300) # 50% profitable days

# Change this to reflect the nonIphone users
algo_b = sp.stats.bernoulli(.6).rvs(300) # 60% profitable days


# This model approximates the avg time spent per session as a distribution

with model: # model specifications in PyMC3 are wrapped in a with-statement
    # Define random variables
    theta_a = pm.Normal('theta_a', mu=15, sd=5) # prior
    theta_b = pm.Normal('theta_b', mu=15, sd=5) # prior
    
    # Define how data relates to unknown causes
    data_a = pm.Normal('observed A',
                          p=theta_a, 
                          observed=algo_a)
    
    data_b = pm.Normal('observed B', 
                          p=theta_b, 
                          observed=algo_b)
    
    # Inference!
    start = pm.find_MAP() # Find good starting point
    step = pm.Slice() # Instantiate MCMC sampling algorithm
    trace = pm.sample(10000, step, start=start, progressbar=False) # draw posterior samples using slice sampling 