# Heavily based on examples on the Baccus github (https://github.com/jl-bernal/BACCUS), many thanks also to Bernal for comments e.g. https://github.com/jl-bernal/BACCUS/blob/master/examples/baccus_example_nb.ipynb


# Import everything
from baccus import BACCUS
import numpy as np
from numpy.random import normal
import theano
import theano.tensor as T

import time

import matplotlib.pyplot as plt
import pylab as pb
import scipy



ndim=number_probes


#Number of walkers # Set as appropriate
nwalkers = 100
steps = 250000



#Variance shift priors
b = -1
xi = 16

number_probes=len(means)

DATA = [None] * number_probes


# Set up multivariate distributions of the probes
for i in range(number_probes):
    
    mean = [means[i], 0.]
    cov = [[sds[i]**2, 0.0], [0.0, 0.1]]  # diagonal covariance
    x, y = np.random.multivariate_normal(mean, cov, 500).T
    sx = np.ones(500)*((prior_max-prior_min)/10.0)
    sy = np.ones(500)*0.2
    
    data_initial = np.stack((x,y,sx,sy),axis=1)
    DATA[i]=data_initial
    
    

prior_bounds_model=[None] * ndim
prior_bounds_shifts=[None] * ndim
prior_bounds_var=[None] * ndim
kind_shifts=[None] * number_probes

for i in range(ndim):
    if i==0:
        prior_bounds_model[i]=(prior_min,prior_max)
    else:
        prior_bounds_model[i]=(-20.,20.)
        
    
    prior_bounds_shifts[i]=(i,-(prior_max-prior_min),prior_max-prior_min)
    prior_bounds_var[i]=(i,0,prior_max-prior_min)

for i in range(number_probes):
    kind_shifts[i]=(0,1)
    



##prior for the variances of the shifts - a lognormal
y = T.dscalar('y')
s = -0.5*((T.log(y)-b)/2./xi)**2.
logprior_sigma2 = theano.function([y],s)
prior_var = []
for ivar in range(0,2):
    prior_var += [logprior_sigma2]
    
    

### With Rescale


def lkl_rescale(theta,DATA):
    x = theta[0]
    y = theta[1]
    
    nrescaling = len(DATA)
    count = 0
    npar = 2
    
    log_lkl = 0
    for i in range(0,len(DATA)):
        alpha = theta[npar+i]
        shift_x = theta[npar+nrescaling+count]
        count += 1
        shift_y = theta[npar+nrescaling+count]
        count += 1
        
        xpar = x+shift_x
        ypar = y+shift_y

        dat = DATA[i]
        xdat = dat[:,0]
        ydat = dat[:,1]
        sx = dat[:,2]
        sy = dat[:,3]
        
        log_lkl -= 0.5*alpha*(np.sum(((xdat-xpar)/sx)**2.) + np.sum(((ydat-ypar)/sy)**2.))
        
    return log_lkl
    
    




# Create the model
want_rescaling = True
model_res = BACCUS(prior_bounds_model=prior_bounds_model,prior_bounds_shifts=prior_bounds_shifts,prior_bounds_var=prior_bounds_var,
              lkl = lkl_rescale, kind_shifts = kind_shifts,prior_var=prior_var,want_rescaling=want_rescaling)
              


#Set initial position, steps and walkers
pos = []

for i in range(0,nwalkers):
    #Model parameters
    pos += [np.array([normal((prior_max+prior_min)/2.0,(prior_max-prior_min)/10.0),normal(0,1)])]#[np.array([normal(0,1),normal(0,1)])]
    #Rescaling parameters, if wanted
    if want_rescaling:
        for j in range(0,number_probes):
            pos[i] = np.append(pos[i],normal(1,0.2))
    #shift_hyperparams
    for j in range(0,number_probes):
        pos[i] = np.append(pos[i],normal(0.,1.))
        pos[i] = np.append(pos[i],normal(0.,1.))
                
    #var for shifts
    pos[i] = np.append(pos[i],normal(1.,0.2))
    pos[i] = np.append(pos[i],normal(1.,0.2))
        
    #correlation of shifts
    pos[i] = np.append(pos[i],normal(0.,0.2))
    

# Run the MCMC
tic = time.clock()
model_res.run_chain(DATA, stepsize=2., pos=pos, nwalkers=nwalkers, steps=steps)
toc = time.clock()
duration= toc - tic



###### Plot the 1D


chain=model_res.INFO.chain

chain=chain.reshape((nwalkers*steps, np.shape(chain)[2]))


plt.figure()
n_bins=50
n_vals=np.zeros([number_probes,n_bins])
bin_vals=np.zeros([number_probes,n_bins+1])

n_final, bins_final, patches = plt.hist(chain[:,0], n_bins,normed=True, facecolor='b', alpha=0.75)



for i in range(number_probes):
    mean = [means[i], 0.]
    cov = [[sds[i]**2, 0.0], [0.0, 0.1]]  # diagonal covariance
    x, y = np.random.multivariate_normal(mean, cov, 500000).T
    n_vals[i,:], bin_vals[i,:], patches = plt.hist(x, n_bins,normed=True, facecolor='r', alpha=0.75)







plt.figure()


pb.plot(bins_final[:n_bins],n_final[0:n_bins]/np.max(n_final[0:n_bins]),'b',label=r"Conservative Estimate")

for i in range(number_probes):
    if i==0:
        pb.plot(bin_vals[i,:n_bins],n_vals[i,0:n_bins]/np.max(n_vals[i,0:n_bins]),'r',label=r"Independent Estimates")
    else:
        pb.plot(bin_vals[i,:n_bins],n_vals[i,0:n_bins]/np.max(n_vals[i,0:n_bins]),'r')
    
    


pb.xscale("linear")
pb.xlabel(r'$E$ (MJ)', fontsize=20)
pb.ylabel(r'$P/P_{MAX}$', fontsize=20)

values=np.arange(prior_min,prior_max,(prior_max-prior_min)/200.0)#np.arange(2,12,0.01)
pdf=np.zeros(len(values))

for i in range(len(values)):

    x=1.0
    for j in range(number_probes):
        mean = [means[j], 0.]
        cov = [[sds[j]**2, 0.0], [0.0, 0.1]]  # diagonal covariance
        x=x*scipy.stats.norm.pdf(values[i], mean[0], cov[0][0]**0.5)
    pdf[i]=x


pdf=pdf/(np.sum(pdf)*(values[2]-values[1]))

pb.plot(values,pdf/np.max(pdf),'k',label=r"Conventional Estimate")
#pb.legend(labels)
plt.legend(loc='upper right')#loc='upper right'

pb.xlim(2,12)
pb.xlim(prior_min,prior_max)

pb.yscale("log")
pb.ylim(1e-6,1)

pb.plot(param_values,param_posterior,'b',label='Posterior (Press96)')

