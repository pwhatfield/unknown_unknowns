# Script to calculate P96 probabilities

#Import everything...
import numpy as np
import pylab as pb
pb.ion()
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy import special
from scipy import stats
from scipy.integrate import simps
import time




##### Define Gaussian
def gaussian(x, mu, sig):
    return np.exp(-((x - mu)**2.)/(2.*sig*sig))* ((2*np.pi*(sig**2.))**-0.5)

#############
## Set up data



#############
## Set up possible param values



param_increment=(prior_max-prior_min)/100.0
param_values=np.arange(prior_min,prior_max,param_increment)

rightness_min=0
rightness_max=1
rightness_increment=0.01
rightness_values=np.arange(rightness_min,rightness_max,rightness_increment)

#############
### Set up priors

## Prior of parameter

# Function for prior
def param_prior_function(x):
    value=1 # Uniform prior (likely should be uniform in log-space for scale parameter)
    return value

# Give values
param_prior=np.zeros(len(param_values))
for i in range(len(param_prior)):
    param_prior[i]=param_prior_function(param_values[i])

# Normalise
param_prior=param_prior/(np.sum(param_prior)*param_increment)


## Prior of prob of being right
def rightness_prior_function(x):
    if x<0.05:
        value=0
    elif x>0.95:
        value=0
    else:
        value=(x*(1-x))**-0.5
    
    return value

# Give values
rightness_prior=np.zeros(len(rightness_values))
for i in range(len(rightness_prior)):
    rightness_prior[i]=rightness_prior_function(rightness_values[i])

# Normalise
rightness_prior=rightness_prior/(np.sum(rightness_prior)*rightness_increment)


#############
### Set up posteriors
param_posterior=np.zeros(len(param_values)) # Posterior for actual parameter of interest
rightness_posterior=np.zeros(len(rightness_values)) # Posterior for the parameter describing probability of being right

traditional_param_posterior=np.zeros(len(param_values))


#############
### Do calculation for param posterior

for i in range(len(param_posterior)):
        
    total_summed_over=0
    
    for j in range(len(rightness_posterior)):
        
        individual_summand=rightness_prior[j]
        
        for k in range(number_probes):
            
            # Equation 16 in P96
            individual_summand=individual_summand*( (rightness_values[j]*gaussian(means[k], param_values[i], sds[k]))  +  ((1-rightness_values[j])*param_prior[i])  )
            
    
        total_summed_over=total_summed_over+individual_summand
        
    
    param_posterior[i]=param_prior[i]*total_summed_over
        

# This is our P96 posterior
param_posterior=param_posterior/(np.sum(param_posterior)*param_increment)


### Do calculation for traditional param posterior
for i in range(len(param_posterior)):
        
    liklihood=1

    for k in range(number_probes):
        liklihood=liklihood*gaussian(means[k], param_values[i], sds[k])
            
    traditional_param_posterior[i]=param_prior[i]*liklihood
   

# This is a conventional posterior 
traditional_param_posterior=traditional_param_posterior/(np.sum(traditional_param_posterior)*param_increment)


### Do calculation for rightness posterior

for i in range(len(rightness_posterior)):
        
    total_summed_over=0
    
    for j in range(len(param_prior)):
        
        individual_summand=param_prior[j]
        
        for k in range(number_probes):
            
            individual_summand=individual_summand*( (rightness_values[i]*gaussian(means[k], param_values[j], sds[k]))  +  ((1-rightness_values[i])*param_prior[j])  )#+  ((1-rightness_values[i])*gaussian(means[k], param_values[j], S))  )
            
    
        total_summed_over=total_summed_over+individual_summand
        
    
    rightness_posterior[i]=rightness_prior[i]*total_summed_over
        
# This is the posterior for probability of being right
rightness_posterior=rightness_posterior/(np.sum(rightness_posterior)*rightness_increment)


### Calculate prob vectors ###

priors_v=np.zeros(2**number_probes)/(2**number_probes)
posteriors_v=np.zeros(2**number_probes)
v_vals=np.zeros([2**number_probes,number_probes])


for i in range(2**number_probes):
    
    for j in range(number_probes):
        
        v_vals[i,j]=float(format(i, '#0'+str(number_probes+2)+'b')[j+2])


for i in range(2**number_probes):
        
    total_summed_over=0
    
    for j1 in range(len(param_prior)):
        
        for j2 in range(len(rightness_prior)):
        
            individual_summand=param_prior[j1]*rightness_prior[j2]
            
            for k in range(number_probes):
                
                if v_vals[i,k]==1:
                
                    individual_summand=individual_summand*( (rightness_values[j2]*gaussian(means[k], param_values[j1], sds[k]))   )
                else:
                    individual_summand=individual_summand*(1-rightness_values[j2])*param_prior[j1] #individual_summand*(  ((1-rightness_values[j2])*gaussian(means[k], param_values[j1], S))  )
                
        
            total_summed_over=total_summed_over+individual_summand
            #print individual_summand
        
    print total_summed_over
    posteriors_v[i]=total_summed_over

        
 

posteriors_v=posteriors_v/(np.sum(posteriors_v))

individual_probs=np.zeros(number_probes)

for i in range(number_probes):
    
    for j in range(2**number_probes):
        
        if v_vals[j,i]==1:
            
            individual_probs[i]=individual_probs[i]+posteriors_v[j]





plt.figure(1)

plt.subplot(211)
pb.plot(param_values,param_posterior,'b',label='Posterior (Press96)')
pb.plot(param_values,traditional_param_posterior,'g',label='Posterior (Traditional)')
pb.plot(param_values,param_prior,'r',label='Prior')
#pb.legend()
pb.xlim(prior_min, prior_max)
pb.ylim(0.000001,np.max([np.max(param_posterior),np.max(traditional_param_posterior)]))
pb.ylabel(r'Posterior')
pb.yscale("log")

plt.subplot(212)
pb.errorbar(means, individual_probs, yerr=None, xerr=sds,marker='o',linestyle='None')
pb.ylim(-0.2,1.2)
pb.xlim(prior_min, prior_max)
pb.ylabel(r'Probability of Measurement Correct')
pb.xlabel(r'log10 Solar Mass')


pb.savefig(main_directory+"P96_toy_1.png")


plt.figure(2)
pb.plot(rightness_values,rightness_posterior,'b',label='Posterior (Press96)')
pb.plot(rightness_values,rightness_prior,'r',label='Prior')

pb.xlim(0,1)
pb.ylim(0,np.max(rightness_posterior))
pb.xlabel(r'Global Probability of Measurement Being Correct')
pb.ylabel(r'Posterior')
pb.legend()
#pb.yscale("log")


pb.savefig(main_directory+"P96_toy_2.png")
