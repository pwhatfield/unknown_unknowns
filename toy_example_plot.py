# Code to run P96 and B18 and plot the results

#Define backend
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
import numpy as np

from numpy import floor as floor
from numpy import log10 as log10

# Where scripts are to be plus where figures will do
main_directory='...'

# Define font size
label_fontsize=10


#### TOY
# Hard prior beyond which you have very high confidence 
prior_min=-10
prior_max=10

# Probability density range (for plotting purposes only)
graph_prob_max=2
grap_prob_min=1e-3

# Number of probes used
number_probes=2

# The means and standard deviations of the estimates from the independent probes
means=[-1,1]
sds=[0.5,0.5]

execfile(main_directory+'press96_toy.py')
#execfile(main_directory+'baccus_1D_toy.py')

# Set up a figure
plt.figure()
plt.subplots_adjust(hspace = 0.5)
plt.subplots_adjust(wspace = 0.4)


#Plot log probabilities in top subplot
plt.subplot(211)
plt.title('Toy Example')



#for i in range(number_probes):
    if i==0:
        pb.plot(bin_vals[i,:n_bins],n_vals[i,0:n_bins]/(np.sum(n_vals[i,0:n_bins])*(bin_vals[i,:n_bins][1]-bin_vals[i,:n_bins][0])),color=faintness,label=r"Independent Estimates")
    else:
        pb.plot(bin_vals[i,:n_bins],n_vals[i,0:n_bins]/(np.sum(n_vals[i,0:n_bins])*(bin_vals[i,:n_bins][1]-bin_vals[i,:n_bins][0])),color=faintness)
    
    
##Cut first half of samples
n_final, bins_final = np.histogram(chain[0:(np.shape(chain[:,0])[0]/2),0], n_bins,normed=True)

pb.plot(bins_final[:n_bins],n_final[0:n_bins]/(np.sum(n_final[0:n_bins])*(bins_final[:n_bins][1]-bins_final[:n_bins][0])),'b',label=r"B18")



pb.plot(param_values,param_posterior,'g',label='P96') # P96 probabilities
pb.plot(values,pdf/(np.sum(pdf)*(values[1]-values[0])),'k',label=r"CB") # Conventional Bayes Probabilities

# Axis labels etc.
pb.xlabel(r'$x$', fontsize=label_fontsize)
pb.ylabel(r'$P$', fontsize=label_fontsize)
pb.xlim(prior_min,prior_max)
pb.ylim(grap_prob_min,graph_prob_max)
pb.xscale("linear")
pb.yscale("log")

#Plot linear probabilities in bottom subplot
plt.subplot(212)


#Plot log probabilities

for i in range(number_probes):
    if i==0:
        pb.plot(bin_vals[i,:n_bins],n_vals[i,0:n_bins]/(np.sum(n_vals[i,0:n_bins])*(bin_vals[i,:n_bins][1]-bin_vals[i,:n_bins][0])),color=faintness,label=r"Independent Estimates")
    else:
        pb.plot(bin_vals[i,:n_bins],n_vals[i,0:n_bins]/(np.sum(n_vals[i,0:n_bins])*(bin_vals[i,:n_bins][1]-bin_vals[i,:n_bins][0])),color=faintness)
    
    
pb.plot(bins_final[:n_bins],n_final[0:n_bins]/(np.sum(n_final[0:n_bins])*(bins_final[:n_bins][1]-bins_final[:n_bins][0])),'b',label=r"B18")


pb.plot(param_values,param_posterior,'g',label='P96') # P96 probabilities
pb.plot(values,pdf/(np.sum(pdf)*(values[1]-values[0])),'k',label=r"CB") # Conventional Bayes Probabilities


# Axis labels etc.
pb.xlabel(r'$x$', fontsize=label_fontsize)
pb.ylabel(r'$P$', fontsize=label_fontsize)
plt.legend(loc='upper right')
pb.xlim(prior_min,prior_max)
pb.ylim(0,graph_prob_max)
pb.xscale("linear")
pb.yscale("linear")

# Save figure
pb.savefig(main_directory+'unknown_unknown_figure.pdf')
