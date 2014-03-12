import numpy as np
import matplotlib.pyplot as pl
import brewer2mpl
from scipy.integrate import simps
from scipy.stats import linregress

dat = np.load('posterior.npz')

pB, normaliz, B, sigmas, fluxes = dat['arr_0'], dat['arr_1'], dat['arr_2'], dat['arr_3'], dat['arr_4']
labels = [r'2 Mx cm$^{-2}$',r'5 Mx cm$^{-2}$',r'10 Mx cm$^{-2}$',r'15 Mx cm$^{-2}$',r'20 Mx cm$^{-2}$',r'40 Mx cm$^{-2}$',r'80 Mx cm$^{-2}$',r'100 Mx cm$^{-2}$']
labels = [r'$\Phi_\mathrm{obs}$='+i for i in labels]

nSigmas = len(sigmas)
nFluxes = len(fluxes)
MMAP = np.zeros((nSigmas,nFluxes))
B95 = np.zeros((nSigmas,nFluxes))
B68 = np.zeros((nSigmas,nFluxes))
fwhmPrior = 1500.0
sigmaPrior = fwhmPrior / (2.0 * np.sqrt(2.0*np.log(2.0)))
priorGaussian = np.exp(-0.5*B**2 / sigmaPrior**2)

B0 = 38
sigmaPrior = 1.7 / np.sqrt(2.0)
priorLogN = np.exp(-(np.log(B)-np.log(B0))**2 / (2.0*sigmaPrior**2)) / B

prior = priorLogN

pl.close('all')
f, ax = pl.subplots(1, 2, figsize=(12,4), num=1)
ax = ax.flatten()

colors = brewer2mpl.get_map('Dark2', 'qualitative', nFluxes).mpl_colors

loop = 0

for i in range(len(fluxes)):
	posterior = prior * pB[2,i,:]
	normaliz = simps(posterior, B)
	ax[loop].plot(B, posterior / normaliz, label=labels[i], linewidth=2, color=colors[i])
	MMAP[2,i] = B[posterior.argmax()]	
ax[loop].set_ylabel(r'p(B|$\Phi_\mathrm{obs}$)')
ax[loop].set_xlabel('B [G]')
#ax[loop].annotate(r'$\sigma_n$='+"{0:4.1f}".format(sigmas[2])+r' Mx cm$^{-2}$', xy=(0.55, 0.86), xycoords='axes fraction', fontsize=14)
ax[loop].set_xlim((0,400))
ax[loop].legend(labelspacing=0.2, prop={'size':12}, loc='upper right')
loop += 1
			
for i in range(len(fluxes)):
	posterior = prior * pB[2,i,:]
	cum = np.cumsum(posterior)
	ax[loop].plot(B, cum / np.max(cum), label=labels[i], linewidth=2, color=colors[i])
		
	cum = cum / np.max(cum)
	print "{0} at 68% - {1} at 95%".format(B[np.argmin(np.abs(cum/np.max(cum)-0.68))],B[np.argmin(np.abs(cum/np.max(cum)-0.95))])
	B68[2,i] = B[np.argmin(np.abs(cum/np.max(cum)-0.68))]
	B95[2,i] = B[np.argmin(np.abs(cum/np.max(cum)-0.95))]
				
ax[loop].set_ylabel('cdf(B|$\Phi_\mathrm{obs}$)')
ax[loop].set_xlabel('B [G]')
ax[loop].set_xlim((0,600))
	
pl.tight_layout()
pl.savefig("posteriorFlux.pdf")

#--------------------------
pl.clf()
f, ax = pl.subplots(1, 2, figsize=(12,4), num=1)
ax = ax.flatten()
labels = [r'0.1 Mx cm$^{-2}$',r'1 Mx cm$^{-2}$',r'5 Mx cm$^{-2}$',r'10 Mx cm$^{-2}$',r'20 Mx cm$^{-2}$']
labels = [r'$\sigma_n$='+i for i in labels]
colors = brewer2mpl.get_map('Dark2', 'qualitative', nSigmas).mpl_colors

loop = 0

for i in range(len(sigmas)):
	posterior = prior * pB[i,5,:]
	normaliz = simps(posterior, B)
	ax[loop].plot(B, posterior / normaliz, label=labels[i], linewidth=2, color=colors[i])
	MMAP[2,i] = B[posterior.argmax()]	
ax[loop].set_ylabel(r'p(B|$\Phi_\mathrm{obs}$)')
ax[loop].set_xlabel('B [G]')
#ax[loop].annotate(r'$\sigma_n$='+"{0:4.1f}".format(sigmas[2])+r' Mx cm$^{-2}$', xy=(0.55, 0.86), xycoords='axes fraction', fontsize=14)
ax[loop].set_xlim((0,300))
ax[loop].legend(labelspacing=0.2, prop={'size':12}, loc='upper right')
loop += 1
			
for i in range(len(sigmas)):
	posterior = prior * pB[i,5,:]
	cum = np.cumsum(posterior)
	ax[loop].plot(B, cum / np.max(cum), label=labels[i], linewidth=2, color=colors[i])
		
	cum = cum / np.max(cum)
	print "{0} at 68% - {1} at 95%".format(B[np.argmin(np.abs(cum/np.max(cum)-0.68))],B[np.argmin(np.abs(cum/np.max(cum)-0.95))])
	B68[2,i] = B[np.argmin(np.abs(cum/np.max(cum)-0.68))]
	B95[2,i] = B[np.argmin(np.abs(cum/np.max(cum)-0.95))]
				
ax[loop].set_ylabel('cdf(B|$\Phi_\mathrm{obs}$)')
ax[loop].set_xlabel('B [G]')
ax[loop].set_xlim((0,300))
	
pl.tight_layout()
pl.savefig("posteriorNoise.pdf")

#---------------------------
dat = np.load('posteriorAnisot.npz')

pB, normaliz, B, a, fluxes = dat['arr_0'], dat['arr_1'], dat['arr_2'], dat['arr_3'], dat['arr_4']
na = len(a)

pl.clf()
f, ax = pl.subplots(1, 2, figsize=(12,4), num=1)
ax = ax.flatten()
labels = ['-0.5', '0', '1', '2','5','10']
labels = ['a='+i for i in labels]
colors = brewer2mpl.get_map('Dark2', 'qualitative', na).mpl_colors

loop = 0

for i in range(len(a)):
	posterior = prior * pB[i,5,:]
	normaliz = simps(posterior, B)
	ax[loop].plot(B, posterior / normaliz, label=labels[i], linewidth=2, color=colors[i])
	MMAP[2,i] = B[posterior.argmax()]	
ax[loop].set_ylabel(r'p(B|$\Phi_\mathrm{obs}$)')
ax[loop].set_xlabel('B [G]')
#ax[loop].annotate(r'$\sigma_n$='+"{0:4.1f}".format(sigmas[2])+r' Mx cm$^{-2}$', xy=(0.55, 0.86), xycoords='axes fraction', fontsize=14)
ax[loop].set_xlim((0,300))
ax[loop].legend(labelspacing=0.2, prop={'size':12}, loc='upper right')
loop += 1
			
for i in range(len(a)):
	posterior = prior * pB[i,5,:]
	cum = np.cumsum(posterior)
	ax[loop].plot(B, cum / np.max(cum), label=labels[i], linewidth=2, color=colors[i])
		
	cum = cum / np.max(cum)
	print "{0} at 68% - {1} at 95%".format(B[np.argmin(np.abs(cum/np.max(cum)-0.68))],B[np.argmin(np.abs(cum/np.max(cum)-0.95))])
	B68[2,i] = B[np.argmin(np.abs(cum/np.max(cum)-0.68))]
	B95[2,i] = B[np.argmin(np.abs(cum/np.max(cum)-0.95))]
				
ax[loop].set_ylabel('cdf(B|$\Phi_\mathrm{obs}$)')
ax[loop].set_xlabel('B [G]')
ax[loop].set_xlim((0,300))
	
pl.tight_layout()
pl.savefig("posteriorAnisot.pdf")