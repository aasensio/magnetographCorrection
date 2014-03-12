import numpy as np
import matplotlib.pyplot as pl
import brewer2mpl
from scipy.integrate import simps
from scipy.stats import linregress

dat = np.load('posterior.npz')

pB, normaliz, B, sigmas, fluxes = dat['arr_0'], dat['arr_1'], dat['arr_2'], dat['arr_3'], dat['arr_4']
labels = [r'2 Mx cm$^{-2}$',r'5 Mx cm$^{-2}$',r'10 Mx cm$^{-2}$',r'15 Mx cm$^{-2}$',r'20 Mx cm$^{-2}$',r'40 Mx cm$^{-2}$',r'80 Mx cm$^{-2}$',r'100 Mx cm$^{-2}$']

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
f, ax = pl.subplots(4, 2, sharex='col', figsize=(10,13), num=1)
ax = ax.flatten()

colors = brewer2mpl.get_map('Dark2', 'qualitative', nFluxes).mpl_colors

loop = 0

for k in range(len(sigmas)):
	for i in range(len(fluxes)):
		posterior = prior * pB[k,i,:]
		normaliz[k,i] = simps(posterior, B)
		ax[loop].plot(B, posterior / normaliz[k,i], label=labels[i], linewidth=2, color=colors[i])
		MMAP[k,i] = B[posterior.argmax()]	
	ax[loop].set_ylabel(r'p(B|$\Phi_\mathrm{obs}$)')
	ax[loop].annotate(r'$\sigma_n$='+"{0:4.1f}".format(sigmas[k])+r' Mx cm$^{-2}$', xy=(0.55, 0.86), xycoords='axes fraction', fontsize=14)
	ax[loop].set_xlim((0,400))
	loop += 1
			
	for i in range(len(fluxes)):
		posterior = prior * pB[k,i,:]
		cum = np.cumsum(posterior)
		ax[loop].plot(B, cum / np.max(cum), label=labels[i], linewidth=2, color=colors[i])
		
		cum = cum / np.max(cum)
		print "{0} at 68% - {1} at 95%".format(B[np.argmin(np.abs(cum/np.max(cum)-0.68))],B[np.argmin(np.abs(cum/np.max(cum)-0.95))])
		B68[k,i] = B[np.argmin(np.abs(cum/np.max(cum)-0.68))]
		B95[k,i] = B[np.argmin(np.abs(cum/np.max(cum)-0.95))]
				
	ax[loop].set_ylabel('cdf(B|$\Phi_\mathrm{obs}$)')
	ax[loop].set_xlim((0,600))
	
	loop += 1
	
ax[-1].set_xlabel('B [G]')
ax[-2].set_xlabel('B [G]')
f.subplots_adjust(hspace=0.2)
pl.setp([a.get_xticklabels() for a in f.axes[:-2]], visible=False)

ax[-1].legend(labelspacing=0.2, prop={'size':12}, loc='lower right')
pl.tight_layout()
pl.savefig("posterior.pdf")

f, ax = pl.subplots(3, 1, sharex=True, figsize=(6,10), num=2)
ax = ax.flatten()
vars = [MMAP, B68, B95]
labelsY = [r'B$_\mathrm{MMAP}$',r'B$_{68}$',r'B$_{95}$']
labels = [r'$\sigma_n$='+"{0:4.1f}".format(sigmas[k])+r' Mx cm$^{-2}$' for k in range(nSigmas)]
colors = brewer2mpl.get_map('Dark2', 'qualitative', nSigmas).mpl_colors
for i in range(3):
	for j in range(len(sigmas)):
		ax[i].plot(fluxes, vars[i][j,:], linewidth=2, color=colors[j], label=labels[j])
	ax[i].set_ylabel(labelsY[i]+ ' [G]')
	if (i == 0):
		slope, intercept, r, p, std = linregress(fluxes, vars[i][0,:])
		ax[i].plot(fluxes, intercept + slope*fluxes, '--', color='k', linewidth=2)
		print "{0} + Phi*{1}".format(intercept,slope)
	else:
		slope, intercept, r, p, std = linregress(np.log(fluxes), np.log(vars[i][0,:]))
		ax[i].plot(fluxes, np.exp(intercept)*fluxes**slope, '--', color='k', linewidth=2)
		print "{0} * Phi**{1}".format(np.exp(intercept),slope)

ax[-1].set_xlabel(r'$\Phi$')
ax[-1].legend(labelspacing=0.2, prop={'size':12}, loc='lower right')
pl.tight_layout()
pl.savefig("calibration.pdf")