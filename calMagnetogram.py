import matplotlib.pyplot as pl
import numpy as np
import scipy.special as sp
import scipy.stats
import matplotlib.cm as cm
from scipy.integrate import simps, quad
import brewer2mpl
import pdb

def integrand(mu, Phi, B, sigma):
	cte = np.sqrt(np.pi/2.0) * sigma
	if (mu == 0):
		out = np.exp(-0.5*Phi**2/sigma**2) / cte
	else:
		out = (sp.erf(Phi/(np.sqrt(2.0)*sigma)) + sp.erf((B*mu-Phi) / (np.sqrt(2.0)*sigma))) / (B*mu)	
	return out

nPoints = 1000
B = 10.0**np.linspace(-3,3.5,nPoints)
sigmas = [0.1, 1.0, 5.0, 10.0, 20.0]
fluxes = [2.0,5.0,10.0,15.0,20.0,40.0,80.0,100.0]
MMAP = np.zeros(len(fluxes))
labels = [r'2 Mx cm$^{-2}$',r'5 Mx cm$^{-2}$',r'10 Mx cm$^{-2}$',r'15 Mx cm$^{-2}$',r'20 Mx cm$^{-2}$',r'40 Mx cm$^{-2}$',r'80 Mx cm$^{-2}$',r'100 Mx cm$^{-2}$']

nSigmas = len(sigmas)
nFluxes = len(fluxes)
pB = np.zeros((nSigmas,nFluxes,nPoints))
normaliz = np.zeros((nSigmas,nFluxes))

colors = brewer2mpl.get_map('Dark2', 'qualitative', nFluxes).mpl_colors

f = pl.figure(num=1)
pl.clf()
loop = 1
for k in range(len(sigmas)):
	for i in range(len(fluxes)):
		print sigmas[k], fluxes[i]
		Phi = fluxes[i] #+ sigmas[k] * np.random.randn()

		for j in range(len(B)):
			args = (Phi, B[j], sigmas[k])
			pB[k,i,j], err = quad(integrand, -1.0, 1.0, args=args ,points=[0.0])
		
# Compute normalization
		normaliz[k,i] = simps(pB[k,i,:], B)
		

	
	#MMAP[i] = B[pB.argmax()]

np.savez('posterior.npz',pB,normaliz,B,sigmas,fluxes)