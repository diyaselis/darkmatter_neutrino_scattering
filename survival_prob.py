import numpy as np
import matplotlib.pyplot as plt
import cascade as cas
from xs import *

data=np.load('../mc/mese_cascades_2010_2016_dnn.npy')
MC=np.load('../mc/mese_cascades_MC_2013_dnn.npy')
muon_mc=np.load('../mc/mese_cascades_muongun_dnn.npy')

column_dens=np.load('column_dens.npy')

def interpolation(x,X,Y):
    dX = np.diff(X)
    dY = np.diff(Y, axis=1)
    w = np.clip((x[:,None] - X[:-1])/dX[:], 0, 1)
    # print(dX,dY,w)
    y = Y[:, 0] + np.sum(w*dY, axis=1)
    return y


def get_att_value_theta(w, v, ci, energy_nodes, E,phi_in,t):
    # print('w',w[0],'v',v[0],'ci',ci[0])
    # print('phi_in',phi_in)
    logE = np.log10(E)[:len(t)]
    w=np.tile(w,[len(t),1])
    # print(np.exp(w.T*t[:len(t)]))
    phisol = np.inner(v,ci*np.exp(w.T*t[:len(t)]).T).T*energy_nodes**(-2)/ phi_in #this is the attenuated flux phi/phi_0
    # print('phisol',phisol[:10])
    interp = interpolation(logE, np.log10(energy_nodes), phisol)#np.interp(logE, np.log10(energy_nodes), phisol)
    return interp

#Choose the flavor & index you want
flavor = 2  # 1,2,3 = e, mu, tau; negative sign for antiparticles
#gamma = 2.  # Power law index of isotropic flux E^-gamma
logemin = 3 #log GeV
logemax = 7 #log GeV
gamma = 2.67


#2428 days
exposure=2428*24*60*60



g,mphi,mx = [1e0, 1e-2,1e0]
g,mphi,mx = [3e-1, 1e7,1e8]
w, v, ci, energy_nodes, phi_0 = cas.get_eigs(g,mphi,mx,'scalar',gamma,logemin,logemax)
# print('phi_in',energy_nodes ** (-gamma))
# t = column_dens
# w=np.tile(w,[len(t),1])
# print(np.inner(v,ci*np.exp(w.T*t[:len(t)]).T).T)

MC_len=len(MC)
break_points=100
flux_astro=np.zeros(MC_len)
for i in range(break_points):
    start = int(i*(MC_len/break_points))
    end = int((i+1)*(MC_len/break_points))

    E = MC['true_energy'][start:end]*GeV # true_energy
    phi_in = energy_nodes ** (-gamma)
    t = column_dens[start:end]
    # print(E[:2], phi_in[:2], t[:2])
    flux_astro[start:end]= get_att_value_theta(w, v, ci, energy_nodes, E, phi_in, t) # true_coords!!!
sort = np.argsort(MC['true_energy'])
spline = interpolate.UnivariateSpline(np.log10(MC['true_energy'][sort]), flux_astro[sort], k=2, s=1e-10)
plt.plot(energy_nodes,spline(np.log10(energy_nodes)))
plt.semilogx()
plt.show()
