import numpy as np
import cascade as cas
from xs import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('../paper.mplstyle')

data=np.load('../mc/mese_cascades_2010_2016_dnn.npy')
MC=np.load('../mc/mese_cascades_MC_2013_dnn.npy')
muon_mc=np.load('../mc/mese_cascades_muongun_dnn.npy')

column_dens=np.load('../created_files/column_dens.npy')

def interpolation(x,X,Y):
    dX = np.diff(X)
    dY = np.diff(Y, axis=1)
    w = np.clip((x[:,None] - X[:-1])/dX[:], 0, 1)
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

# ===================== functions =========================
def get_weights(g, mphi, mx):
    w, v, ci, energy_nodes, phi_0 = cas.get_eigs(g,mphi,mx,'scalar',gamma,logemin,logemax)

    MC_len=len(MC)
    break_points=100
    flux_astro=np.ones(MC_len)
    if g != 0.:
        for i in range(break_points):
            start = int(i*(MC_len/break_points))
            end = int((i+1)*(MC_len/break_points))

            E = MC['true_energy'][start:end]*GeV # true_energy
            phi_in = energy_nodes ** (-gamma)
            t = column_dens[start:end]
            flux_astro[start:end]= get_att_value_theta(w, v, ci, energy_nodes, E, phi_in, t) # true_coords!!!
    # print(flux_astro)
    nu_weight_astro=2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**(-2.67)*exposure*flux_astro
    # nu_weight_E2=MC['oneweight']*(MC['true_energy'])**-2*exposure
    nu_weight_atm=1.08*MC['weight_conv']*exposure #phi_atm*MC['weight_conv']*exposure*(MC['true_energy']/2020)**(-dgamma)
    mu_weight=0.8851*muon_mc['weight']*exposure
    return nu_weight_astro,nu_weight_atm,mu_weight


try:
    random_data_ind=np.load('../created_files/random_data_ind_DM.npy')
except:
    g, mphi, mx = [3e-1,1e7,1e8]
    nu_weight_astro,nu_weight_atm,mu_weight = get_weights(g, mphi, mx)
    # random_data_ind_astro=np.random.choice(len(MC),int(np.sum(nu_weight_astro)), replace=False,p=nu_weight_astro/np.sum(nu_weight_astro))
    # random_data_ind_atm=np.random.choice(len(MC),int(np.sum(nu_weight_atm)), replace=False,p=nu_weight_atm/np.sum(nu_weight_atm))
    # random_data_ind_muon=np.random.choice(len(muon_mc),int(np.sum(mu_weight)), replace=True,p=mu_weight/np.sum(mu_weight))+len(MC)
    # random_data_ind=np.append(np.append(random_data_ind_atm,random_data_ind_muon),random_data_ind_astro)
    num_observed = len(data) #np.random.poisson(len(data))
    random_data_ind = np.random.choice(len(MC)+len(muon_mc), num_observed, replace=False,p=np.append(nu_weight_astro+nu_weight_atm,mu_weight)/np.sum(np.append(nu_weight_astro+nu_weight_atm,mu_weight)))
    np.save('../created_files/random_data_ind_DM.npy',random_data_ind)

# print(len(random_data_ind))
# print(random_data_ind[:10])
