import numpy as np

from create_fake_data import *
from plot_fake_data import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('../paper.mplstyle')

data=np.load('../mc/mese_cascades_2010_2016_dnn.npy') # GeV
MC=np.load('../mc/mese_cascades_MC_2013_dnn.npy') # GeV
muon_mc=np.load('../mc/mese_cascades_muongun_dnn.npy') # GeV

column_dens=np.load('../created_files/column_dens.npy')

random_data_ind=np.load('../created_files/random_data_ind_DM.npy')
num_observed = len(random_data_ind)

#Choose the flavor & index you want
flavor = 2  # 1,2,3 = e, mu, tau; negative sign for antiparticles
#gamma = 2.  # Power law index of isotropic flux E^-gamma
logemin = 3 #log GeV
logemax = 7 #log GeV
gamma = 2.67

#2428 days
exposure=2428*24*60*60


# test plotting

# g, mphi, mx = [3e-1,1e7,1e8]
# # g, mphi, mx = [1e0, 1e8,1e9]
# nu_weight_astro,nu_weight_atm,mu_weight = get_weights(g, mphi, mx)
# nu_weight_astro_null= 2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**-(2.67)*exposure
# weight_null = np.append(nu_weight_astro_null+nu_weight_atm,mu_weight)
# DM_weight = np.append(nu_weight_astro+nu_weight_atm,mu_weight)
#
#
# h_DM,_,_ = plt.hist(np.append(MC['energy'],muon_mc['energy']),
#                           bins=np.logspace(2,8,25),weights=DM_weight,histtype="step",label='DM',lw=1)
# h_null,_,_ = plt.hist(np.append(MC['energy'],muon_mc['energy']),
#                             bins=np.logspace(2,8,25),weights=weight_null,histtype="step", label='Null', color='black',lw=1)
# plot_att_Edist(g, mphi, mx)


def nevents_DMparams(g,mphi,mx):
    nu_weight_astro,nu_weight_atm,mu_weight = get_weights(g,mphi,mx)
    nu_weight_astro_null = 2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**(-2.67)*exposure

    weight_null = np.append(nu_weight_astro_null+nu_weight_atm,mu_weight)
    DM_weight = np.append(nu_weight_astro+nu_weight_atm,mu_weight)

    # print('sum null full', np.sum(weight_null))
    # print('sum DM full',np.sum(DM_weight))

    num_expected_atm=np.sum(nu_weight_atm)
    num_expected_muon=np.sum(mu_weight)
    num_expected_astro=np.sum(nu_weight_astro)
    num_expected=num_expected_atm+num_expected_muon+num_expected_astro

    # print('num_expected', num_expected)
    # print('num_obs', num_observed)
    h_DM,_  = np.histogram(np.append(MC['energy'],muon_mc['energy']),bins=np.logspace(3,7,25),weights=DM_weight)
    h_null,_ = np.histogram(np.append(MC['energy'],muon_mc['energy']),bins=np.logspace(3,7,25),weights=weight_null)
    return h_null,h_DM # num_expected


def log_poisson(k,mu):
    '''
    k : n_observed
    mu : num_expected
    '''
    logf = k*np.log(k)-k+np.log(2*np.pi*k)/2 + 1/(12*k) # approximated with stirling series
    logp = k*np.log(mu) - mu - logf
    logp[np.where(np.isnan(logp) == True)] = 0.0 # change NaN to zero
    return logp

def TS_(k,mu):
    '''
    k : n_observed
    mu : num_expected
    returns: TS, Z = sqrt(TS)
    '''
    TS = -2*(log_poisson(k,mu) - log_poisson(k,k))
    return TS, np.sqrt(TS)

def sumTS(g,mphi,mx):
    h_null,h_DM = nevents_DMparams(g,mphi,mx)
    # num_expected = nevents_DMparams(g,mphi,mx)
    # TS = TS_(num_observed,num_expected)
    TS, Z = TS_(h_null,h_DM)
    return np.sum(TS)


g_list = np.logspace(-2,0,17)
print(g_list)
TS_g = [sumTS(g,1e7,1e8) for g in g_list]

mphi_list = np.logspace(6,8,21)
print(mphi_list)
TS_mphi = [sumTS(3e-1,mphi,1e8) for mphi in mphi_list]

mx_list = np.logspace(7,9,21)
print(mx_list)
TS_mx = [sumTS(3e-1,1e7,mx) for mx in mx_list]


plt.figure()
plt.plot(g_list,TS_g)
plt.title(r'$m_\phi = 10^7$ GeV, $m_\chi = 10^8$ GeV')
plt.ylabel(r'$\sum TS$')
plt.xlabel('g')
plt.loglog()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(mphi_list,TS_mphi)
plt.title(r'$g = 3\times 10^{-1}$, $m_\chi = 10^8$ GeV')
plt.ylabel(r'$\sum TS$')
plt.xlabel(r'$m_\phi$ (GeV)')
plt.loglog()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(mx_list,TS_mx)
plt.title(r'$g = 3\times 10^{-1}$, $m_\phi = 10^7$ GeV')
plt.ylabel(r'$\sum TS$')
plt.xlabel(r'$m_\chi$ (GeV)')
plt.loglog()
plt.tight_layout()
plt.show()

# extra

# g, mphi, mx = [3e-1,1e6,1e7]
# h_null,h_DM = nevents_DMparams(g,mphi,mx)
# # lp = log_poisson(h_null,h_DM)
# # TS = TS_(h_null,h_DM)
#
# num_expected = nevents_DMparams(g,mphi,mx)
# TS = TS_(num_observed,num_expected)

# bins = np.logspace(3,7,25)
# bin_centers = bins[:-1] + np.diff(bins)/2

# plt.figure()
# plt.hist(bin_centers,bins=bins,weights=-lp,histtype="step",label='NLL',lw=1)
# plt.hist(bin_centers,bins=bins,weights=TS,histtype="step",label='TS',lw=1)
# plt.ylabel(r'$N_{events}$')
# plt.xlabel(r'Energy (GeV)')
# plt.loglog()
# plt.legend(loc='upper left')
# plt.show()
