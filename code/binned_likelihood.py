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

bins= np.logspace(3,7,25)
bin_centers = (bins[1:]+bins[:-1])/2

def nevents_DMparams(g,mphi,mx):
    nu_weight_astro,nu_weight_atm,mu_weight = get_weights(g,mphi,mx)
    nu_weight_astro_null = 2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**(-2.67)*exposure

    weight_null = np.append(nu_weight_astro_null+nu_weight_atm,mu_weight)
    DM_weight = np.append(nu_weight_astro+nu_weight_atm,mu_weight)

    num_expected_atm=np.sum(nu_weight_atm)
    num_expected_muon=np.sum(mu_weight)
    num_expected_astro=np.sum(nu_weight_astro)
    num_expected=num_expected_atm+num_expected_muon+num_expected_astro

    h_DM,_  = np.histogram(np.append(MC['energy'],muon_mc['energy']),bins=bins,weights=DM_weight)
    h_null,_ = np.histogram(np.append(MC['energy'],muon_mc['energy']),bins=bins,weights=weight_null)
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

def sumTS_null(g,mphi,mx):
    h_null,h_DM = nevents_DMparams(g,mphi,mx)
    # num_expected = nevents_DMparams(g,mphi,mx)
    # TS = TS_(num_observed,num_expected)
    TS, Z = TS_(h_null,h_DM)
    # plt.figure()
    # plt.plot(bin_centers, -log_poisson(h_null,h_null), label='NLL - Null')
    # plt.plot(bin_centers, -log_poisson(h_null,h_DM), label='NLL - DM')
    # plt.plot(bin_centers, TS, label='TS')
    # plt.legend()
    # plt.loglog()
    # title =  'g = {:.0e} GeV, '.format(g/GeV) + r'$m_\phi = $' + '{:.0e} GeV, '.format(mphi/GeV) + r'$m_\chi = $' + '{:.0e} GeV'.format(mx/GeV)
    # plt.title(title)
    # plt.show()
    return np.sum(TS)

def sumTS_DM(g,mphi,mx):
    _,h_DM_true = nevents_DMparams(3e-1,1e7,1e8)
    _,h_DM_hyp = nevents_DMparams(g,mphi,mx)
    TS, Z = TS_(h_DM_true,h_DM_hyp)
    return np.sum(TS)

#ranges of interest
# g  = 1e-9 - 1e4
# mphi = 1e2 - 1e6
# mx = 1e0 - 1e8

g_list = np.logspace(-4,0)
mphi_list = np.logspace(6,11)
mx_list = np.logspace(6,10)

TS_g_null = [sumTS_null(g,1e7,1e8) for g in g_list]
TS_mphi_null = [sumTS_null(3e-1,mphi,1e8) for mphi in mphi_list]
TS_mx_null = [sumTS_null(3e-1,1e7,mx) for mx in mx_list]

TS_g_DM = [sumTS_DM(g,1e7,1e8) for g in g_list]
TS_mphi_DM = [sumTS_DM(3e-1,mphi,1e8) for mphi in mphi_list]
TS_mx_DM = [sumTS_DM(3e-1,1e7,mx) for mx in mx_list]

plt.figure(1)
ax = plt.gca()
plt.plot(g_list/GeV,TS_g_null, label='Null Hypo')
plt.plot(g_list/GeV,TS_g_DM, label='DM Hypo (g = %0.0e eV)'%(3e-1/GeV))
plt.title(r'$m_\phi$ = %0.0e eV'%(1e7/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
plt.ylabel(r'$\sum TS$')
plt.xlabel('g (GeV)')
plt.loglog()
plt.legend()
plt.tick_params(axis='both', which='minor')
plt.tight_layout()
plt.show()

plt.figure(2)
ax = plt.gca()
plt.plot(mphi_list/GeV,TS_mphi_null, label='Null Hypo')
plt.plot(mphi_list/GeV,TS_mphi_DM, label=r'DM Hypo ($m_\phi$ = %0.0e eV)'%(1e7/GeV))
plt.title('g = %0.0e eV'%(3e-1/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
plt.ylabel(r'$\sum TS$')
plt.xlabel(r'$m_\phi$ (GeV)')
plt.loglog()
plt.legend()
plt.tick_params(axis='both', which='minor')
plt.tight_layout()
plt.show()

plt.figure(3)
ax = plt.gca()
plt.plot(mx_list/GeV,TS_mx_null, label='Null Hypo')
plt.plot(mx_list/GeV,TS_mx_DM, label='DM Hypo ($m_\chi$ = %0.0e eV)'%(1e8/GeV))
plt.title('g = %0.0e eV'%(3e-1/GeV)+r', $m_\phi$ = %0.0e eV'%(1e7/GeV))
plt.ylabel(r'$\sum TS$')
plt.xlabel(r'$m_\chi$ (GeV)')
plt.loglog()
plt.legend()
plt.tick_params(axis='both', which='minor')
plt.tight_layout()
plt.show()


# test plotting
# for g in np.logspace(-4,0,5):
#     for mphi in np.logspace(4,11,5):
#         for mx in np.logspace(4,10,5):
#             nu_weight_astro_null = 2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**(-2.67)*exposure
#             nu_weight_astro,_,_ = get_weights(g, mphi, mx)
#             print(np.round(np.sum(nu_weight_astro_null),4))
#             print(np.round(np.sum(nu_weight_astro),4))
#             if (np.round(np.sum(nu_weight_astro),4) < np.round(np.sum(nu_weight_astro_null)-0.1*np.sum(nu_weight_astro_null),4)):
#                 print('Attenuation!')
#                 plot_att_Edist(g, mphi, mx)
