import numpy as np
import scipy as sp
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.special as ss
from scipy.integrate import ode
from numpy import linalg as LA

from xs import *
from dxs_scalar import *
from dxs_fermion import *
from dxs_fermscal import *
from dxs_vecferm import *

import NFW
import cascade as cas

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('../paper.mplstyle')

data=np.load('../mc/mese_cascades_2010_2016_dnn.npy')
MC=np.load('../mc/mese_cascades_MC_2013_dnn.npy')
muon_mc=np.load('../mc/mese_cascades_muongun_dnn.npy')

#Choose the flavor & index you want
flavor = 2  # 1,2,3 = e, mu, tau; negative sign for antiparticles
#gamma = 2.  # Power law index of isotropic flux E^-gamma
logemin = 3 #log GeV
logemax = 7 #log GeV
gamma = 2.67


#2428 days
exposure=2428*24*60*60


def interpolation(x,X,Y):
    dX = np.diff(X)
    dY = np.diff(Y, axis=1)
    w = np.clip((x[:,None] - X[:-1])/dX[:], 0, 1)
    y = Y[:, 0] + np.sum(w*dY, axis=1)
    return y


def get_att_value_theta(w, v, ci, energy_nodes, E,phi_in,t):

    logE = np.log10(E)[:len(t)]
    w=np.tile(w,[len(t),1])
    phisol = np.inner(v,ci*np.exp(w.T*t[:len(t)]).T).T*energy_nodes**(-2)/ phi_in #this is the attenuated flux phi/phi_0
    return interpolation(logE, np.log10(energy_nodes), phisol)#np.interp(logE, np.log10(energy_nodes), phisol)


try:
    column_dens=np.load('column_dens.npy')
except:
    ras = np.append(MC['ra'],muon_mc['ra'])
    decs = np.append(MC['dec'],muon_mc['dec'])
    column_dens = NFW.get_t_NFW(ras,decs) * gr * Na /cm**2  # g/ cm^2 -> eV^3
    np.save('column_dens',column_dens)

try:
    random_data_ind=np.load('random_data_ind_DM.npy')
    # random_data_ind=np.load('random_data_ind_null.npy') # lod nulll not with DM parameters
except:
    nu_weight_astro= 2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**-(2.67)*exposure # assuming no attenuation - no DM
    nu_weight_atm=(1.08)*MC['weight_conv']*exposure
    mu_weight=(0.8851)*muon_mc['weight']*exposure
    num_observed = len(data) #np.random.poisson(len(data))
    random_data_ind = np.random.choice(len(MC)+len(muon_mc), num_observed, replace=False,p=np.append(nu_weight_astro+nu_weight_atm,mu_weight)/np.sum(np.append(nu_weight_astro+nu_weight_atm,mu_weight)))
    np.save('random_data_ind_null',random_data_ind)

num_observed = len(random_data_ind)


# g, mphi, mx = [3e-1,1e7,1e8]
# w, v, ci, energy_nodes, phi_0 = cas.get_eigs(g,mphi,mx,'scalar',gamma,logemin,logemax)
#
# MC_len=len(MC)
# break_points=100
# flux_astro=np.ones(MC_len)
# for i in range(break_points):
#     start = int(i*(MC_len/break_points))
#     end = int((i+1)*(MC_len/break_points))
#
#     E = MC['true_energy'][start:end]*GeV # true_energy
#     phi_in = energy_nodes ** (-gamma)
#     t = column_dens[start:end]
#
#     flux_astro[start:end]= get_att_value_theta(w, v, ci, energy_nodes, E, phi_in, t) # true_coords!!!
#
# nu_weight_astro_null = 2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**(-2.67)*exposure
# nu_weight_astro = nu_weight_astro_null* flux_astro
# nu_weight_atm =(1.08)*MC['weight_conv']*exposure
# mu_weight = (0.8851)*muon_mc['weight']*exposure
#
# weight_null = np.append(nu_weight_astro_null+nu_weight_atm,mu_weight)
# DM_weight = np.append(nu_weight_astro+nu_weight_atm,mu_weight)


# plt.figure(dpi=100)
# astro,astrobin=np.histogram(MC['energy'],bins=np.logspace(2,8, 25),weights=nu_weight_astro)
# atm,atmbin=np.histogram(MC['energy'],bins=np.logspace(2,8, 25),weights=nu_weight_atm)
# muon,muonbin=np.histogram(muon_mc['energy'],bins=np.logspace(2,8, 25),weights=mu_weight)
# width = np.diff(astrobin)
# center = (astrobin[:-1] + astrobin[1:]) / 2
# plt.bar(center, astro+atm+muon, align='center', width=width,label='astro',color='#94e3bd')
# plt.bar(center, atm+muon, align='center', width=width,label='atm',color='#85a3b1')
# plt.bar(center, muon, align='center',width=width,label='muon', color='#e0bdb4')
#
# h_DM,_,_ = plt.hist(np.append(MC['energy'],muon_mc['energy']),
#                           bins=np.logspace(2,8,25),weights=DM_weight,histtype="step",label='DM',lw=1)
# h_null,_,_ = plt.hist(np.append(MC['energy'],muon_mc['energy']),
#                             bins=np.logspace(2,8,25),weights=weight_null,histtype="step", label='Null', color='black',lw=1)
# plt.loglog()
# plt.legend()
# plt.ylabel(r'$N_{events}$')
# plt.xlabel(r'Energy (GeV)')
# plt.xlim(1e2, 2e7)
# plt.ylim(1e-1, 2e3)
# plt.show()


# ====================================================
def nevents_DMparams(g,mphi,mx):
    w, v, ci, energy_nodes, phi_0 = cas.get_eigs(g,mphi,mx,'scalar',gamma,logemin,logemax)

    MC_len=len(MC)
    break_points=100
    flux_astro=np.ones(MC_len)
    for i in range(break_points):
        start = int(i*(MC_len/break_points))
        end = int((i+1)*(MC_len/break_points))

        E = MC['true_energy'][start:end]*GeV # true_energy
        phi_in = energy_nodes ** (-gamma)
        t = column_dens[start:end]

        flux_astro[start:end]= get_att_value_theta(w, v, ci, energy_nodes, E, phi_in, t) # true_coords!!!

    nu_weight_astro_null = 2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**(-2.67)*exposure
    nu_weight_astro = nu_weight_astro_null* flux_astro
    nu_weight_atm =(1.08)*MC['weight_conv']*exposure
    mu_weight = (0.8851)*muon_mc['weight']*exposure

    weight_null = np.append(nu_weight_astro_null+nu_weight_atm,mu_weight)
    DM_weight = np.append(nu_weight_astro+nu_weight_atm,mu_weight)


    w, v, ci, energy_nodes, phi_0 = cas.get_eigs(3e-1,1e6,1e7,'scalar',gamma,logemin,logemax)

    MC_len=len(MC)
    break_points=100
    flux_astro=np.ones(MC_len)
    for i in range(break_points):
        start = int(i*(MC_len/break_points))
        end = int((i+1)*(MC_len/break_points))

        E = MC['true_energy'][start:end]*GeV # true_energy
        phi_in = energy_nodes ** (-gamma)
        t = column_dens[start:end]

        flux_astro[start:end]= get_att_value_theta(w, v, ci, energy_nodes, E, phi_in, t)

    weight_DM1 = np.append(nu_weight_astro_null*flux_astro +nu_weight_atm,mu_weight)
    # print('sum null full', np.sum(weight_null))
    # print('sum DM full',np.sum(DM_weight))

    # num_expected_atm=np.sum(nu_weight_atm)
    # num_expected_muon=np.sum(mu_weight)
    # num_expected_astro=np.sum(nu_weight_astro)
    # num_expected=num_expected_atm+num_expected_muon+num_expected_astro

    # print('num_expected', num_expected)
    # print('num_obs', num_observed)

    h_DM,_,_ = plt.hist(np.append(MC['energy'],muon_mc['energy']),bins=np.logspace(2,8,25),weights=DM_weight, histtype='step', label='DM')
    # h_null,_,_ = plt.hist(np.append(MC['energy'],muon_mc['energy']),bins=np.logspace(2,8,25),weights=weight_DM1, histtype='step', label='DM true (g=3e-1,mphi=1e6,mx=1e7)')
    h_null,_,_ = plt.hist(np.append(MC['energy'],muon_mc['energy']),bins=np.logspace(2,8,25),weights=weight_null, histtype='step', label='Null')
    # plt.loglog()
    # plt.xlabel('E (GeV)')
    # title =  'g = {:.0e}, '.format(g) + r'$m_\phi = $' + '{:.0e} GeV, '.format(mphi) + r'$m_\chi = $' + '{:.0e} GeV'.format(mx)
    # plt.title(title)
    # plt.legend()
    # plt.tight_layout()
    # fname = 'g'+str(g)+'_mphi'+str(mphi)+'_mx'+str(mx)
    # plt.savefig(fname+'.png',dpi=200)
    plt.close()
    return h_null,h_DM
    # return num_expected

def log_poisson(k,mu):
    '''
    k : c
    mu : num_expected
    '''
    logf = k*np.log(k)-k+np.log(2*np.pi*k)/2 + 1/(12*k) # approximated with stirling series
    logp = k*np.log(mu) - mu - logf
    logp[np.where(np.isnan(logp) == True)] = 0.0 # change NaN to zero
    return logp

def TS_(k,mu):
    TS = -2*(log_poisson(k,mu) - log_poisson(k,k))
    return TS

def sumTS(g,mphi,mx):
    h_null,h_DM = nevents_DMparams(g,mphi,mx)
    # num_expected = nevents_DMparams(g,mphi,mx)
    # TS = TS_(num_observed,num_expected)
    TS = TS_(h_null,h_DM)
    return np.sum(TS)

def Z(k,mu):
    return np.sqrt(TS_(k,mu))


bins = np.logspace(2,8,25)
bin_centers = bins[:-1] + np.diff(bins)/2

# g, mphi, mx = [3e-1,1e6,1e7]
# h_null,h_DM = nevents_DMparams(g,mphi,mx)
# # lp = log_poisson(h_null,h_DM)
# # TS = TS_(h_null,h_DM)
#
# num_expected = nevents_DMparams(g,mphi,mx)
# TS = TS_(num_observed,num_expected)





g_list = np.logspace(-2,0,17)
print(g_list)
TS_g = [sumTS(g,1e7,1e8) for g in g_list]

mphi_list = np.logspace(6,8,21)
print(mphi_list)
TS_mphi = [sumTS(3e-1,mphi,1e8) for mphi in mphi_list]

mx_list = np.logspace(7,9,21)
print(mx_list)
TS_mx = [sumTS(3e-1,1e7,mx) for mx in mx_list]

# plt.figure()
# plt.hist(bin_centers,bins=bins,weights=-lp,histtype="step",label='NLL',lw=1)
# plt.hist(bin_centers,bins=bins,weights=TS,histtype="step",label='TS',lw=1)
# plt.ylabel(r'$N_{events}$')
# plt.xlabel(r'Energy (GeV)')
# plt.loglog()
# plt.legend(loc='upper left')
# plt.show()

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
