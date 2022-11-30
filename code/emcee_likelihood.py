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
    column_dens=np.load('../created_files/column_dens.npy')
except:
    ras = np.append(MC['ra'],muon_mc['ra'])
    decs = np.append(MC['dec'],muon_mc['dec'])
    column_dens = NFW.get_t_NFW(ras,decs) * gr * Na /cm**2  # g/ cm^2 -> eV^3
    np.save('../created_files/column_dens',column_dens)


try:
    random_data_ind=np.load('../created_files/random_data_ind_DM_null.npy') # lod nulll not with DM parameters
except:
    nu_weight_astro= 2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**-(2.67)*exposure # assuming no attenuation - no DM
    nu_weight_atm=(1.08)*MC['weight_conv']*exposure
    mu_weight=(0.8851)*muon_mc['weight']*exposure
    num_observed = len(data) #np.random.poisson(len(data))
    random_data_ind = np.random.choice(len(MC)+len(muon_mc), num_observed, replajvub ce=False,p=np.append(nu_weight_astro+nu_weight_atm,mu_weight)/np.sum(np.append(nu_weight_astro+nu_weight_atm,mu_weight)))
    np.save('../created_files/random_data_ind_null',random_data_ind)

num_observed = len(random_data_ind)

class prior:
    gamma_low = -3.0
    gamma_hi = -1.0
    phi_astro_low = 0.0
    phi_astro_hi = 10.0
    phi_atm_low = 0.0
    phi_atm_hi = 10.0
    phi_muon_low = 0.0
    phi_muon_hi = 10.0
    dgamma_low = -1.0
    dgamma_hi = 1.0
    g_low = -5.0
    g_hi = 3.0
    mx_low = -4.0
    mx_hi = 9.0
    mphi_low = -3.0
    mphi_hi = 8.0

    def __init__(self, theta):
        gamma,phi_astro,phi_atm,phi_muon,dgamma = [-2.0, 1.0,1.0,1.0,-0.1]
        g,mphi,mx = theta

        if self.gamma_low < gamma < self.gamma_hi and self.phi_astro_low < phi_astro < self.phi_astro_hi and self.phi_atm_low < phi_atm < self.phi_atm_hi and self.phi_muon_low < phi_muon < self.phi_muon_hi and self.dgamma_low < dgamma < self.dgamma_hi and self.g_low<g<self.g_hi and self.mx_low<mx<self.mx_hi and self.mphi_low < mphi <self.mphi_hi:
            self.log =  -np.log(self.gamma_hi-self.gamma_low)-np.log(self.phi_astro_hi-self.phi_astro_low)-np.log(self.phi_atm_hi-self.phi_atm_low)-np.log(self.phi_muon_hi-self.phi_muon_low)-np.log(self.dgamma_hi-self.dgamma_low)-np.log(10**self.g_hi-10**self.g_low)-np.log(10**self.mx_hi-10**self.mx_low)-np.log(10**self.mphi_hi-10**self.mphi_low)
        else:
            self.log = -np.inf


def loglikelihood(theta):
    gamma,phi_astro,phi_atm,phi_muon,dgamma = [-2.0, 1.0,1.0,1.0,-0.1]
    log_g,log_mphi,log_mx = theta
    g, mphi,mx = [10**log_g,10**log_mphi,10**log_mx]
    # get the eigenvalues for the specific DM model
    w, v, ci, energy_nodes, phi_0 = cas.get_eigs(g, mphi,mx,'scalar',gamma,logemin,logemax) # scanning in log space

    MC_len=len(MC)
    break_points=100
    flux_astro=np.zeros(MC_len)
    for i in range(break_points):
        start = int(i*(MC_len/break_points))
        end = int((i+1)*(MC_len/break_points))

        E = MC['true_energy'][start:end]*GeV # true_energy
        phi_in = energy_nodes ** (-gamma)
        t = column_dens[start:end]
        flux_astro[start:end]= get_att_value_theta(w, v, ci, energy_nodes, E, phi_in, t) # true_coords!!!

    nu_weight_astro=phi_astro * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**(gamma)*exposure*flux_astro+1e-14#2.1464
    nu_weight_atm=phi_atm*MC['weight_conv']*exposure*(MC['true_energy']/2020)**(-dgamma)+1e-14 #1.08
    mu_weight=phi_muon*muon_mc['weight']*exposure #0.8851

    num_expected_atm=np.sum(nu_weight_atm)
    num_expected_muon=np.sum(mu_weight)
    num_expected_astro=np.sum(nu_weight_astro)
    num_expected=num_expected_atm+num_expected_muon+num_expected_astro

    log_factorial = lambda k: k*np.log(k)-k+np.log(2*np.pi*k)/2 + 1/(12*k)
    #     print('num_expected', num_expected, 'num_observed',num_observed, 'log_factorial',log_factorial)
    log_poisson = -num_expected+num_observed*np.log(num_expected)-log_factorial(num_observed)

    def kde(parameter,source):
        num_bins=15
        if source =='atm':
            mc_data=MC[parameter]
            weight=nu_weight_atm
        if source =='astro':
            mc_data=MC[parameter]
            weight=nu_weight_astro
        if source == 'muon':
            mc_data=muon_mc[parameter]
            weight=mu_weight
            num_bins=3

        if parameter=='energy':
            sample_data=np.log10(np.append(MC[parameter],muon_mc[parameter])[random_data_ind])
            hist,histbins=np.histogram(np.log10(mc_data),bins=np.log10(np.logspace(3,7, num_bins+2)),weights=weight,density=True)

        if parameter=='dec':
            sample_data=np.sin(np.append(MC[parameter],muon_mc[parameter])[random_data_ind])
            hist,histbins=np.histogram(np.sin(mc_data),bins=np.linspace(-1,1, num_bins+1),weights=weight,density=True)

        if parameter=='ra':
            sample_data=np.append(MC[parameter],muon_mc[parameter])[random_data_ind]
            hist,histbins=np.histogram(mc_data,bins=np.linspace(0, 2*np.pi, num_bins+1),weights=weight,density=True)

        ind=np.searchsorted(histbins,sample_data)
        ind[ind==17]=16
        ind[ind==0]=1
        pdf=hist[ind-1]

        return pdf

    f=np.zeros(num_observed)
    kde_energy_astro=kde('energy','astro')
    kde_energy_atm=kde('energy','atm')
    kde_energy_muon=kde('energy','muon')
    kde_dec_astro=kde('dec','astro')
    kde_dec_atm=kde('dec','atm')
    kde_dec_muon=kde('dec','muon')
    kde_ra_astro=kde('ra','astro')
    kde_ra_atm=kde('ra','atm')
    kde_ra_muon=kde('ra','muon')

    for i in range(num_observed):
        f[i]=np.log(num_expected_astro*(kde_ra_astro[i]*kde_dec_astro[i]*kde_energy_astro[i])+num_expected_atm*(kde_ra_atm[i]*kde_dec_atm[i]*kde_energy_atm[i])+num_expected_muon*(kde_ra_muon[i]*kde_dec_muon[i]*kde_energy_muon[i]))-np.log(num_expected)
    likelihood=log_poisson+np.sum(f)

    if not np.isfinite(likelihood):
        return -np.inf
    return likelihood


class prior:
    gamma_low = -3.0
    gamma_hi = -1.0
    phi_astro_low = 0.0
    phi_astro_hi = 10.0
    phi_atm_low = 0.0
    phi_atm_hi = 10.0
    phi_muon_low = 0.0
    phi_muon_hi = 10.0
    dgamma_low = -1.0
    dgamma_hi = 1.0
    g_low = -5.0
    g_hi = 3.0
    mx_low = -4.0
    mx_hi = 9.0
    mphi_low = -3.0
    mphi_hi = 8.0

    def __init__(self, theta):
        gamma,phi_astro,phi_atm,phi_muon,dgamma = [-2.0, 1.0,1.0,1.0,0.0]
        g,mphi,mx = theta

        if self.gamma_low < gamma < self.gamma_hi and self.phi_astro_low < phi_astro < self.phi_astro_hi and self.phi_atm_low < phi_atm < self.phi_atm_hi and self.phi_muon_low < phi_muon < self.phi_muon_hi and self.dgamma_low < dgamma < self.dgamma_hi and self.g_low<g<self.g_hi and self.mx_low<mx<self.mx_hi and self.mphi_low < mphi <self.mphi_hi:
            self.log =  -np.log(self.gamma_hi-self.gamma_low)-np.log(self.phi_astro_hi-self.phi_astro_low)-np.log(self.phi_atm_hi-self.phi_atm_low)-np.log(self.phi_muon_hi-self.phi_muon_low)-np.log(self.dgamma_hi-self.dgamma_low)-np.log(10**self.g_hi-10**self.g_low)-np.log(10**self.mx_hi-10**self.mx_low)-np.log(10**self.mphi_hi-10**self.mphi_low)
        else:
            self.log = -np.inf


def log_probability(theta):
    lp  = prior(theta).log
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood(theta)

theta = [-1,7,8]
print(log_probability(theta))
