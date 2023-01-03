import numpy as np
from xs import *
import cascade as cas
from scipy.stats import poisson

data=np.load('../mc/mese_cascades_2010_2016_dnn.npy') # GeV
MC=np.load('../mc/mese_cascades_MC_2013_dnn.npy') # GeV
muon_MC=np.load('../mc/mese_cascades_muongun_dnn.npy') # GeV

column_dens=np.load('../created_files/column_dens.npy')
random_data_ind=np.load('../created_files/random_data_ind_DM.npy')
num_observed = len(random_data_ind)

#Choose the flavor & index you want
gamma = 2.67
#gamma = 2.  # Power law index of isotropic flux E^-gamma
logemin = 3 #log GeV
logemax = 7 #log GeV

# 10 yrs - 2428 days
yr2sec = 31556926 # s
# exposure=2428*24*60*60 # s
exposure = 10 * yr2sec

bins = np.logspace(3,7)
bin_centers = (bins[1:]+bins[:-1])/2

# helper functions
def interpolation(x,X,Y):
    dX = np.diff(X)
    dY = np.diff(Y, axis=1)
    w = np.clip((x[:,None] - X[:-1])/dX[:], 0, 1)
    y = Y[:, 0] + np.sum(w*dY, axis=1)
    return y

def get_att_value_theta(w, v, ci, energy_nodes, E,phi_in,t):
    logE = np.log10(E)[:len(t)]
    w=np.tile(w,[len(t),1])
    phisol = np.inner(v,ci*np.exp(w.T*t[:len(t)]).T).T*energy_nodes**(-2)/ phi_in #attenuated flux phi/phi_0
    interp = interpolation(logE, np.log10(energy_nodes), phisol)
    return interp

class BinnedLikelihoodFunction:

    def __init__(self,g,mphi,mx,interaction): # (data,MC,muon_MC,binning):
        self.g = g
        self.mphi = mphi
        self.mx = mx
        self.interaction = interaction
        weight_null,weight_DM = self.GetWeights(self.g,self.mphi,self.mx,self.interaction)
        self.E_null,_ = np.histogram(np.append(MC['energy'],muon_MC['energy']),bins=bins,weights=weight_null)
        self.E_DM,_  = np.histogram(np.append(MC['energy'],muon_MC['energy']),bins=bins,weights=weight_DM)
        self.RA_null,_ = np.histogram(np.append(MC['ra'],muon_MC['ra']),bins=np.linspace(0, 2*np.pi),weights=weight_null)
        self.RA_DM,_  = np.histogram(np.append(MC['ra'],muon_MC['ra']),bins=np.linspace(0, 2*np.pi),weights=weight_DM)
        self.dec_null,_ = np.histogram(np.append(np.sin(MC['dec']),np.sin(muon_MC['energy'])),bins=np.linspace(-1,1),weights=weight_null)
        self.dec_DM,_  = np.histogram(np.append(np.sin(MC['dec']),np.sin(muon_MC['energy'])),bins=np.linspace(-1,1),weights=weight_DM)


    def AttenuatedFlux(self,g,mphi,mx,interaction):
        w, v, ci, energy_nodes, phi_0 = cas.get_eigs(g,mphi,mx,interaction,gamma,logemin,logemax)

        MC_len=len(MC)
        break_points=100
        flux_astro=np.ones(MC_len)
        for i in range(break_points):
            start = int(i*(MC_len/break_points))
            end = int((i+1)*(MC_len/break_points))
            E = MC['true_energy'][start:end]*GeV
            phi_in = energy_nodes ** (-gamma)
            t = column_dens[start:end]
            flux_astro[start:end]= get_att_value_theta(w, v, ci, energy_nodes, E, phi_in, t)
        return flux_astro

    def GetWeights(self,g,mphi,mx,interaction):
        nu_weight_astro_null = 2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**(-2.67)*exposure
        nu_weight_astro = nu_weight_astro_null * self.AttenuatedFlux(g,mphi,mx,interaction)
        nu_weight_atm=1.08*MC['weight_conv']*exposure
        mu_weight=0.8851*muon_MC['weight']*exposure
        weight_null = np.append(nu_weight_astro_null+nu_weight_atm,mu_weight)
        weight_DM = np.append(nu_weight_astro+nu_weight_atm,mu_weight)
        return weight_null,weight_DM

    def LogPoisson(self,k,mu):
        logf = k*np.log(k)-k+np.log(2*np.pi*k)/2 + 1/(12*k) # approximated with stirling series
        logp = k*np.log(mu) - mu - logf
        logp[np.where(np.isnan(logp) == True)] = 0.0 # change NaN to zero
        # logp = poisson.logpmf(np.round(k),mu)
        return logp

    def TestStatistics(self,k,mu): # k,mu : n_observed, num_expected
        return -2*(self.LogPoisson(k,mu) - self.LogPoisson(k,k))

    def __call__(self,*params): #*params
        if params == (): # compare to null hypo
            TS_E_null = np.sum(self.TestStatistics(self.E_null,self.E_DM))
            TS_RA_null = np.sum(self.TestStatistics(self.RA_null,self.RA_DM))
            TS_dec_null = np.sum(self.TestStatistics(self.dec_null,self.dec_DM))
            return TS_E_null,TS_RA_null,TS_dec_null
        else: # compare to DM parameters
            _,weight_DM_true = self.GetWeights(params[0], params[1], params[2],params[3])
            E_DM_true,_  = np.histogram(np.append(MC['energy'],muon_MC['energy']),bins=bins,weights=weight_DM_true)
            RA_DM_true,_  = np.histogram(np.append(MC['ra'],muon_MC['ra']),bins=np.linspace(0, 2*np.pi),weights=weight_DM_true)
            dec_DM_true,_  = np.histogram(np.append(np.sin(MC['energy']),np.sin(muon_MC['energy'])),bins=np.linspace(-1,1),weights=weight_DM_true)
            TS_E_DM = np.sum(self.TestStatistics(E_DM_true,self.E_DM))
            TS_RA_DM = np.sum(self.TestStatistics(RA_DM_true,self.RA_DM))
            TS_dec_DM = np.sum(self.TestStatistics(dec_DM_true,self.dec_DM))
            return TS_E_DM,TS_RA_DM,TS_dec_DM
