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

#2428 days
exposure=2428*24*60*60  # s

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


class EmceeUnbinnedLikelihoodFunction:

    def __init__(self,theta,interaction='scalar'): # (data,MC,muon_MC,binning):
        self.theta = theta
        # self.g = theta[0]
        # self.mphi = theta[1]
        # self.mx = theta[2]
        self.interaction = interaction
        # bounds for LogPrior
        self.gamma_low = -3.0
        self.gamma_hi = -1.0
        self.phi_astro_low = 0.0
        self.phi_astro_hi = 10.0
        self.phi_atm_low = 0.0
        self.phi_atm_hi = 10.0
        self.phi_muon_low = 0.0
        self.phi_muon_hi = 10.0
        self.dgamma_low = -1.0
        self.dgamma_hi = 1.0
        self.g_low = -5.0
        self.g_hi = 3.0
        self.mx_low = -3.0
        self.mx_hi = 9.0
        self.mphi_low = -3.0
        self.mphi_hi = 9.0

    def __call__(self): #*params
        return None

    def LogPrior(self):
        gamma,phi_astro,phi_atm,phi_muon,dgamma = [-2.0, 1.0,1.0,1.0,-0.1]
        g,mphi,mx = self.theta

        if self.gamma_low < gamma < self.gamma_hi and self.phi_astro_low < phi_astro < self.phi_astro_hi and self.phi_atm_low < phi_atm < self.phi_atm_hi and self.phi_muon_low < phi_muon < self.phi_muon_hi and self.dgamma_low < dgamma < self.dgamma_hi and self.g_low<g<self.g_hi and self.mx_low<mx<self.mx_hi and self.mphi_low < mphi <self.mphi_hi:
            logprior =  -np.log(self.gamma_hi-self.gamma_low)-np.log(self.phi_astro_hi-self.phi_astro_low)-np.log(self.phi_atm_hi-self.phi_atm_low)-np.log(self.phi_muon_hi-self.phi_muon_low)-np.log(self.dgamma_hi-self.dgamma_low)-np.log(10**self.g_hi-10**self.g_low)-np.log(10**self.mx_hi-10**self.mx_low)-np.log(10**self.mphi_hi-10**self.mphi_low)
        else:
            logprior = -np.inf
        return logprior

    def AttenuatedFlux(self,theta):
        g,mphi,mx = theta
        w, v, ci, energy_nodes, phi_0 = cas.get_eigs(10**g,10**mphi,10**mx,self.interaction,gamma,logemin,logemax)

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

    def GetWeights(self,theta,sum_weights=False):
        nu_weight_astro_null = 2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**(-2.67)*exposure
        nu_weight_astro = nu_weight_astro_null * self.AttenuatedFlux(theta)
        nu_weight_atm=1.08*MC['weight_conv']*exposure
        mu_weight=0.8851*muon_MC['weight']*exposure
        if sum_weights == True:
            # weight_null = np.append(nu_weight_astro_null+nu_weight_atm,mu_weight)
            weight_DM = np.append(nu_weight_astro+nu_weight_atm,mu_weight)
            num_expected=np.sum(weight_DM)
            return num_expected
        return nu_weight_astro,nu_weight_atm,mu_weight

    def LogPoisson(self,k,mu): # k,mu : num_observed, num_expected
        # logf = k*np.log(k)-k+np.log(2*np.pi*k)/2 + 1/(12*k) # approximated with stirling series
        # logp = k*np.log(mu) - mu - logf
        # logp[np.where(np.isnan(logp) == True)] = 0.0 # change NaN to zero
        logp = poisson.logpmf(k,mu)
        return logp

    def TestStatistics(self,*params):
        if params == (): # compare tonull hypo
            num_expected = self.GetWeights(self.theta,sum_weights=True)
            TS = -2*(self.LogPoisson(num_observed, num_expected) - self.LogPoisson(num_observed,num_observed))
            return TS
        else: # compare to DM parameters
            num_expected_DMtrue = self.GetWeights(params,sum_weights=True)
            num_expected = self.GetWeights(self.theta,sum_weights=True)
            TS = -2*(self.LogPoisson(np.round(num_expected_DMtrue), num_expected) - self.LogPoisson(np.round(num_expected_DMtrue),num_expected_DMtrue))
            return TS

    def LogProbability(self):
        lp  = self.LogPrior()
        if not np.isfinite(lp):
            return -np.inf
        nu_weight_astro,nu_weight_atm,mu_weight = self.GetWeights(self.theta)
        num_expected_atm=np.sum(nu_weight_atm)
        num_expected_muon=np.sum(mu_weight)
        num_expected_astro=np.sum(nu_weight_astro)
        num_expected=num_expected_atm+num_expected_muon+num_expected_astro
        
        def kde(parameter,source):
            num_bins=15
            if source =='atm':
                mc_data=MC[parameter]
                weight=nu_weight_atm
            if source =='astro':
                mc_data=MC[parameter]
                weight=nu_weight_astro
            if source == 'muon':
                mc_data=muon_MC[parameter]
                weight=mu_weight
                num_bins=3

            if parameter=='energy':
                sample_data=np.log10(np.append(MC[parameter],muon_MC[parameter])[random_data_ind])
                hist,histbins=np.histogram(np.log10(mc_data),bins=np.log10(np.logspace(3,7, num_bins+2)),weights=weight,density=True)

            if parameter=='dec':
                sample_data=np.sin(np.append(MC[parameter],muon_MC[parameter])[random_data_ind])
                hist,histbins=np.histogram(np.sin(mc_data),bins=np.linspace(-1,1, num_bins+1),weights=weight,density=True)

            if parameter=='ra':
                sample_data=np.append(MC[parameter],muon_MC[parameter])[random_data_ind]
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

        likelihood=self.LogPoisson(num_observed, num_expected)+np.sum(f)
        if not np.isfinite(likelihood):
            return -np.inf
        return lp + likelihood
