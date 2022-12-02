import numpy as np
from xs import *
import cascade as cas

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('../paper.mplstyle')

data=np.load('../mc/mese_cascades_2010_2016_dnn.npy') # GeV
MC=np.load('../mc/mese_cascades_MC_2013_dnn.npy') # GeV
muon_MC=np.load('../mc/mese_cascades_muongun_dnn.npy') # GeV

column_dens=np.load('../created_files/column_dens.npy')
random_data_ind=np.load('../created_files/random_data_ind_DM.npy')
# num_observed = len(random_data_ind)

#Choose the flavor & index you want
flavor = 2  # 1,2,3 = e, mu, tau; negative sign for antiparticles
#gamma = 2.  # Power law index of isotropic flux E^-gamma
logemin = 3 #log GeV
logemax = 7 #log GeV
gamma = 2.67

#2428 days
exposure=2428*24*60*60

bins = np.logspace(3,7,25)
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
        self.h_null,_ = np.histogram(np.append(MC['energy'],muon_MC['energy']),bins=bins,weights=weight_null)
        self.h_DM,_  = np.histogram(np.append(MC['energy'],muon_MC['energy']),bins=bins,weights=weight_DM)

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
        '''
        k,mu : n_observed, num_expected
        '''
        logf = k*np.log(k)-k+np.log(2*np.pi*k)/2 + 1/(12*k) # approximated with stirling series
        logp = k*np.log(mu) - mu - logf
        logp[np.where(np.isnan(logp) == True)] = 0.0 # change NaN to zero
        return logp

    def TestStatistics(self,k,mu):
        '''
        k,mu : n_observed, num_expected
        '''
        TS = -2*(self.LogPoisson(k,mu) - self.LogPoisson(k,k))
        return TS

    def __call__(self,*params): #*params
        TS_null = np.sum(self.TestStatistics(self.h_null,self.h_DM))
        # compare to DM parameters
        if params == ():
            return TS_null
        else:
            _,weight_DM_true = self.GetWeights(params[0], params[1], params[2],params[3])
            self.h_DM_true,_  = np.histogram(np.append(MC['energy'],muon_MC['energy']),bins=bins,weights=weight_DM_true)
            TS_DM = np.sum(self.TestStatistics(self.h_DM_true,self.h_DM))
            return TS_DM

# class testing
# g,mphi,mx = [3e-1,1e7,1e8]
# lfn_scalar = BinnedLikelihoodFunction(g,mphi,mx,'scalar')
# ts_null_scalar = lfn_scalar()
# print(ts_null_scalar)

# TS scan over one DM parameter
# g_list = np.logspace(-4,0)
# mphi_list = np.logspace(6,11)
# mx_list = np.logspace(6,10)
#
# TS_g_null = [BinnedLikelihoodFunction(g,1e7,1e8,'scalar')() for g in g_list]
# TS_mphi_null = [BinnedLikelihoodFunction(3e-1,mphi,1e8,'scalar')() for mphi in mphi_list]
# TS_mx_null = [BinnedLikelihoodFunction(3e-1,1e7,mx,'scalar')() for mx in mx_list]
#
# TS_g_DM = [BinnedLikelihoodFunction(g,1e7,1e8,'scalar')(3e-1,1e7,1e8,'scalar') for g in g_list]
# TS_mphi_DM = [BinnedLikelihoodFunction(3e-1,mphi,1e8,'scalar')(3e-1,1e7,1e8,'scalar') for mphi in mphi_list]
# TS_mx_DM = [BinnedLikelihoodFunction(3e-1,1e7,mx,'scalar')(3e-1,1e7,1e8,'scalar') for mx in mx_list]
#
# plt.figure(1)
# ax = plt.gca()
# plt.plot(g_list/GeV,TS_g_null, label='Null Hypo')
# plt.plot(g_list/GeV,TS_g_DM, label='DM Hypo (g = %0.0e eV)'%(3e-1/GeV))
# plt.title(r'$m_\phi$ = %0.0e eV'%(1e7/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
# plt.ylabel(r'$\sum TS$')
# plt.xlabel('g (GeV)')
# plt.loglog()
# plt.legend()
# plt.tick_params(axis='both', which='minor')
# plt.tight_layout()
# plt.show()
#
# plt.figure(2)
# ax = plt.gca()
# plt.plot(mphi_list/GeV,TS_mphi_null, label='Null Hypo')
# plt.plot(mphi_list/GeV,TS_mphi_DM, label=r'DM Hypo ($m_\phi$ = %0.0e eV)'%(1e7/GeV))
# plt.title('g = %0.0e eV'%(3e-1/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
# plt.ylabel(r'$\sum TS$')
# plt.xlabel(r'$m_\phi$ (GeV)')
# plt.loglog()
# plt.legend()
# plt.tick_params(axis='both', which='minor')
# plt.tight_layout()
# plt.show()
#
# plt.figure(3)
# ax = plt.gca()
# plt.plot(mx_list/GeV,TS_mx_null, label='Null Hypo')
# plt.plot(mx_list/GeV,TS_mx_DM, label='DM Hypo ($m_\chi$ = %0.0e eV)'%(1e8/GeV))
# plt.title('g = %0.0e eV'%(3e-1/GeV)+r', $m_\phi$ = %0.0e eV'%(1e7/GeV))
# plt.ylabel(r'$\sum TS$')
# plt.xlabel(r'$m_\chi$ (GeV)')
# plt.loglog()
# plt.legend()
# plt.tick_params(axis='both', which='minor')
# plt.tight_layout()
# plt.show()


# Differents of N_events per bin
# g,mphi,mx = [3e-1,1e7,1e8]
# lfn_scalar = BinnedLikelihoodFunction(g,mphi,mx,'scalar')
# diff_perbin = np.abs(lfn_scalar.h_null - lfn_scalar.h_DM)
# plt.step(bin_centers, diff_perbin, where='mid')
# plt.xlim(1e3, 1e7)
# plt.xlabel('Energy (GeV)')
# plt.ylabel(r'$|\Delta N_{events}|$')
# plt.gca().set_yscale("log")
# plt.gca().set_xscale("log")
# plt.tight_layout()
# plt.show()
