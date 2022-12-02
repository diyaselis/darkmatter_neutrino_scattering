import numpy as np
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

class BinnedLikelihoodFunction:

    def __init__(self,g,mphi,mx,interaction):
        self.g = g
        self.mphi = mphi
        self.mx = mx
        self.interaction = interaction
        # (data,MC,muon_MC,binning):

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
        mu_weight=0.8851*muon_mc['weight']*exposure
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

    def __call__(self): #*params
        weight_null,weight_DM = self.GetWeights(self.g,self.mphi,self.mx,self.interaction)
        h_null,_ = np.histogram(np.append(MC['energy'],muon_mc['energy']),bins=bins,weights=weight_null)
        h_DM,_  = np.histogram(np.append(MC['energy'],muon_mc['energy']),bins=bins,weights=weight_DM)
        TS_null = np.sum(self.TestStatistics(h_null,h_DM))
        # compare to DM parameters
        _,weight_DM_true = self.GetWeights(params)
        h_DM_true,_  = np.histogram(np.append(MC['energy'],muon_mc['energy']),bins=bins,weights=weight_DM_true)
        TS_DM = np.sum(self.TestStatistics(h_DM_true,h_DM))
        return TS_null #TS_DM


# lfn_scalar = BinnedLikelihoodFunction(g,mphi,mx,'scalar')
# ts_null_scalar = lfn_scalar()


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
    print(np.exp(w.T*t[:len(t)]))
    phisol = np.inner(v,ci*np.exp(w.T*t[:len(t)]).T).T*energy_nodes**(-2)/ phi_in #attenuated flux phi/phi_0
    interp = interpolation(logE, np.log10(energy_nodes), phisol)
    return interp
