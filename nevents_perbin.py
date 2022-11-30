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
    y = Y[:, 0] + np.sum(w*dY, axis=1)
    return y


def get_att_value_theta(w, v, ci, energy_nodes, E,phi_in,t):
    # print('w',w[0],'v',v[0],'ci',ci[0])
    # print('phi_in',phi_in)
    logE = np.log10(E)[:len(t)]
    w=np.tile(w,[len(t),1])
    phisol = np.inner(v,ci*np.exp(w.T*t[:len(t)]).T).T*energy_nodes**(-2)/ phi_in #this is the attenuated flux phi/phi_0
    # print('phisol',phisol[0])
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

g, mphi, mx = [3e-1,1e7,1e8]
w, v, ci, energy_nodes, phi_0 = cas.get_eigs(g,mphi,mx,'scalar',gamma,logemin,logemax)
# print('phi_in',energy_nodes ** (-gamma))

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

nu_weight_astro_null=2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**(-2.67)*exposure
nu_weight_astro = nu_weight_astro_null*flux_astro
# nu_weight_E2=MC['oneweight']*(MC['true_energy'])**-2*exposure
nu_weight_atm=1.08*MC['weight_conv']*exposure #phi_atm*MC['weight_conv']*exposure*(MC['true_energy']/2020)**(-dgamma)
mu_weight=0.8851*muon_mc['weight']*exposure


weight_null = np.append(nu_weight_astro_null+nu_weight_atm,mu_weight)
DM_weight = np.append(nu_weight_astro+nu_weight_atm,mu_weight)


num_observed = np.random.poisson(len(data))
print(num_observed)

try:
    random_data_ind=np.load('random_data_ind_DM.npy')
except:
    random_data_ind_astro=np.random.choice(len(MC),int(np.sum(nu_weight_astro)), replace=False,p=nu_weight_astro/np.sum(nu_weight_astro))
    random_data_ind_atm=np.random.choice(len(MC),int(np.sum(nu_weight_atm)), replace=False,p=nu_weight_atm/np.sum(nu_weight_atm))
    random_data_ind_muon=np.random.choice(len(muon_mc),int(np.sum(mu_weight)), replace=True,p=mu_weight/np.sum(mu_weight))+len(MC)
    random_data_ind=np.append(np.append(random_data_ind_atm,random_data_ind_muon),random_data_ind_astro)

    np.save('random_data_ind_DM.npy',random_data_ind)

fake_energy=np.append(MC['energy'],muon_mc['energy'])[random_data_ind]
fake_ra=np.append(MC['ra'],muon_mc['ra'])[random_data_ind]
fake_dec=np.append(MC['dec'],muon_mc['dec'])[random_data_ind]

plt.figure(1)
#energy
# bins = np.logspace(2,8, 25)
bins = np.logspace(3,7, 16+1)
astro,astrobin=np.histogram(MC['energy'],bins=bins,weights=nu_weight_astro)
atm,atmbin=np.histogram(MC['energy'],bins=bins,weights=nu_weight_atm)
muon,muonbin=np.histogram(muon_mc['energy'],bins,weights=mu_weight)
width = np.diff(astrobin)
center = (astrobin[:-1] + astrobin[1:]) / 2
plt.bar(center, astro+atm+muon, align='center', width=width,label='astro',color='#94e3bd')
plt.bar(center, atm+muon, align='center', width=width,label='atm',color='#85a3b1')
plt.bar(center, muon, align='center',width=width,label='muon', color='#e0bdb4')

h_DM,_,_ = plt.hist(np.append(MC['energy'],muon_mc['energy']),
                          bins=bins,weights=DM_weight,histtype="step",label='DM',color='red',lw=1)
h_null,_,_ = plt.hist(np.append(MC['energy'],muon_mc['energy']),
                            bins=bins,weights=weight_null,histtype="step", label='Null', color='black',lw=1)

counts,bin_edges = np.histogram(data['energy'], bins=bins)
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.

# muon=np.load('../mese_cascades_muongun_dnn.npy')
fake_data,fake_databin=np.histogram(fake_energy,bins=bins)
#plt.bar(center, astro+atm, align='center', width=width,label='astro',color='green')
#plt.bar(center,fake_data,align='center',width=width,label='fake',fill=False,edgecolor='k',linewidth=3)
plt.errorbar(center,fake_data,yerr=np.sqrt(fake_data)+1e-3,xerr=width/2,color='r',capsize=4,linestyle='None',linewidth=3,label='DM attenuation')
plt.errorbar(bin_centres,counts,yerr=np.sqrt(counts),xerr=width/2,color='k',capsize=4,linestyle='None',linewidth=2,label='simulated data')
E=np.linspace(1e3,1e7) # GeV
volume=1e2*(1e2)**2
#plt.plot(E,(E/1e5)**(-2.0)*exposure*volume*1e-18*E,'k')
#plt.plot(E,(E/1e5)**(-2.67)*exposure*volume*1e-18*E,'r')
#plt.plot(E,(E/1e5)**(-3.0)*exposure*volume*1e-18*E,'g')
print(np.sum(nu_weight_astro))
plt.xlim(1e3, 1e7)
plt.xlabel('Energy (GeV)')
plt.ylabel('Number of Events')
plt.ylim(5e-1, 2e3)
plt.legend()
plt.gca().set_yscale("log")
plt.gca().set_xscale("log")
# plt.show()

print(h_DM,'\n', np.sum(h_DM))
quit()

#dec
plt.figure(3)
counts,bin_edges = np.histogram(np.sin(data['dec']), np.linspace(-1,1, 15+1))
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
astro,astrobin=np.histogram(np.sin(MC['dec']),bins=np.linspace(-1,1, 15+1),weights=nu_weight_astro)
atm,atmbin=np.histogram(np.sin(MC['dec']),bins=np.linspace(-1,1, 15+1),weights=nu_weight_atm)
muon,muonbin=np.histogram(np.sin(muon_mc['dec']),bins=np.linspace(-1,1, 15+1),weights=mu_weight)
width = np.diff(astrobin)
center = (astrobin[:-1] + astrobin[1:]) / 2
plt.bar(center, astro+atm+muon, align='center', width=width,label='astro')
plt.bar(center, atm+muon, align='center', width=width,label='atm')
plt.bar(center, muon, align='center',width=width,label='muon')
fake_data,fake_databin=np.histogram(np.sin(fake_dec),bins=np.linspace(-1,1, 15+1))
#plt.bar(center, astro+atm, align='center', width=width,label='astro',color='green')
#plt.bar(center,fake_data,align='center',width=width,label='fake',fill=False,edgecolor='k',linewidth=3)
plt.errorbar(center,fake_data,yerr=np.sqrt(fake_data)+1e-3,xerr=width/2,color='r',capsize=4,linestyle='None',linewidth=3,label='fake data')
plt.errorbar(bin_centres,counts,yerr=np.sqrt(counts),xerr=width/2,color='k',capsize=4,linestyle='None',linewidth=2,label='data')

plt.xlabel('dec')
plt.ylabel('Number of Events')
plt.ylim(5e-1, 2e3)
plt.legend()
plt.gca().set_yscale("log")
plt.show()

#ra
plt.figure(4)
counts,bin_edges = np.histogram(data['ra'], np.linspace(0, 2*np.pi, 15+1))
bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
astro,astrobin=np.histogram(MC['ra'],bins=np.linspace(0, 2*np.pi, 15+1),weights=nu_weight_astro)
atm,atmbin=np.histogram(MC['ra'],bins=np.linspace(0, 2*np.pi, 15+1),weights=nu_weight_atm)
muon,muonbin=np.histogram(muon_mc['ra'],bins=np.linspace(0, 2*np.pi, 15+1),weights=mu_weight)
width = np.diff(astrobin)
center = (astrobin[:-1] + astrobin[1:]) / 2
plt.bar(center, astro+atm+muon, align='center', width=width,label='astro')
plt.bar(center, atm+muon, align='center', width=width,label='atm')
plt.bar(center, muon, align='center',width=width,label='muon')
fake_data,fake_databin=np.histogram(fake_ra,bins=np.linspace(0, 2*np.pi, 15+1))
#plt.bar(center, astro+atm, align='center', width=width,label='astro',color='green')
#plt.bar(center,fake_data,align='center',width=width,label='fake',fill=False,edgecolor='k',linewidth=3)
plt.errorbar(center,fake_data,yerr=np.sqrt(fake_data)+1e-3,xerr=width/2,color='r',capsize=4,linestyle='None',linewidth=3,label='fake data')
plt.errorbar(bin_centres,counts,yerr=np.sqrt(counts),xerr=width/2,color='k',capsize=4,linestyle='None',linewidth=2,label='data')
plt.xlabel('RA')
plt.ylabel('Number of Events')
plt.legend()
plt.gca().set_yscale("log")
plt.show()




hist,histbins=np.histogram(np.log10(MC['energy']),bins=np.log10(np.logspace(3,7, 16+1)),weights=nu_weight_astro,density=False)
ind=np.searchsorted(histbins,np.log10(np.append(MC['energy'],muon_mc['energy'])[random_data_ind]))
ind[ind==17]=16
ind[ind==0]=1
pdf=hist[ind-1]

width = np.diff(histbins)
center = (histbins[:-1] + histbins[1:]) / 2
plt.bar(center, hist, align='center', capsize=4, width=width, fill=False)# yerr=astro_err)
plt.plot(np.log10(np.append(MC['energy'],muon_mc['energy'])[random_data_ind]), pdf,'o')#/kde_norm
plt.show()


plt.bar(center, astro, align='center', width=width,label='astro',color='green')
E=np.linspace(1e3,1e7) # GeV
volume=1e2*(1e2)**2
plt.plot(E,(E/1e5)**(-2.0)*exposure*volume*1e-18*E,'k')
plt.plot(E,(E/1e5)**(-2.67)*exposure*volume*1e-18*E,'r')
#plt.plot(E,(E/1e5)**(-3.0)*exposure*volume*1e-18*E,'g')
plt.xlim(1e3, 1e7)
plt.xlabel('Energy')
plt.ylabel('Number of Events')
plt.ylim(5e-1, 2e3)
plt.gca().set_yscale("log")
plt.gca().set_xscale("log")
plt.show()
