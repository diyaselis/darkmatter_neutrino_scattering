import numpy as np
import cascade as cas
from xs import *
from create_fake_data import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('../paper.mplstyle')

data=np.load('../mc/mese_cascades_2010_2016_dnn.npy')
MC=np.load('../mc/mese_cascades_MC_2013_dnn.npy')
muon_mc=np.load('../mc/mese_cascades_muongun_dnn.npy')

bins=np.logspace(3,7,25)

def plot_att_Edist(g, mphi, mx):
    nu_weight_astro,nu_weight_atm,mu_weight = get_weights(g, mphi, mx)
    plt.figure()
    counts,bin_edges = np.histogram(data['energy'], bins=bins)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    astro,astrobin=np.histogram(MC['energy'],bins=bins,weights=nu_weight_astro)
    atm,atmbin=np.histogram(MC['energy'],bins=bins,weights=nu_weight_atm)
    muon,muonbin=np.histogram(muon_mc['energy'],bins=bins,weights=mu_weight)
    width = np.diff(astrobin)
    center = (astrobin[:-1] + astrobin[1:]) / 2
    plt.bar(center, astro+atm+muon, align='center', width=width,label='astro')
    plt.bar(center, atm+muon, align='center', width=width,label='atm')
    plt.bar(center, muon, align='center',width=width,label='muon')
    plt.errorbar(bin_centres,counts,yerr=np.sqrt(counts),xerr=width/2,color='k',capsize=4,linestyle='None',linewidth=2,label='data')

    # plt.hist(MC['true_energy'],bins=np.logspace(2,7, 16),weights=nu_weight_astro,color='b',alpha=0.3)
    # plt.hist(MC['true_energy'],bins=np.logspace(2,7, 16),weights=nu_weight_atm,color='r',alpha=0.3)
    # plt.hist(muon_mc['true_energy'],bins=np.logspace(2,7, 16),weights=mu_weight,color='g',alpha=0.3)

    plt.legend()
    plt.gca().set_yscale("log")
    plt.gca().set_xscale("log")
    title =  'g = {:.0e} GeV, '.format(g/GeV) + r'$m_\phi = $' + '{:.0e} GeV, '.format(mphi/GeV) + r'$m_\chi = $' + '{:.0e} GeV'.format(mx/GeV)
    plt.title(title)
    plt.xlim(1e3, 1e7)
    plt.ylim(5e-1, 1e3)
    plt.ylabel('Number of Events')
    plt.xlabel('Energy (GeV)')
    plt.tight_layout()
    plt.show()
    # fname = 'att%0.2f'%(np.sum(nu_weight_astro))+'g%0.2e'%(g)+'_mphi%0.2e'%(mphi)+'_mx%0.2e'%(mx)+'.png'
    # plt.savefig('../created_files/nevents/'+fname, dpi=200,bbox_inches='tight')
    # plt.close()

num_observed = np.random.poisson(len(data))
random_data_ind=np.load('../created_files/random_data_ind_DM.npy')

fake_energy=np.append(MC['energy'],muon_mc['energy'])[random_data_ind]
fake_ra=np.append(MC['ra'],muon_mc['ra'])[random_data_ind]
fake_dec=np.append(MC['dec'],muon_mc['dec'])[random_data_ind]

def plot_att_Edist_fakedata(g, mphi, mx):
    nu_weight_astro,nu_weight_atm,mu_weight = get_weights(g, mphi, mx)
    plt.figure(dpi=100)
    counts,bin_edges = np.histogram(data['energy'], bins=bins)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    astro,astrobin=np.histogram(MC['energy'],bins=bins,weights=nu_weight_astro)
    atm,atmbin=np.histogram(MC['energy'],bins=bins,weights=nu_weight_atm)
    muon,muonbin=np.histogram(muon_mc['energy'],bins=bins,weights=mu_weight)
    width = np.diff(astrobin)
    center = (astrobin[:-1] + astrobin[1:]) / 2
    #energy
    plt.bar(center, astro+atm+muon, align='center', width=width,label='astro')
    plt.bar(center, atm+muon, align='center', width=width,label='atm')
    plt.bar(center, muon, align='center',width=width,label='muon')
    fake_data,fake_databin=np.histogram(fake_energy,bins=bins)
    #plt.bar(center, astro+atm, align='center', width=width,label='astro',color='green')
    #plt.bar(center,fake_data,align='center',width=width,label='fake',fill=False,edgecolor='k',linewidth=3)
    plt.errorbar(center,fake_data,yerr=np.sqrt(fake_data)+1e-3,xerr=width/2,color='r',capsize=4,linestyle='None',linewidth=3,label='DM attenuation')
    plt.errorbar(bin_centres,counts,yerr=np.sqrt(counts),xerr=width/2,color='k',capsize=4,linestyle='None',linewidth=2,label='simulated data')
    # E=np.linspace(1e3,1e7) # GeV
    # volume=1e2*(1e2)**2
    #plt.plot(E,(E/1e5)**(-2.0)*exposure*volume*1e-18*E,'k')
    #plt.plot(E,(E/1e5)**(-2.67)*exposure*volume*1e-18*E,'r')
    #plt.plot(E,(E/1e5)**(-3.0)*exposure*volume*1e-18*E,'g')

    plt.xlim(1e3, 1e7)
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Number of Events')
    plt.ylim(5e-1, 2e3)
    plt.legend()
    plt.gca().set_yscale("log")
    plt.gca().set_xscale("log")
    plt.tight_layout()
    plt.show()


#dec
def plot_att_decdist(g, mphi, mx):
    nu_weight_astro,nu_weight_atm,mu_weight = get_weights(g, mphi, mx)
    plt.figure()
    counts,bin_edges = np.histogram(np.sin(data['dec']), np.linspace(-1,1, 25))
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    astro,astrobin=np.histogram(np.sin(MC['dec']),bins=np.linspace(-1,1, 25),weights=nu_weight_astro)
    atm,atmbin=np.histogram(np.sin(MC['dec']),bins=np.linspace(-1,1, 25),weights=nu_weight_atm)
    muon,muonbin=np.histogram(np.sin(muon_mc['dec']),bins=np.linspace(-1,1, 25),weights=mu_weight)
    width = np.diff(astrobin)
    center = (astrobin[:-1] + astrobin[1:]) / 2
    plt.bar(center, astro+atm+muon, align='center', width=width,label='astro')
    plt.bar(center, atm+muon, align='center', width=width,label='atm')
    plt.bar(center, muon, align='center',width=width,label='muon')
    fake_data,fake_databin=np.histogram(np.sin(fake_dec),bins=np.linspace(-1,1, 25))
    #plt.bar(center, astro+atm, align='center', width=width,label='astro',color='green')
    #plt.bar(center,fake_data,align='center',width=width,label='fake',fill=False,edgecolor='k',linewidth=3)
    plt.errorbar(center,fake_data,yerr=np.sqrt(fake_data)+1e-3,xerr=width/2,color='r',capsize=4,linestyle='None',linewidth=3,label='fake data')
    plt.errorbar(bin_centres,counts,yerr=np.sqrt(counts),xerr=width/2,color='k',capsize=4,linestyle='None',linewidth=2,label='data')

    plt.xlabel('dec')
    plt.ylabel('Number of Events')
    plt.ylim(5e-1, 2e3)
    plt.legend()
    plt.tight_layout()
    plt.gca().set_yscale("log")
    plt.show()

#ra
def plot_att_RAdist(g, mphi, mx):
    nu_weight_astro,nu_weight_atm,mu_weight = get_weights(g, mphi, mx)
    plt.figure()
    counts,bin_edges = np.histogram(data['ra'], np.linspace(0, 2*np.pi, 25))
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    astro,astrobin=np.histogram(MC['ra'],bins=np.linspace(0, 2*np.pi, 25),weights=nu_weight_astro)
    atm,atmbin=np.histogram(MC['ra'],bins=np.linspace(0, 2*np.pi, 25),weights=nu_weight_atm)
    muon,muonbin=np.histogram(muon_mc['ra'],bins=np.linspace(0, 2*np.pi, 25),weights=mu_weight)
    width = np.diff(astrobin)
    center = (astrobin[:-1] + astrobin[1:]) / 2
    plt.bar(center, astro+atm+muon, align='center', width=width,label='astro')
    plt.bar(center, atm+muon, align='center', width=width,label='atm')
    plt.bar(center, muon, align='center',width=width,label='muon')
    fake_data,fake_databin=np.histogram(fake_ra,bins=np.linspace(0, 2*np.pi,25))
    #plt.bar(center, astro+atm, align='center', width=width,label='astro',color='green')
    #plt.bar(center,fake_data,align='center',width=width,label='fake',fill=False,edgecolor='k',linewidth=3)
    plt.errorbar(center,fake_data,yerr=np.sqrt(fake_data)+1e-3,xerr=width/2,color='r',capsize=4,linestyle='None',linewidth=3,label='fake data')
    plt.errorbar(bin_centres,counts,yerr=np.sqrt(counts),xerr=width/2,color='k',capsize=4,linestyle='None',linewidth=2,label='data')
    plt.xlabel('RA')
    plt.ylabel('Number of Events')
    plt.legend()
    plt.tight_layout()
    plt.gca().set_yscale("log")
    plt.show()

# =========== plot ================
# g, mphi, mx = [3e-1,1e7,1e8]
# # g, mphi, mx = [1e0, 1e8,1e9]

# plot_att_Edist(g, mphi, mx)
# plot_att_Edist_fakedata(g, mphi, mx)
# plot_att_decdist(g, mphi, mx)
# plot_att_RAdist(g, mphi, mx)

# extra

# hist,histbins=np.histogram(np.log10(MC['energy']),bins=np.log10(np.logspace(3,7, 16+1)),weights=nu_weight_astro,density=False)
# ind=np.searchsorted(histbins,np.log10(np.append(MC['energy'],muon_mc['energy'])[random_data_ind]))
# ind[ind==17]=16
# ind[ind==0]=1
# pdf=hist[ind-1]
#
# width = np.diff(histbins)
# center = (histbins[:-1] + histbins[1:]) / 2
# plt.bar(center, hist, align='center', capsize=4, width=width, fill=False)# yerr=astro_err)
# plt.plot(np.log10(np.append(MC['energy'],muon_mc['energy'])[random_data_ind]), pdf,'o')#/kde_norm
# plt.show()
#
#
# plt.bar(center, astro, align='center', width=width,label='astro',color='green')
# E=np.linspace(1e3,1e7) # GeV
# volume=1e2*(1e2)**2
# plt.plot(E,(E/1e5)**(-2.0)*exposure*volume*1e-18*E,'k')
# plt.plot(E,(E/1e5)**(-2.67)*exposure*volume*1e-18*E,'r')
# #plt.plot(E,(E/1e5)**(-3.0)*exposure*volume*1e-18*E,'g')
# plt.xlim(1e3, 1e7)
# plt.xlabel('Energy')
# plt.ylabel('Number of Events')
# plt.ylim(5e-1, 2e3)
# plt.gca().set_yscale("log")
# plt.gca().set_xscale("log")
# plt.show()
