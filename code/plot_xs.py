import numpy as np
from xs import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('../paper.mplstyle')

def plot_xsec(g,mphi,mx,interaction):
    energies = np.logspace(3,7)*GeV # eV?
    xsec = np.array([interaction(e,g,mphi,mx) for e in energies])

    plt.plot(energies/GeV,xsec,label=interaction.__name__)
    title = 'g = {:.2f},  '.format(g) + r'$m_\phi = $  '+ '{:.0e} eV,   '.format(mphi) + r'  $m_\chi = $  '+ '{:.0e} eV'.format(mx)
    plt.title(title,fontsize=14)
    plt.xlabel('E (GeV)',fontsize=14)
    plt.ylabel(r'$\sigma_{scat}$ (eV$^{-2}$)',fontsize=14)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.loglog()


# g,mphi,mx  = [1e-1,1e7,1e8]
# plt.figure(dpi=150)
# plot_xsec(g,mphi,mx,SFHeavyMediator)
# plot_xsec(g,mphi,mx,SVHeavyMediator)
# plot_xsec(g,mphi,mx,SSHeavyMediator)
# plot_xsec(g,mphi,mx,FSHeavyMediator)
# plt.show()

# trying to get sensitivities back
# E_nu = 46e12 #eV
# mx_list = np.logspace(-7,-1)*GeV # eV
# xsec = np.array([SSHeavyMediator(E_nu,g,mphi,mx) for mx in mx_list])
# title = 'E = 46 TeV: g = {:.2f},  '.format(g) + r'$m_\phi = $  '+ '{:.0e} eV   '.format(mphi) #+ r'  $m_\chi = $  '+ '{:.0e} eV'.format(mx)
#
# plt.figure(figsize=(8,6),dpi=150)
# plt.plot(mx_list/GeV,xsec/(cm**2),label=title)
# # plt.title(title)
# plt.tight_layout()
# plt.xlabel(r'$m_\chi$ (GeV)')
# plt.ylabel(r'$\sigma_{scat}$ (cm$^{2}$)')
# legend = plt.legend(loc='best')
# legend.set_title('Scalar-Scalar')
# plt.loglog()
# plt.show()


# heatmaps
E_nu = 1e2*GeV # eV
# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

# mphi vs mx
g = 1e0*GeV # eV
mphi_list = np.logspace(-4, 1, int(10/dy))*GeV # use difference between start and stop values
mx_list = np.logspace(-4, 1, int(10/dx))*GeV

# X, Y = np.meshgrid(mx_list, mphi_list)
xs_vals = [[SSHeavyMediator(E_nu,g,mphi,mx)/(cm**2) for mx in mx_list] for mphi in mphi_list]
# xs_vals = [[SSHeavyMediator(E_nu,g,mphi,mx)/(cm**2) for mphi in mphi_list] for mx in mx_list]

extent =  np.min(mx_list/GeV), np.max(mx_list/GeV), np.min(mphi_list/GeV), np.max(mphi_list/GeV)
# extent =  np.min(mphi_list/GeV), np.max(mphi_list/GeV), np.min(mx_list/GeV), np.max(mx_list/GeV)

fig = plt.figure()
im1 = plt.pcolormesh(mx_list/GeV, mphi_list/GeV, np.log10(xs_vals))
# im1 = plt.imshow(np.log10(xs_vals), cmap=plt.cm.viridis, interpolation='nearest', extent=extent) #norm=matplotlib.colors.LogNorm()
# ticks = matplotlib.ticker.LogLocator()
# formatter = matplotlib.ticker.LogFormatterExponent(base=10)

cbar = plt.colorbar(shrink=0.8, pad=0.08,label=r'$\log_{10}\sigma$ (cm$^2$)')
plt.tight_layout()
plt.loglog()
plt.title('g = 1 GeV, E = 100 GeV')
plt.xlabel(r'$m_\chi$ (GeV)')
plt.ylabel(r'$m_\phi$ (GeV)')
plt.show()
