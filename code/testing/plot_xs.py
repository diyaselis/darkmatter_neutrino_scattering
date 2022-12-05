import numpy as np
from xs import *
from dxs_scalar import *
from dxs_fermion import *
from dxs_fermscal import *
from dxs_vecferm import *

import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.style.use('../paper.mplstyle')
matplotlib.rcParams['figure.figsize'] = [10,7]
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams ['xtick.direction'] = 'in'
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['font.size'] = 19
matplotlib.rcParams['font.family']= 'DejaVu Serif'
matplotlib.rcParams['mathtext.default'] = 'regular'
matplotlib.rcParams['errorbar.capsize'] = 3
matplotlib.rcParams['figure.facecolor'] = (1,1,1)

def plot_xsec(g,mphi,mx,interaction):
    energies = np.logspace(3,7)*GeV # GeV?
    xsec = np.array([interaction(e,g,mphi,mx) for e in energies])

    plt.plot(energies/GeV,xsec/GeV**2,label=interaction.__name__)
    title = 'g = {:.2f},  '.format(g) + r'$m_\phi = $  '+ '{:.0e} eV,   '.format(mphi) + r'  $m_\chi = $  '+ '{:.0e} eV'.format(mx)
    plt.title(title,fontsize=14)
    plt.xlabel('E (GeV)',fontsize=14)
    plt.ylabel(r'$\sigma_{scat}$ (GeV$^{-2}$)',fontsize=14)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.loglog()

# g,mphi,mx  = [4e-1,1e6,1e8]
g,mphi,mx  = [3e-1,1e7,1e8]
plt.figure(dpi=150)
plot_xsec(g,mphi,mx,SFHeavyMediator)
plot_xsec(g,mphi,mx,SVHeavyMediator)
plot_xsec(g,mphi,mx,SSHeavyMediator)
plot_xsec(g,mphi,mx,FSHeavyMediator)
plt.show()
