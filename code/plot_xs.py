import numpy as np
# import scipy as sp
# import scipy.integrate as integrate
# import scipy.interpolate as interpolate
# from scipy.integrate import ode
# from numpy import linalg as LA
from xs import *
from dxs_scalar import *
from dxs_fermion import *
from dxs_fermscal import *
from dxs_vecferm import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('../paper.mplstyle')

def plot_xsec(g,mphi,mx,interaction):
    energies = np.logspace(-2,7)*GeV # GeV?
    xsec = np.array([interaction(e,g,mphi,mx) for e in energies])

    plt.plot(energies,xsec,label=interaction.__name__)
    title = 'g = {:.2f},  '.format(g) + r'$m_\phi = $  '+ '{:.0e} eV,   '.format(mphi) + r'  $m_\chi = $  '+ '{:.0e} eV'.format(mx)
    plt.title(title,fontsize=14)
    plt.xlabel('E (eV)',fontsize=14)
    plt.ylabel(r'$\sigma_{scat}$ (eV$^{-2}$)',fontsize=14)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.loglog()


g = 4e-1
mphi = 1e6
mx = 1e8
plt.figure(dpi=150)
plot_xsec(g,mphi,mx,SFHeavyMediator)
plot_xsec(g,mphi,mx,SVHeavyMediator)
plot_xsec(g,mphi,mx,SSHeavyMediator)
plot_xsec(g,mphi,mx,FSHeavyMediator)
plt.show()
