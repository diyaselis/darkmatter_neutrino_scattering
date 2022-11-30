import numpy as np
import scipy as sp
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from scipy.integrate import ode
from numpy import linalg as LA
from xs import *
from dxs_scalar import *
from dxs_fermion import *
from dxs_fermscal import *
from dxs_vecferm import *


parsec = 3.0857e16*meter
kpc = 1.0e3*parsec


def get_RHS_matrices(g,mphi,mx, interaction,energy_nodes):
    NumNodes = energy_nodes.shape[0]

    # auxiliary functions
    if interaction == 'scalar':
        sigma = lambda E: SSHeavyMediator(E,g,mphi,mx)
        DiffXS = lambda Ei,Ef: dxsdE_f_scalar(Ei, Ef, g, mx, mphi) if (Ei > Ef) else 0
    if interaction == 'fermion':
        sigma = lambda E: SFHeavyMediator(E,g,mphi,mx)
        DiffXS = lambda Ei,Ef: dxsdE_f_fermion(Ei, Ef, g, mx, mphi) if (Ei > Ef) else 0
    elif interaction == 'vector':
        sigma = lambda E: SVHeavyMediator(E,g,mphi,mx)
        DiffXS = lambda Ei,Ef: dxsdE_f_vector(Ei, Ef, g, mx, mphi) if (Ei > Ef) else 0
    elif interaction == 'fermscal': #fermion mediator, scalar DM
        sigma = lambda E: FSHeavyMediator(E,g,mphi,mx)
        DiffXS = lambda Ei,Ef: dxsdE_f_fermscal(Ei, Ef, g, mx, mphi) if (Ei > Ef) else 0

    sigma_array = np.array(list(map(sigma,energy_nodes))) # this is for python 3

    #sigma_array = np.array(map(sigma,energy_nodes)) # this is for python 2
    # matrices and arrays
    dsigmadE = np.array([[DiffXS(Ei,Ef) for Ei in energy_nodes] for Ef in energy_nodes])
    DeltaE = np.diff(np.log(energy_nodes))
    # print(dsigmadE.shape, dsigmadE)
    RHSMatrix = np.zeros((NumNodes,NumNodes))
# fill in diagonal terms
    for i in range(NumNodes):
        for j in range(i+1,NumNodes):
            RHSMatrix[i][j] = DeltaE[j-1]*dsigmadE[i][j]*energy_nodes[j]**-1*energy_nodes[i]**2
    return RHSMatrix, sigma_array

def get_eigs(g,mphi,mx, interaction, gamma,logemin,logemax):
    """ Returns the eigenvalues and vectors of matrix M in eqn 6

        Args:
            mp: class for DM-nu scenario containing the coupling (mp.g), DM mass (mp.mx) in eV and mediator mass (mp.mphi) in eV
            interaction: interaction between DM and nu
            gamma: power law index of isotropic flux E^-gamma
            logemin: min nu energy log of GeV
            logemax: max nu energy log of GeV

        Returns:
            w: eigenvalues of M matrix in eV
            v: eigenvectors of M matrix in eV
            ci:coefficients in eqn 7
            energy_nodes: neutrino energy in eV
            phi0: initial neutrino flux
        """
    #Note that the solution is scaled by E^2; if you want to modify the incoming spectrum a lot, you'll need to change this here, as well as in the definition of RHS.
    NumNodes = 5 #120
    energy_nodes = np.logspace(logemin,logemax,NumNodes)*GeV # eV
    RHSMatrix, sigma_array = get_RHS_matrices(g,mphi,mx,interaction,energy_nodes)
    phi_0 = energy_nodes**(2-gamma) #eV^(2-gamma)

    w,v = LA.eig(-np.diag(sigma_array)+RHSMatrix)
    # print(w,v)
    ci = LA.lstsq(v,phi_0,rcond=None)[0]
    # print(ci)
    return w,v,ci,energy_nodes, phi_0
#
# gamma,logemin,logemax = [2.67,-2,7]
#
# g,mphi,mx = [3e-1,1e0,1e3]
# get_eigs(g,mphi,mx,'scalar',gamma,logemin,logemax)
# quit()
#
# g,mphi,mx = [3e-1,1e0,1e3]
# get_eigs(g,mphi,mx,'scalar',gamma,logemin,logemax)
