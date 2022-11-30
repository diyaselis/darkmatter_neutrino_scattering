import numpy as np
# This is for Dirac DM fermion and scalar singlet mediator
# Enu_i: Initial Energy before scattering
# Enu_f: Final Energy after scattering
# theta: 1-x

#costheta eq 7
def theta(Enu_i, Enu_f, mx):
    return mx*(1./Enu_f-1/Enu_i)

def dthetadE(Enu_f, mx):
    return mx/Enu_f**2


#eq 13
def dxsdE_f_fermion(Enu_i, Enu_f, g, mx, mphi):
    Theta = -theta(Enu_i, Enu_f, mx)
    return g**2*Theta*Enu_i**2*mx**2*(2*Theta*Enu_i*mx+Theta*Enu_i**2-2*mx**2)/(8*np.pi*(mx-Theta*Enu_i)**2*(Theta*Enu_i*mphi**2-mx*(mphi**2-2*Theta*Enu_i**2))**2)*dthetadE(Enu_f, mx)

