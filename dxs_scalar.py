import numpy as n

# Enu_i: Initial Energy before scattering
# Enu_f: Final Energy after scattering
# theta: 1-x

def theta(Enu_i, Enu_f, mx):
    return mx*(1./Enu_f-1/Enu_i)

def dthetadE(Enu_f, mx):
    return mx/Enu_f**2

def dxsdE_f_scalar(Enu_i, Enu_f, g, mx, mphi):
    th = theta(Enu_i, Enu_f, mx)
    return dthetadE(Enu_f, mx)*g**2/(16*n.pi)*(th*Enu_i**2*mx)/((th*Enu_i+mx)*\
                                                              (th*Enu_i*mphi**2+\
                                                               mx*(mphi**2+2*th*Enu_i**2))**2) #corrected dsign mistake here
