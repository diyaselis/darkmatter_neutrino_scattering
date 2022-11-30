import numpy as np

# Enu_i: Initial Energy before scattering
# Enu_f: Final Energy after scattering
# theta: 1-x

def theta(Enu_i, Enu_f, mx):
    return mx*(1./Enu_f-1/Enu_i)

def dthetadE(Enu_f, mx):
    return mx/Enu_f**2

def dxsdE_f_vector(Enu_i, Enu_f, g, mx, mphi):
    Theta = theta(Enu_i, Enu_f, mx)
    return g ** 2 * Enu_i ** 2 * mx ** 2 * (Theta*(2.-Theta)*Enu_i*mx+Theta**2*Enu_i**2+(2.-Theta)*mx**2) / (
                4. * np.pi * (Theta * Enu_i + mx) * (
                    Theta * Enu_i * mphi ** 2 + mx * (mphi ** 2 + 2. * Theta * Enu_i ** 2)) ** 2) * dthetadE(Enu_f, mx)
    # return g**2*Enu_i**2*mx**2*(2.*Theta*Enu_i+(2.-Theta)*mx)/(4.*np.pi*(Theta*Enu_i+mx)*(Theta*Enu_i*mphi**2+mx*(mphi**2+2.*Theta*Enu_i**2))**2)*dthetadE(Enu_f, mx)

