import numpy as np
# This is for a scalar DM particle with a fermionic mediator
# Enu_i: Initial Energy before scattering
# Enu_f: Final Energy after scattering
# theta: 1-x

def Theta(Enu_i, Enu_f, mx):
    return mx*(1./Enu_f-1./Enu_i)

def dthetadE(Enu_f, mx):
    return mx/Enu_f**2

def dxsdE_f_fermscal(Enu_i, Enu_f, g, mx, mphi):
    x = 1.-Theta(Enu_i, Enu_f, mx) #regular x
    Enu = Enu_i
    dxs = dthetadE(Enu_f, mx)*g**2*(1.+x)*Enu**4*mx**5*((1.-x)*Enu+2*mx)**2/((4.*np.pi)*(mphi**2-mx*(2*Enu+mx))**2*((1.-x)*Enu+mx)**3*(Enu*((x-1.)*mphi**2-(x+1.)*mx**2)+mx**3-mx*mphi**2)**2)
    #     dxs = dthetadE(Enu, mx)*g**2*Enu**2/(8.*np.pi*((mphi**2-mx**2)**2-4.*Enu**2*mx**2)**2*(mx-(x-1.)*Enu)**4)
#     dxs = dxs*(12.*(x-1.)**2*Enu**4*mx**3-4.*(x-1.)*(x+7.)*Enu**3*mx**4+Enu**2*mx*((x*(x+2.)+13.)*mx**4-2.*(x-1.)**2*mx**2*mphi**2+(x-1.)**2*mphi**4)+(x-1.)**2*Enu*(mphi**2-mx**2)**2*(mx**2+2.*mphi**2)-(x-1.)*mx*(mx**6-3.*mx**2*mphi**4+2.*mphi**6))
    return dxs

def dxsdE_f_fermscal_finiteWidth(Enu_i, Enu_f, g, mx, mphi):
    x = 1.-Theta(Enu_i, Enu_f, mx) #regular x
    Enu = Enu_i
    if mphi > mx:
        ECM = (mx**2 + 2.*Enu*mx)**(1/2)
        width = g/4./np.pi *(mphi**2 -mx**2)**2/mphi**3
        width = width*ECM/mphi #lorentz boost
        dxs = dthetadE(Enu, mx)*g**2*Enu**2*mx*(2*mx*mphi**2*(width**2*(x-1)*Enu-width**2*mx+2.*(x-1.)*mx**3)+32.*Enu**2*mx**3*((x-1)*Enu-mx)+(x-1)*mphi**4*(width**2-8.*mx**2)+4.*(x-1)*mphi**6)
        dxs = dxs/(16.*np.pi*(mx*(2.*Enu-mx)+mphi**2)**2*((x-1)*Enu-mx)**3*(mphi**2*(width**2-2*mx*(2*Enu+mx))+mx**2*(2*Enu+mx)**2+mphi**4))
    else:
        dxs = dxsdE_f_fermscal(Enu_i, Enu_f, g, mx, mphi)
    
    #     dxs = dthetadE(Enu, mx)*g**2*Enu**2/(8.*np.pi*((mphi**2-mx**2)**2-4.*Enu**2*mx**2)**2*(mx-(x-1.)*Enu)**4)
    #     dxs = dxs*(12.*(x-1.)**2*Enu**4*mx**3-4.*(x-1.)*(x+7.)*Enu**3*mx**4+Enu**2*mx*((x*(x+2.)+13.)*mx**4-2.*(x-1.)**2*mx**2*mphi**2+(x-1.)**2*mphi**4)+(x-1.)**2*Enu*(mphi**2-mx**2)**2*(mx**2+2.*mphi**2)-(x-1.)*mx*(mx**6-3.*mx**2*mphi**4+2.*mphi**6))
    return dxs

#u-channel only
# def dxsdE_f_fermscal(Enu_i, Enu_f, g, mx, mphi):
#     theta = Theta(Enu_i, Enu_f, mx)
#     return dthetadE(Enu_f, mx)*g**2*Enu_i**2*mx*(theta*mphi**2+mx*(2.*theta*Enu_i+(2.-theta)*mx))/(16.*np.pi*(theta*Enu_i+mx)**3*(mphi**2+(2.*Enu_i-mx)*mx)**2)
