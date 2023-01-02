import numpy as np
import scipy as sp
import scipy.stats as stats
import random as rd
from decimal import Decimal as D
from decimal import getcontext
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import os

getcontext().prec=50

# natural units
GeV = 1.0e9
MeV = 1.0e6
keV = 1.0e3
meter = 5.06773093741e6 # m to 1/eV
cm = 1.0e-2*meter # cm to 1/eV
kg = 5.62e35 # kg to eV
gr = 1e-3*kg # g to eV
Na = 6.0221415e+23
hbarc = 3.16152677e-26
parsec = 3.085678e13 # pc to km

# class ModelParameters:
#     def __init__(self,g,mphi,mx):
#         self.g = g
#         self.mphi = mphi
#         self.mx = mx

def CMEnu(Enulab,mx):
    return Enulab*(mx/np.sqrt(mx**2+2.*Enulab*mx))

def CMEx(Enulab,mx):
    return np.sqrt(Enu(Enulab,mx)**2+mx*mx)

def SSHeavyMediator(Enu,gs,mphi,mx): #Scalar DM, Scalar mediator
    ## THIS FORMULA IS IN THE LAB FRAME (eV)
    Es=D(Enu)
    dm=D(mx)
    p=D(mphi)
    g=D(gs)
    E2=Es**D(2.0)
    m2=dm**D(2.0)
    p2=p**D(2.0)
    g2=g**D(2.0)

    t1=p2*(D(2.0)*Es+dm)
    t2=D(2.0)*Es*p2+D(4.0)*E2*dm+p2*dm
    logs=t1.ln()-t2.ln()

    num=-g2*(D(4.0)*E2*dm+t2*logs)
    den=D(64.0)*D(np.pi)*E2*m2*t2
    sig=num/den
    return float(sig)

def SFHeavyMediator(Enu,gf,mphi,mx): #Fermion DM, scalar mediator
    Es=D(Enu)
    dm=D(mx)
    p=D(mphi)
    g=D(gf)
    E2=Es**D(2.0)
    m2=dm**D(2.0)
    p2=p**D(2.0)
    g2=g**D(2.0)

    sig=g2/(32*D(np.pi)*E2*m2)*(Es*dm-Es*m2/(2*Es+dm)-Es*m2*p2*(p2-4*m2)/((2*Es*dm+p2)*(4*E2*m2+2*Es*p2+dm*p2))+Es*dm*(p2-4*m2)/(2*Es*dm+p2)+(p2-2*m2)*(p2*(2*Es+dm)/(4*E2*dm+2*Es*p2+dm*p2)).ln())
    return float(sig)

def SVHeavyMediator(Enu,gf,mphi,mx): #Fermion DM, Vector mediator
    Es=D(Enu)
    dm=D(mx)
    p=D(mphi)
    g=D(gf)
    E2=Es**D(2.0)
    m2=dm**D(2.0)
    p2=p**D(2.0)
    g2=g**D(2.0)
    sig=g2/(D(16)*D(np.pi)*E2*m2)*((p2+m2+D(2)*Es*dm)*(p2*(2*Es+dm)/(dm*(4*E2+p2)+2*Es*p2)).ln() + D(4)*E2*(D(1)+m2/p2 - (D(2)*Es*(D(4)*E2*dm + Es*(m2+D(2)*p2)+dm*p2))/((D(2)*Es+dm)*(dm*(D(4)*E2+p2)+D(2)*Es*p2))))
    return float(sig)

def FSHeavyMediator(Enu,gf,mphi,mx): #Scalar DM, fermion mediator. Be super careful with this shit
    Es=D(Enu)
    dm=D(mx)
    p=D(mphi)
    g=D(gf)
    E2=Es**D(2.0)
    m2=dm**D(2.0)
    p2=p**D(2.0)
    g2=g**D(2.0)
    logs=D(4.0)*E2*dm/(p2*(D(2.0)*Es+dm)-dm**D(3.0))+D(1.0)

    try:
        lns=logs.ln()
        sig=g2/(D(64.0)*D(np.pi))* (( D(8.0)*E2*dm/((D(2.0)*Es+dm)*(p2-dm*(D(2.0)*Es+dm))**D(2.0)) + D(4.0)/(-D(2.0)*Es*dm+m2-p2) + D(8.0)/(dm*(D(2.0)*Es+dm)-p2) + ((D(4.0)*Es*dm-D(2.0)*(m2+D(3.0)*p2))/(Es*dm*(dm*(D(2.0)*Es+dm)-p2))+D(3.0)/E2)*lns))*D(hbarc)**D(2.0)*D(100.0)**D(2.0)
    except:
        sig=gf**2/(64*np.pi)*(8.*Enu**2*mx/((2*Enu+mx)*(mphi**2-mx*(2*Enu+mx))**2)+4./(-2*Enu*mx+mx**2-mphi**2)+8./(mx*(2*Enu+mx)-mphi**2)+((4*Enu*mx-2*(mx**2+3*mphi**2))/(Enu*mx*(mx*(2*Enu+mx)-mphi**2))+3./Enu**2)*np.log(4*Enu**2*mx/(mphi**2*(2*Enu+mx)-mx**3)+1.))
    return float(sig)

def FSHeavyMediatorFiniteWidth(Enu,gf,mphi,mx): #Scalar DM, fermion mediator. Be super careful!
    if mphi > mx:
        ECM = (mx**2 + 2.*Enu*mx)**(1/2)
        width = gf/4./np.pi *(mphi**2 -mx**2)**2/mphi**3
        width = width*ECM/mphi #lorentz boost
        sig = gf**2*Enu**2*(2.*mx*mphi**2*(2.*width**2*mx+2.*mx**3)+32.*Enu**2*mx**3*(2.*Enu+mx)+mphi**4*(width**2-8.*mx**2)+4.*mphi**6)
        sig = sig/(8.*np.pi*(2*Enu+mx)**2*(mx*(2.*Enu-mx)+mphi**2)**2*(mphi**2*(width**2-2.*mx*(2.*Enu+mx))+mx**2*(2*Enu+mx)**2+mphi**4))
    else:
        sig = FSHeavyMediator(Enu,gf,mphi,mx)
    return sig
