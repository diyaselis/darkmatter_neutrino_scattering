import numpy as np
#this gives the mass density; divide by mx to get number density!
def DMdensity(b,l,x):
	#b, l: galactic lat, long in radians
	#x: distance from earth, kpc
	R = 8.5 # solar radius, kpc
	r  = np.sqrt(x**2+R**2-2.*R*x*np.cos(b)*np.cos(l))
	#Einasto
	a  =0.17 #profile shape
	rs = 26.
	rhos = 0.4*np.exp(2./a*(pow((R/rs),a)-1.)) #density normalisation, GeV/cm**3
	rho = rhos*np.exp(-2/a*(pow((r/rs),a)-1.))
    #NFW (uncomment if you want it)
#     g = 1.;
#     rhos = 0.4/pow(2,(3-g))*pow((D/rs),g)*pow((1-D/rs),(3.-g))
#     rho = pow(2.,(3.-g))./pow((r/rs),g)./pow((1+r/rs),(3-g))
	return rho # GeV/cm^3. Remember units if you integrate along the l.o.s.