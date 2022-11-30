import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord

# Values based on Iooco paper https://arxiv.org/pdf/1901.02460.pdf
# RS = 20*3.096e19/1e3  # kpc -> km
# gamma=1 # standard NFW
# rho_s=1e7*2e30*1e3/(3.096e19*1e2)**3 # 1e7 M sun/kpc^3 -> g/cm^3
#
# R=8.3*3.096e19/1e3 # Distance from sun to galactic centre kpc -> km

kpc=3.08567758147e16 #kpc -> km
# Values based on Iooco paper https://arxiv.org/pdf/1901.02460.pdf
RS = 26*kpc*1e5  # kpc -> cm
gamma=1 # standard NFW
rho_s=1e7*2e30*1e3/(kpc*1e5)**3 # 1e7 M sun/kpc^3 -> g/cm^3
R=8.3*kpc*1e5 # Distance from sun to galactic centre kpc -> cm


def get_angle(ra,dec):
    # ra radians from 0 to 2pi
    # dec radians from -pi/2 to pi/2
    dec_G=27.12825*np.pi/180
    ra_G=192.85948*np.pi/180
    l_NCP=122.93192*np.pi/180

    b = np.arcsin(np.sin(dec)*np.sin(dec_G)
                    +np.cos(dec)*np.cos(dec_G)*np.cos(ra-ra_G))
    b[np.where(dec<0)] += np.pi

    l = l_NCP-np.arcsin(np.cos(dec)*np.sin(ra-ra_G)/np.cos(b))
    theta=np.arccos(np.cos(b)*np.cos(l))
    return theta


def rho_NFW(theta, x):
    """ Returns the NFW density in gr/cm^3.

    Args:
	theta: angle to galactic centre (radians)
	x: distance from Earth to source (km)
	r: distance from galactic centre to source (km)

    Returns:
        rho: density in gr/cm^3
    """
    r = np.sqrt(x ** 2 + R ** 2 - 2 * x * R * np.cos(theta))
    rho=rho_s/((r/RS)**gamma*(1+r/RS)**(3-gamma))
    return rho


def get_t_NFW(ra,dec):
    """ Returns the NFW column density for a given zenith angle.

    Args:
        theta: zenith angle in radians.

    Returns:
        t: column density in g/cm^2
    """
    # x: distance from Earth to source

    xmax =  RS #np.sqrt((RS - d) ** 2 * np.cos(theta) ** 2 + d * (2 * RS - d)) - (RS - d) * np.cos(theta)
    theta = get_angle(ra,dec)
    t= []
    for i in range(len(theta)):
        n = lambda x: rho_NFW(theta, x)  # mass density
        t.append(integrate.quad(lambda x: n(xmax - x), 0, xmax, epsrel=1.0e-3, epsabs=1.0e-18)[0])# g/cm^2
    return t
