#!/usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

import cascade as cas
from xs import *

import sys
import time
import h5py


#Choose the flavor & index you want
# flavor = 2  # 1,2,3 = e, mu, tau; negative sign for antiparticles
#gamma = 2.  # Power law index of isotropic flux E^-gamma
logemin = 3 #log GeV
logemax = 7 #log GeV

def interpolation(x,X,Y):
    dX = np.diff(X)
    dY = np.diff(Y, axis=1)
    w = np.clip((x[:,None] - X[:-1])/dX[:], 0, 1)
    y = Y[:, 0] + np.sum(w*dY, axis=1)
    return y


def get_att_value_theta(w, v, ci, energy_nodes, E,phi_in,t):
    logE = np.log10(E)[:len(t)]
    w=np.tile(w,[len(t),1])
    phisol = np.inner(v,ci*np.exp(w.T*t[:len(t)]).T).T*energy_nodes**(-2)/ phi_in #this is the attenuated flux
    return interpolation(logE, np.log10(energy_nodes), phisol)#np.interp(logE, np.log10(energy_nodes), phisol)


#2428 days
exposure=2428*24*60*60

#average evaluation time 123 s
# This is the same as last weeks file except the 'data' has been replaced with fake data. This is to see if the

data=np.load('../mc/mese_cascades_2010_2016_dnn.npy')
MC=np.load('../mc/mese_cascades_MC_2013_dnn.npy')
muon_mc=np.load('../mc/mese_cascades_muongun_dnn.npy')

nu_weight_astro= 2.1464 * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**-(2.67)*exposure #(0.23 * np.random.randn() + 2.1464) * 1e-18 * MC['oneweight'] * (MC['true_energy']/1e5)**-(0.04 * np.random.randn() + 2.67)*exposure
nu_weight_atm=(1.08)*MC['weight_conv']*exposure
mu_weight=(0.8851)*muon_mc['weight']*exposure


column_dens=np.load('column_dens.npy')

random_data_ind=np.load('random_data_ind_null.npy')#np.random.choice(len(MC)+len(muon_mc), num_observed, replace=False,p=np.append(nu_weight_astro+nu_weight_atm,mu_weight)/np.sum(np.append(nu_weight_astro+nu_weight_atm,mu_weight)))
print(random_data_ind)


mc_energy=np.log10(MC['energy'])
mc_ra=MC['ra']
mc_dec=np.sin(MC['dec'])
muon_mc_energy=np.log10(muon_mc['energy'])
muon_mc_ra=muon_mc['ra']
muon_mc_dec=np.sin(muon_mc['dec'])

def initializeKDE():
    random_data_en=np.log10(np.append(MC['energy'],muon_mc['energy'])[random_data_ind])
    random_data_dec=np.sin(np.append(MC['dec'],muon_mc['dec'])[random_data_ind])
    random_data_ra=np.append(MC['ra'],muon_mc['ra'])[random_data_ind]

    kde_energy = KernelDensity(bandwidth=0.5, kernel='epanechnikov').fit(mc_energy[:, None])
    #log_dens_energy = kde_energy.score_samples(random_data_en[:,None])

    kde_dec = KernelDensity(bandwidth=0.5, kernel='epanechnikov').fit(mc_dec[:, None])
    #log_dens_dec = kde_dec.score_samples(random_data_dec[:,None])

    kde_ra = KernelDensity(bandwidth=1.0, kernel='epanechnikov').fit(mc_ra[:, None])
    #log_dens_ra = kde_ra.score_samples(random_data_ra[:,None])

    kde_muon_energy = KernelDensity(bandwidth=0.5, kernel='epanechnikov').fit(muon_mc_energy[:, None])
    #log_dens_muon_energy = kde_muon_energy.score_samples(random_data_en[:,None])

    kde_muon_dec = KernelDensity(bandwidth=0.5, kernel='epanechnikov').fit(muon_mc_dec[:, None])
    #log_dens_muon_dec = kde_muon_dec.score_samples(random_data_dec[:,None])

    kde_muon_ra = KernelDensity(bandwidth=1.0, kernel='epanechnikov').fit(muon_mc_ra[:, None])
    #log_dens_muon_ra = kde_muon_ra.score_samples(random_data_ra[:,None])

    def get_kernel(kde,random_data,bw,mc_data):
        nearest=kde.tree_.query_radius(random_data[:, None],bw)
        nearest=np.unique(np.hstack(nearest))
        kernel=1.0/bw*3.0/4.0*(1-((random_data-mc_data[nearest])/bw)**2)

        return nearest, kernel

    def appendh5(hf,dataset,data):
        hf[dataset].resize((hf[dataset].shape[0] + data.shape[0]), axis = 0)
        hf[dataset][-data.shape[0]:] = data

    nearest_muon_energy=[]
    nearest_muon_dec=[]
    nearest_muon_ra=[]
    kernel_muon_energy=[]
    kernel_muon_dec=[]
    kernel_muon_ra=[]
    num_nearest_muon_energy=[]
    num_nearest_muon_dec=[]
    num_nearest_muon_ra=[]
    num_kernel_muon_energy=[]
    num_kernel_muon_dec=[]
    num_kernel_muon_ra=[]

    for i in range(len(random_data_ind)):
        print(i)
        #energy
        random_data=np.asarray([random_data_en[i]])
        bw=0.5
        nearest,kernel=get_kernel(kde_muon_energy,random_data,bw,muon_mc_energy)
        nearest_muon_energy.append(nearest)
        kernel_muon_energy.append(kernel)
        num_nearest_muon_energy.append(len(nearest))
        num_kernel_muon_energy.append(len(kernel))
        #dec
        random_data=np.asarray([random_data_dec[i]])
        bw=0.5
        nearest,kernel=get_kernel(kde_muon_dec,random_data,bw,muon_mc_dec)
        nearest_muon_dec.append(nearest)
        kernel_muon_dec.append(kernel)
        num_nearest_muon_dec.append(len(nearest))
        num_kernel_muon_dec.append(len(kernel))

        #ra
        random_data=np.asarray([random_data_ra[i]])
        bw=1.0
        nearest,kernel=get_kernel(kde_muon_ra,random_data,bw,muon_mc_ra)
        nearest_muon_ra.append(nearest)
        kernel_muon_ra.append(kernel)
        num_nearest_muon_ra.append(len(nearest))
        num_kernel_muon_ra.append(len(kernel))


    with h5py.File('kdes.h5', 'w') as hf:
        hf.create_dataset("nearest_energy",  data=[],maxshape=(None,))
        hf.create_dataset("kernel_energy",  data=[],maxshape=(None,))
        hf.create_dataset("nearest_muon_energy",  data=np.concatenate(nearest_muon_energy).ravel())
        hf.create_dataset("kernel_muon_energy",  data=np.concatenate(kernel_muon_energy).ravel())
        hf.create_dataset("nearest_dec",  data=[],maxshape=(None,))
        hf.create_dataset("kernel_dec",  data=[],maxshape=(None,))
        hf.create_dataset("nearest_muon_dec",  data=np.concatenate(nearest_muon_dec).ravel())
        hf.create_dataset("kernel_muon_dec",  data=np.concatenate(kernel_muon_dec).ravel())
        hf.create_dataset("nearest_ra",  data=[],maxshape=(None,))
        hf.create_dataset("kernel_ra",  data=[],maxshape=(None,))
        hf.create_dataset("nearest_muon_ra",  data=np.concatenate(nearest_muon_ra).ravel())
        hf.create_dataset("kernel_muon_ra",  data=np.concatenate(kernel_muon_ra).ravel())
        hf.create_dataset("num_nearest_energy",  data=[],maxshape=(None,))
        hf.create_dataset("num_kernel_energy",  data=[],maxshape=(None,))
        hf.create_dataset("num_nearest_muon_energy",  data=num_nearest_muon_energy)
        hf.create_dataset("num_kernel_muon_energy",  data=num_kernel_muon_energy)
        hf.create_dataset("num_nearest_dec",  data=[],maxshape=(None,))
        hf.create_dataset("num_kernel_dec",  data=[],maxshape=(None,))
        hf.create_dataset("num_nearest_muon_dec",  data=num_nearest_muon_dec)
        hf.create_dataset("num_kernel_muon_dec",  data=num_kernel_muon_dec)
        hf.create_dataset("num_nearest_ra",  data=[],maxshape=(None,))
        hf.create_dataset("num_kernel_ra",  data=[],maxshape=(None,))
        hf.create_dataset("num_nearest_muon_ra",  data=num_nearest_muon_ra)
        hf.create_dataset("num_kernel_muon_ra",  data=num_kernel_muon_ra)

    for i in range(len(random_data_ind)):
        print(i)
        #energy
        random_data=np.asarray([random_data_en[i]])
        bw=0.5
        nearest_energy,kernel_energy=get_kernel(kde_energy,random_data,bw,mc_energy)

        #dec
        random_data=np.asarray([random_data_dec[i]])
        bw=0.5
        nearest_dec,kernel_dec=get_kernel(kde_dec,random_data,bw,mc_dec)

        #ra
        random_data=np.asarray([random_data_ra[i]])
        bw=1.0
        nearest_ra,kernel_ra=get_kernel(kde_ra,random_data,bw,mc_ra)

        with h5py.File('kdes.h5', 'a') as hf:
            appendh5(hf,"nearest_energy",nearest_energy)
            appendh5(hf,"kernel_energy",kernel_energy)
            appendh5(hf,"num_nearest_energy",np.asarray([len(nearest_energy)]))
            appendh5(hf,"num_kernel_energy",np.asarray([len(kernel_energy)]))

            appendh5(hf,"nearest_dec",nearest_dec)
            appendh5(hf,"kernel_dec",kernel_dec)
            appendh5(hf,"num_nearest_dec",np.asarray([len(nearest_dec)]))
            appendh5(hf,"num_kernel_dec",np.asarray([len(kernel_dec)]))

            appendh5(hf,"nearest_ra",nearest_ra)
            appendh5(hf,"kernel_ra",kernel_ra)
            appendh5(hf,"num_nearest_ra",np.asarray([len(nearest_ra)]))
            appendh5(hf,"num_kernel_ra",np.asarray([len(kernel_ra)]))

    quit()

initializeKDE()
