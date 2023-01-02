from binned_loglikelihood import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('../paper.mplstyle')

# ========= class testing =========
# g,mphi,mx = [3e-1,1e7,1e8]
# lfn_scalar = BinnedLikelihoodFunction(g,mphi,mx,'scalar')
# ts_null_scalar = lfn_scalar() # E, RA, dec
# print(ts_null_scalar[0])

# logf = k*np.log(k)-k+np.log(2*np.pi*k)/2 + 1/(12*k) # approximated with stirling series
# logp = k*np.log(mu) - mu - logf
# logp[np.where(np.isnan(logp) == True)] = 0.0 # change NaN to zero

# ========= TS scan over one DM parameter =========
inters = ['scalar','fermion','vector','fermscal']
inter_title = ['Scalar-Scalar DM','Scalar-Fermion DM','Vector-Fermion DM','Fermion-Scalar DM']
for i,interaction in enumerate(inters):
    g_list = np.logspace(-4,0)
    mphi_list = np.logspace(6,11)
    mx_list = np.logspace(6,10)

    TS_g_null = [BinnedLikelihoodFunction(g,1e7,1e8,interaction)() for g in g_list]
    TS_mphi_null = [BinnedLikelihoodFunction(3e-1,mphi,1e8,interaction)() for mphi in mphi_list]
    TS_mx_null = [BinnedLikelihoodFunction(3e-1,1e7,mx,interaction)() for mx in mx_list]

    TS_g_DM = [BinnedLikelihoodFunction(g,1e7,1e8,interaction)(3e-1,1e7,1e8,interaction) for g in g_list]
    TS_mphi_DM = [BinnedLikelihoodFunction(3e-1,mphi,1e8,interaction)(3e-1,1e7,1e8,interaction) for mphi in mphi_list]
    TS_mx_DM = [BinnedLikelihoodFunction(3e-1,1e7,mx,interaction)(3e-1,1e7,1e8,interaction) for mx in mx_list]

    plt.figure(1,figsize=(5,6))
    ax = plt.gca()
    plt.plot(g_list/GeV,[val[0] for val in TS_g_null],'-r', label='Null - Energy')
    plt.plot(g_list/GeV,[val[1] for val in TS_g_null],'--r', label='Null - RA')
    plt.plot(g_list/GeV,[val[2] for val in TS_g_null],'-.r', label='Null - Dec')
    plt.plot(g_list/GeV,[val[0] for val in TS_g_DM],'-b', label='DM - Energy')
    plt.plot(g_list/GeV,[val[1] for val in TS_g_DM],'--b', label='DM - RA')
    plt.plot(g_list/GeV,[val[2] for val in TS_g_DM],'-.b', label='DM - Dec')
    plt.title(r'g = %0.0e eV, '%(3e-1/GeV)+r'$m_\phi$ = %0.0e eV'%(1e7/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
    plt.ylabel(r'$\sum TS$')
    plt.xlabel('g (GeV)')
    plt.loglog()
    legend = plt.legend(fontsize=14)
    legend.set_title(inter_title[i])
    plt.tick_params(axis='both', which='minor')
    plt.tight_layout()
    # plt.show()
    plt.savefig('plots/TS_g_'+interaction+'.png',dpi=200)
    plt.close()

    plt.figure(2,figsize=(5,6))
    ax = plt.gca()
    plt.plot(mphi_list/GeV,[val[0] for val in TS_mphi_null],'-r', label='Null - Energy')
    plt.plot(mphi_list/GeV,[val[1] for val in TS_mphi_null],'--r', label='Null - RA')
    plt.plot(mphi_list/GeV,[val[2] for val in TS_mphi_null],'-.r', label='Null - Dec')
    plt.plot(mphi_list/GeV,[val[0] for val in TS_mphi_DM],'-b', label='DM - Energy')
    plt.plot(mphi_list/GeV,[val[1] for val in TS_mphi_DM],'--b', label='DM - RA')
    plt.plot(mphi_list/GeV,[val[2] for val in TS_mphi_DM],'-.b', label='DM - Dec')
    plt.title(r'g = %0.0e eV, '%(3e-1/GeV)+r'$m_\phi$ = %0.0e eV'%(1e7/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
    plt.ylabel(r'$\sum TS$')
    plt.xlabel(r'$m_\phi$ (GeV)')
    plt.loglog()
    legend = plt.legend(fontsize=14)
    legend.set_title(inter_title[i])
    plt.tick_params(axis='both', which='minor')
    plt.tight_layout()
    # plt.show()
    plt.savefig('plots/TS_mphi_'+interaction+'.png',dpi=200)
    plt.close()

    plt.figure(3,figsize=(5,6))
    ax = plt.gca()
    plt.plot(mx_list/GeV,[val[0] for val in TS_mx_null],'-r', label='Null - Energy')
    plt.plot(mx_list/GeV,[val[1] for val in TS_mx_null],'--r', label='Null - RA')
    plt.plot(mx_list/GeV,[val[2] for val in TS_mx_null],'-.r', label='Null - Dec')
    plt.plot(mx_list/GeV,[val[0] for val in TS_mx_DM],'-b', label='DM - Energy')
    plt.plot(mx_list/GeV,[val[1] for val in TS_mx_DM],'--b', label='DM - RA')
    plt.plot(mx_list/GeV,[val[2] for val in TS_mx_DM],'-.b', label='DM - Dec')
    plt.title(r'g = %0.0e eV, '%(3e-1/GeV)+r'$m_\phi$ = %0.0e eV'%(1e7/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
    plt.ylabel(r'$\sum TS$')
    plt.xlabel(r'$m_\chi$ (GeV)')
    plt.loglog()
    legend = plt.legend(fontsize=14)
    legend.set_title(inter_title[i])
    plt.tick_params(axis='both', which='minor')
    plt.tight_layout()
    # plt.show()
    plt.savefig('plots/TS_mx_'+interaction+'.png',dpi=200)
    plt.close()

# ========= Differents of N_events per bin =========
# g,mphi,mx = [3e-1,1e7,1e8]
# lfn_scalar = BinnedLikelihoodFunction(g,mphi,mx,'scalar')
# diff_perbin = np.abs(lfn_scalar.E_null - lfn_scalar.E_DM)
# plt.step(bin_centers, diff_perbin, where='mid')
# plt.xlim(1e3, 1e7)
# plt.xlabel('Energy (GeV)')
# plt.ylabel(r'$|\Delta N_{events}|$')
# plt.gca().set_yscale("log")
# plt.gca().set_xscale("log")
# plt.tight_layout()
# plt.show()

# ========= Contour plots =========
# from mpl_toolkits.mplot3d import Axes3D
# g_list = np.logspace(-4,0,10)
# mphi_list = np.logspace(6,11,10)
# mx_list = np.logspace(6,10,10)
# X,Y,Z = np.meshgrid(mphi_list/GeV,mx_list/GeV,g_list/GeV)
# TS = [BinnedLikelihoodFunction(g,mphi,mx,'scalar')()[0] for g,mphi,mx in zip(Z.flatten(),X.flatten(),Y.flatten())]
# TS = np.asarray(TS).reshape(X.shape)

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# img  = ax.scatter3D(X, Y, Z, c=np.log10(TS), cmap=matplotlib.colormaps['GnBu'], alpha=0.7, marker='.')
# ax.set_xlabel(r'$m_\phi$ (GeV)')
# ax.set_ylabel(r'$m_\chi$ (GeV)')
# ax.set_zlabel('g (GeV)')
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_zscale('log')
# fig.colorbar(img)
# plt.tight_layout()
# plt.show()

# fig = plt.figure()
# ax = plt.axes(projection="3d")
# img  = ax.scatter3D(X, Y, Z, c=10**np.log10(TS), cmap=plt.jet(), alpha=0.7, marker='.')
# ax.set_xlabel(r'$m_\phi$ (GeV)')
# ax.set_ylabel(r'$m_\chi$ (GeV)')
# ax.set_zlabel('g (GeV)')
# fig.colorbar(img)
# # plt.tight_layout()
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# img = ax.scatter(X, Y, Z, s = 50, c=TS, cmap=plt.jet())
# ax.set_xlabel(r'$m_\phi$ (GeV)')
# ax.set_ylabel(r'$m_\chi$ (GeV)')
# ax.set_zlabel('g (GeV)')
# fig.colorbar(img)
# ax.grid(None)
# plt.show()


# Likelihood Contours
# mphi_list = np.logspace(6,11,10)
# mx_list = np.logspace(6,10,10)
# X,Y = np.meshgrid(mphi_list/GeV,mx_list/GeV)
# TS = [BinnedLikelihoodFunction(3e-1,mphi,mx,'scalar')()[0] for mphi,mx in zip(X.flatten(),Y.flatten())]
# TS = np.asarray(TS).reshape(X.shape)
# plt.contour(X,Y,TS) #,norm=matplotlib.colors.LogNorm()
# plt.colorbar()
# plt.xlabel(r'$m_\phi$ (GeV)')
# plt.ylabel(r'$m_\chi$ (GeV)')
# plt.loglog()
# plt.tight_layout()
# plt.show()


# Maximum likelihood
# import scipy.optimize as opt
#
# g,mphi,mx = [3e-1,1e7,1e8]
# lfn_scalar = BinnedLikelihoodFunction(g,mphi,mx,'scalar')
# result = opt.minimize(lambda x: -lfn_scalar(*x),x0=(1e-1,1e6,1e7),method='Nelder-Mead')
