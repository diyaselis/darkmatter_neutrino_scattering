from unbinned_loglikelihood import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('../paper.mplstyle')

# ========= class testing =========
# theta = [-1,7,8]
# lfn_scalar = UnbinnedLikelihoodFunction(theta,'scalar')
# ts_null_scalar = lfn_scalar.TestStatistics() # E
# ts_DM_scalar = lfn_scalar.TestStatistics(-1,7,8) # E
# logprob_scalar = lfn_scalar.LogProbability()
# print(ts_null_scalar,ts_DM_scalar,logprob_scalar)

# ========= TS scan over one DM parameter =========
g_vals = np.logspace(-3,0)
g_coords = [[np.log10(g),7.,8.] for g in g_vals]
g_TS_null = np.array([UnbinnedLikelihoodFunction(theta,'scalar').TestStatistics() for theta in g_coords])
g_TS_DM = np.array([UnbinnedLikelihoodFunction(theta,'scalar').TestStatistics(-1,7,8) for theta in g_coords])

mphi_vals = np.logspace(3,8)
mphi_coords = [[np.log10(3e-1),np.log10(mphi),8.] for mphi in mphi_vals]
mphi_TS_null = np.array([UnbinnedLikelihoodFunction(theta,'scalar').TestStatistics() for theta in mphi_coords])
mphi_TS_DM = np.array([UnbinnedLikelihoodFunction(theta,'scalar').TestStatistics(-1,7,8) for theta in mphi_coords])

mx_vals = np.logspace(3,8)
mx_coords = [[np.log10(3e-1),7., np.log10(mx)] for mx in mx_vals]
mx_TS_null = np.array([UnbinnedLikelihoodFunction(theta,'scalar').TestStatistics() for theta in mx_coords])
mx_TS_DM = np.array([UnbinnedLikelihoodFunction(theta,'scalar').TestStatistics(-1,7,8) for theta in mx_coords])


plt.figure(1,dpi=150)
plt.plot(g_vals, g_TS_null, label='Null Hypo')
plt.plot(g_vals, g_TS_DM, label='DM Hypo (g = %0.0e eV)'%(3e-1))
plt.xlabel('g (GeV)',fontsize=18)
plt.ylabel(r'$-2\Delta$LLH',fontsize=18)
title = r'$m_\phi = $  '+ '{:.0e} GeV,   '.format(10**g_coords[0][1]) + r'  $m_\chi = $  '+ '{:.0e} GeV'.format(10**g_coords[0][2])
plt.title(title,fontsize=18)
plt.tick_params(axis='both', which='minor')
plt.tight_layout()
plt.loglog()
plt.legend()
plt.show()


plt.figure(2,dpi=150)
plt.plot(mphi_vals, mphi_TS_null, label='Null Hypo')
plt.plot(mphi_vals, mphi_TS_DM, label='DM Hypo ($m_\phi$ = %0.0e eV)'%(1e7))
plt.xlabel(r'$m_\phi$ (GeV)',fontsize=18)
plt.ylabel(r'$-2\Delta$LLH',fontsize=18)
title = r'$g = $  '+ '{:.0e},   '.format(10**mphi_coords[0][0]) + r'  $m_\chi = $  '+ '{:.0e} GeV'.format(10**mphi_coords[0][2])
plt.title(title,fontsize=18)
plt.tick_params(axis='both', which='minor')
plt.tight_layout()
plt.loglog()
plt.legend()
plt.show()

plt.figure(3,dpi=150)
plt.plot(mx_vals, mx_TS_null, label='Null Hypo')
plt.plot(mx_vals, mx_TS_DM, label='DM Hypo ($m_\chi$ = %0.0e eV)'%(1e8))
plt.xlabel(r'$m_\chi$ (GeV)',fontsize=18)
plt.ylabel(r'$-2\Delta$LLH',fontsize=18)
title =  r'$g = $  '+ '{:.0e},   '.format(10**mx_coords[0][0]) + r'$m_\phi = $  '+ '{:.0e} GeV,   '.format(10**mx_coords[0][1])
plt.title(title,fontsize=18)
plt.tick_params(axis='both', which='minor')
plt.tight_layout()
plt.loglog()
plt.legend()
plt.show()


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
