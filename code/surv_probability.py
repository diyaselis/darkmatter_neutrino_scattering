from binned_loglikelihood import *
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
matplotlib.style.use('../paper.mplstyle')

cl90 = 2.71
cl95 = 3.84
cl99= 6.63

n = 2
interaction = 'scalar'
g_list = np.logspace(-3,0,n) # eV
mphi_list = np.logspace(5,10,n) # eV
mx_list = np.logspace(5,12,n) # eV

def no_neg(array):
    m,n = array.shape
    for i in range(m):
        for j in range(n):
            if array[i][j] < 0:
                array[i][j] = 0
    return array

ts_g_null_list = np.array([[BinnedLikelihoodFunction(3e-1,mphi,mx,interaction)()[0] for mx in mx_list] for mphi in mphi_list])
# ts_mphi_null_list = np.array([[BinnedLikelihoodFunction(g,1e7,mx,interaction)()[0] for mx in mx_list] for g in g_list])
# ts_mx_null_list = np.array([[BinnedLikelihoodFunction(g,mphi,1e8,interaction)()[0] for mphi in mphi_list] for g in g_list])
# ts_g_null_list = no_neg(ts_g_null_list)
# ts_mphi_null_list = no_neg(ts_mphi_null_list)
# ts_mx_null_list = no_neg(ts_mx_null_list)

# ts_g_DM_list = np.array([[BinnedLikelihoodFunction(3e-1,mphi,mx,interaction)(3e-1,1e7,1e8,interaction)[0] for mx in mx_list] for mphi in mphi_list])
# ts_mphi_DM_list = np.array([[BinnedLikelihoodFunction(g,1e7,mx,interaction)(3e-1,1e7,1e8,interaction)[0] for mx in mx_list] for g in g_list])
# ts_mx_DM_list = np.array([[BinnedLikelihoodFunction(g,mphi,1e8,interaction)(3e-1,1e7,1e8,interaction)[0] for mphi in mphi_list] for g in g_list])
# ts_g_DM_list = no_neg(ts_g_DM_list)
# ts_mphi_DM_list = no_neg(ts_mphi_DM_list)
# ts_mx_DM_list = no_neg(ts_mx_DM_list)


norm = lambda z: matplotlib.colors.LogNorm(vmin=z.min(), vmax=z.max())

X, Y = np.meshgrid(mphi_list,mx_list)

Z = np.array([[BinnedLikelihoodFunction(3e-1,X[i][j],Y[i][j],interaction)()[0] for i in range(X.shape[0])] for j in range(X.shape[1])])
Z = np.ma.masked_where(Z <= 0, Z)

fig, ax = plt.subplots()
# lev_exp = np.arange(np.floor(np.log10(Z.min())-1),np.ceil(np.log10(Z.max())+1))
# levs = np.power(10, lev_exp)
# norm=matplotlib.colors.LogNorm()
# locator = ticker.LogLocator()
# extent1 = mphi_list.min(), mphi_list.max(), mx_list.min(), mx_list.max()
# plt.imshow(Z, extent=extent1, norm=norm(Z))
# cp = ax.contourf(X,Y,Z, cmap="bone") #locator = ticker.LogLocator()
cp = ax.contourf(X,Y,Z, locator = ticker.LogLocator(),  cmap="bone")
plt.colorbar(cp)
plt.loglog()
plt.xlabel(r'$m_\phi$ (GeV)')
plt.ylabel(r'$m_\chi$ (GeV)')
plt.show()
# plt.savefig('plots/TS_2D_g.png',dpi=200)
# plt.close()
quit()








plt.figure()
extent2 = g_list.min(), g_list.max(), mx_list.min(), mx_list.max()
plt.imshow(ts_mphi_null_list, extent=extent2, norm=norm(ts_mphi_null_list))
plt.colorbar()
plt.loglog()
plt.xlabel(r'$g$ (GeV)')
plt.ylabel(r'$m_\chi$ (GeV)')
# plt.show()
plt.savefig('plots/TS_2D_mphi.png',dpi=200)
# plt.close()

plt.figure()
extent3 = g_list.min(), g_list.max(), mphi_list.min(), mphi_list.max()
plt.imshow(ts_mx_null_list, extent=extent3, norm=norm(ts_mx_null_list))
plt.colorbar()
plt.loglog()
plt.xlabel(r'$g$ (GeV)')
plt.ylabel(r'$m_\phi$ (GeV)')
# plt.show()
plt.savefig('plots/TS_2D_mx.png',dpi=200)
# plt.close()
quit()

# fig, ax = plt.subplots()
#
# c = ax.pcolormesh(mphi_list/GeV, mx_list/GeV, ts_null_list, cmap='RdBu', norm=matplotlib.colors.LogNorm(vmin=ts_null_list.min(), vmax=ts_null_list.max()))
# # ax.axis([mphi_list.min(), mphi_list.max(), mx_list.min(), mx_list.max()])
# fig.colorbar(c, ax=ax)
# plt.loglog()
# plt.title('Null: g = 1 GeV')
# plt.xlabel(r'$m_\chi$ (GeV)')
# plt.ylabel(r'$m_\phi$ (GeV)')
# plt.show()
# quit()


# plt.figure(dpi=100)
# plt.colorbar()
# plt.loglog()
# plt.title('DM')
# plt.show()
