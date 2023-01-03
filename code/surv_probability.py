from binned_loglikelihood import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker, cm, colors
matplotlib.style.use('../paper.mplstyle')

cl90 = 2.71
cl95 = 3.84
cl99= 6.63

n = 10
interaction = 'scalar'
g_list = np.logspace(-3,0,n) # eV
mphi_list = np.logspace(5,10,n) # eV
mx_list = np.logspace(5,12,n) # eV


# TS arrays
ts_g_null_list = np.array([[BinnedLikelihoodFunction(3e-1,mphi,mx,interaction)()[0] for mx in mx_list] for mphi in mphi_list])
ts_mphi_null_list = np.array([[BinnedLikelihoodFunction(g,1e7,mx,interaction)()[0] for mx in mx_list] for g in g_list])
ts_mx_null_list = np.array([[BinnedLikelihoodFunction(g,mphi,1e8,interaction)()[0] for mphi in mphi_list] for g in g_list])
ts_g_null_list = np.ma.masked_where(ts_g_null_list <= 0,ts_g_null_list)
ts_mphi_null_list = np.ma.masked_where(ts_mphi_null_list <= 0,ts_mphi_null_list)
ts_mx_null_list = np.ma.masked_where(ts_mx_null_list <= 0,ts_mx_null_list)

ts_g_DM_list = np.array([[BinnedLikelihoodFunction(3e-1,mphi,mx,interaction)(3e-1,1e7,1e8,interaction)[0] for mx in mx_list] for mphi in mphi_list])
ts_mphi_DM_list = np.array([[BinnedLikelihoodFunction(g,1e7,mx,interaction)(3e-1,1e7,1e8,interaction)[0] for mx in mx_list] for g in g_list])
ts_mx_DM_list = np.array([[BinnedLikelihoodFunction(g,mphi,1e8,interaction)(3e-1,1e7,1e8,interaction)[0] for mphi in mphi_list] for g in g_list])
ts_g_DM_list = np.ma.masked_where(ts_g_DM_list <= 0,ts_g_DM_list)
ts_mphi_DM_list = np.ma.masked_where(ts_mphi_DM_list <= 0,ts_mphi_DM_list)
ts_mx_DM_list = np.ma.masked_where(ts_mx_DM_list <= 0,ts_mx_DM_list)


# norm = lambda z: colors.LogNorm(vmin=z.min(), vmax=z.max())

mphi, mx = np.meshgrid(mphi_list,mx_list)

fig, ax = plt.subplots(figsize=(8,6))
locator = ticker.LogLocator(base=10)
# cp = ax.pcolormesh(X,Y,Z, cmap="Reds",norm=norm(Z),shading='nearest')
cp = ax.contourf(mphi, mx, ts_g_null_list, locator=locator, cmap="Reds")
cbar  = plt.colorbar(cp, ticks=locator) #, extend='max'
cbar.set_label(r'$-2\Delta LLH$',rotation=90)
# cbar.minorticks_off()
# cbar.ax.set_yscale('log')
plt.loglog()
plt.xlabel(r'$m_\phi$ (GeV)')
plt.ylabel(r'$m_\chi$ (GeV)')
plt.xlim(mphi.min(), mphi.max())
plt.ylim(mx.min(), mx.max())
plt.title('Null')
# plt.show()
plt.savefig('plots/TS_2D_g_null.png',dpi=200)
plt.close()


g, mx = np.meshgrid(g_list,mx_list)

fig, ax = plt.subplots(figsize=(8,6))
locator = ticker.LogLocator(base=10)
# cp = ax.pcolormesh(X,Y,Z, cmap="Reds",norm=norm(Z),shading='nearest')
cp = ax.contourf(g, mx, ts_mphi_null_list, locator=locator, cmap="Reds")
cbar  = plt.colorbar(cp, ticks=locator) #, extend='max'
cbar.set_label(r'$-2\Delta LLH$',rotation=90)
# cbar.minorticks_off()
# cbar.ax.set_yscale('log')
plt.loglog()
plt.xlabel(r'$g$ (GeV)')
plt.ylabel(r'$m_\chi$ (GeV)')
plt.xlim(g.min(), g.max())
plt.ylim(mx.min(), mx.max())
plt.title('Null')
# plt.show()
plt.savefig('plots/TS_2D_mphi_null.png',dpi=200)
plt.close()


g, mphi = np.meshgrid(g_list,mphi_list)

fig, ax = plt.subplots(figsize=(8,6))
locator = ticker.LogLocator(base=10)
# cp = ax.pcolormesh(X,Y,Z, cmap="Reds",norm=norm(Z),shading='nearest')
cp = ax.contourf(g, mphi, ts_mx_null_list, locator=locator, cmap="Reds")
cbar  = plt.colorbar(cp, ticks=locator) #, extend='max'
cbar.set_label(r'$-2\Delta LLH$',rotation=90)
# cbar.minorticks_off()
# cbar.ax.set_yscale('log')
plt.loglog()
plt.xlabel(r'$g$ (GeV)')
plt.ylabel(r'$m_\phi$ (GeV)')
plt.xlim(g.min(), g.max())
plt.ylim(mphi.min(), mphi.max())
plt.title('Null')
# plt.show()
plt.savefig('plots/TS_2D_mx_null.png',dpi=200)
plt.close()



mphi, mx = np.meshgrid(mphi_list,mx_list)

fig, ax = plt.subplots(figsize=(8,6))
locator = ticker.LogLocator(base=10)
# cp = ax.pcolormesh(X,Y,Z, cmap="Blues",norm=norm(Z),shading='nearest')
cp = ax.contourf(mphi, mx, ts_g_null_list, locator=locator, cmap="Blues")
cbar  = plt.colorbar(cp, ticks=locator) #, extend='max'
cbar.set_label(r'$-2\Delta LLH$',rotation=90)
# cbar.minorticks_off()
# cbar.ax.set_yscale('log')
plt.loglog()
plt.xlabel(r'$m_\phi$ (GeV)')
plt.ylabel(r'$m_\chi$ (GeV)')
plt.xlim(mphi.min(), mphi.max())
plt.ylim(mx.min(), mx.max())
plt.title('DM')
# plt.show()
plt.savefig('plots/TS_2D_g_DM.png',dpi=200)
plt.close()


g, mx = np.meshgrid(g_list,mx_list)

fig, ax = plt.subplots(figsize=(8,6))
locator = ticker.LogLocator(base=10)
# cp = ax.pcolormesh(X,Y,Z, cmap="Blues",norm=norm(Z),shading='nearest')
cp = ax.contourf(g, mx, ts_mphi_null_list, locator=locator, cmap="Blues")
cbar  = plt.colorbar(cp, ticks=locator) #, extend='max'
cbar.set_label(r'$-2\Delta LLH$',rotation=90)
# cbar.minorticks_off()
# cbar.ax.set_yscale('log')
plt.loglog()
plt.xlabel(r'$g$ (GeV)')
plt.ylabel(r'$m_\chi$ (GeV)')
plt.xlim(g.min(), g.max())
plt.ylim(mx.min(), mx.max())
plt.title('DM')
# plt.show()
plt.savefig('plots/TS_2D_mphi_DM.png',dpi=200)
plt.close()


g, mphi = np.meshgrid(g_list,mphi_list)

fig, ax = plt.subplots(figsize=(8,6))
locator = ticker.LogLocator(base=10)
# cp = ax.pcolormesh(X,Y,Z, cmap="Blues",norm=norm(Z),shading='nearest')
cp = ax.contourf(g, mphi, ts_mx_null_list, locator=locator, cmap="Blues")
cbar  = plt.colorbar(cp, ticks=locator) #, extend='max'
cbar.set_label(r'$-2\Delta LLH$',rotation=90)
# cbar.minorticks_off()
# cbar.ax.set_yscale('log')
plt.loglog()
plt.xlabel(r'$g$ (GeV)')
plt.ylabel(r'$m_\phi$ (GeV)')
plt.xlim(g.min(), g.max())
plt.ylim(mphi.min(), mphi.max())
plt.title('DM')
# plt.show()
plt.savefig('plots/TS_2D_mx_DM.png',dpi=200)
plt.close()



# Survival Prob arrays

# MC=np.load('../mc/mese_cascades_MC_2013_dnn.npy') # GeV
#
# surv_prob_g = np.array([BinnedLikelihoodFunction(g,1e7,1e8,interaction).AttenuatedFlux(g,1e7,1e8,interaction)[0] for g in g_list])
# surv_prob_mphi = np.array([BinnedLikelihoodFunction(3e-1,mphi,1e8,interaction).AttenuatedFlux(3e-1,mphi,1e8,interaction)[0] for mphi in mphi_list])
# surv_prob_mx = np.array([BinnedLikelihoodFunction(3e-1,1e7,mx,interaction).AttenuatedFlux(3e-1,1e7,mx,interaction)[0]for mx in mx_list])
#
