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
# interaction = 'scalar'
interactions = {'scalar':r'Scalar $\chi$ - Scalar $\phi$','fermion':r'Fermion $\chi$ - Scalar $\phi$',
                'vector':r'Fermion $\chi$ - Vector $\phi$','fermscal':r'Scalar $\chi$ - Fermion $\phi$'}
g_list = np.logspace(-3,0,n) # eV
mphi_list = np.logspace(5,10,n) # eV
mx_list = np.logspace(5,12,n) # eV

# Survival Prob arrays

MC = np.load('../mc/mese_cascades_MC_2013_dnn.npy') # GeV
for interaction in interactions.keys():
    surv_prob_g_list = np.array([[BinnedLikelihoodFunction(3e-1,mphi,mx,interaction).SurvivalProbability(3e-1,mphi,mx,interaction)[0] for mx in mx_list] for mphi in mphi_list])
    surv_prob_mphi_list = np.array([[BinnedLikelihoodFunction(g,1e7,mx,interaction).SurvivalProbability(g,1e7,mx,interaction)[0] for mx in mx_list] for g in g_list])
    surv_prob_mx_list = np.array([[BinnedLikelihoodFunction(g,mphi,1e8,interaction).SurvivalProbability(g,mphi,1e8,interaction)[0] for mphi in mphi_list] for g in g_list])
    surv_prob_g_list = np.ma.masked_where(surv_prob_g_list <= 0,surv_prob_g_list)
    surv_prob_mphi_list = np.ma.masked_where(surv_prob_mphi_list <= 0,surv_prob_mphi_list)
    surv_prob_mx_list = np.ma.masked_where(surv_prob_mx_list <= 0,surv_prob_mx_list)


    mphi, mx = np.meshgrid(mphi_list,mx_list)

    fig, ax = plt.subplots(figsize=(8,7))
    locator = ticker.LogLocator(base=10)
    # cp = ax.pcolormesh(X,Y,Z, cmap="Reds",norm=norm(Z),shading='nearest')
    cp = ax.contourf(mphi, mx, surv_prob_g_list, locator=locator, cmap="Reds")
    cbar  = plt.colorbar(cp, ticks=locator) #, extend='max'
    cbar.set_label(r'$\exp (\sigma \tau / \m_\chi)$',rotation=90)
    # cbar.minorticks_off()
    # cbar.ax.set_yscale('log')
    plt.loglog()
    plt.xlabel(r'$m_\phi$ (GeV)')
    plt.ylabel(r'$m_\chi$ (GeV)')
    plt.xlim(mphi.min(), mphi.max())
    plt.ylim(mx.min(), mx.max())
    plt.title('Survival Probability: '+interactions[interaction])
    # plt.show()
    plt.savefig('plots/surv_prob_2D_g_'+interaction+'.png',dpi=200)
    plt.close()


    g, mx = np.meshgrid(g_list,mx_list)

    fig, ax = plt.subplots(figsize=(8,7))
    locator = ticker.LogLocator(base=10)
    # cp = ax.pcolormesh(X,Y,Z, cmap="Reds",norm=norm(Z),shading='nearest')
    cp = ax.contourf(g, mx, surv_prob_mphi_list, locator=locator, cmap="Reds")
    cbar  = plt.colorbar(cp, ticks=locator) #, extend='max'
    cbar.set_label(r'$\exp (\sigma \tau / \m_\chi)$',rotation=90)
    # cbar.minorticks_off()
    # cbar.ax.set_yscale('log')
    plt.loglog()
    plt.xlabel(r'$g$ (GeV)')
    plt.ylabel(r'$m_\chi$ (GeV)')
    plt.xlim(g.min(), g.max())
    plt.ylim(mx.min(), mx.max())
    plt.title('Survival Probability: '+interactions[interaction])
    # plt.show()
    plt.savefig('plots/surv_prob_2D_mphi_'+interaction+'.png',dpi=200)
    plt.close()


    g, mphi = np.meshgrid(g_list,mphi_list)

    fig, ax = plt.subplots(figsize=(8,7))
    locator = ticker.LogLocator(base=10)
    # cp = ax.pcolormesh(X,Y,Z, cmap="Reds",norm=norm(Z),shading='nearest')
    cp = ax.contourf(g, mphi, surv_prob_mx_list, locator=locator, cmap="Reds")
    cbar  = plt.colorbar(cp, ticks=locator) #, extend='max'
    cbar.set_label(r'$\exp (\sigma \tau / \m_\chi)$',rotation=90)
    # cbar.minorticks_off()
    # cbar.ax.set_yscale('log')
    plt.loglog()
    plt.xlabel(r'$g$ (GeV)')
    plt.ylabel(r'$m_\phi$ (GeV)')
    plt.xlim(g.min(), g.max())
    plt.ylim(mphi.min(), mphi.max())
    plt.title('Survival Probability: '+interactions[interaction])
    # plt.show()
    plt.savefig('plots/surv_prob_2D_mx_'+interaction+'.png',dpi=200)
    plt.close()

quit()

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
