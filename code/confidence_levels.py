from binned_loglikelihood import *
# import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('../paper.mplstyle')

cl90 = 2.71
cl95 = 3.84
cl99= 6.63

n = 25
interaction = 'scalar'
g_list = np.logspace(-5,0,n) # eV
mphi_list = np.logspace(5,15,n) # eV
mx_list = np.logspace(5,15,n) # eV

TS_g_null = [BinnedLikelihoodFunction(g,1e7,1e8,interaction)() for g in g_list]
TS_mphi_null = [BinnedLikelihoodFunction(3e-1,mphi,1e8,interaction)() for mphi in mphi_list]
TS_mx_null = [BinnedLikelihoodFunction(3e-1,1e7,mx,interaction)() for mx in mx_list]

TS_g_DM = [BinnedLikelihoodFunction(g,1e7,1e8,interaction)(3e-1,1e7,1e8,interaction) for g in g_list]
TS_mphi_DM = [BinnedLikelihoodFunction(3e-1,mphi,1e8,interaction)(3e-1,1e7,1e8,interaction) for mphi in mphi_list]
TS_mx_DM = [BinnedLikelihoodFunction(3e-1,1e7,mx,interaction)(3e-1,1e7,1e8,interaction) for mx in mx_list]


# maximize !!

# ==================== define CL limits ====================
def get_CL_vals(ts_vals,cl_limit):
    indx = np.array([i for i,ts in enumerate(ts_vals) if ts[0] <= cl_limit])
    vals = np.array([ts[0] for i,ts in enumerate(ts_vals) if ts[0] <= cl_limit])
    if indx.size == 0:
        return np.array([None]),np.array([None])
    return indx,vals

# --------------   Null  ----------------
g_null_CL90 = get_CL_vals(TS_g_null,cl90)
g_null_CL95 = get_CL_vals(TS_g_null,cl95)
g_null_CL99 = get_CL_vals(TS_g_null,cl99)

mphi_null_CL90 = get_CL_vals(TS_mphi_null,cl90)
mphi_null_CL95 = get_CL_vals(TS_mphi_null,cl95)
mphi_null_CL99 = get_CL_vals(TS_mphi_null,cl99)

mx_null_CL90 = get_CL_vals(TS_mx_null,cl90)
mx_null_CL95 = get_CL_vals(TS_mx_null,cl95)
mx_null_CL99 = get_CL_vals(TS_mx_null,cl99)

# ----------------   DM  ------------------
g_DM_CL90 = get_CL_vals(TS_g_DM,cl90)
g_DM_CL95 = get_CL_vals(TS_g_DM,cl95)
g_DM_CL99 = get_CL_vals(TS_g_DM,cl99)

mphi_DM_CL90 = get_CL_vals(TS_mphi_DM,cl90)
mphi_DM_CL95 = get_CL_vals(TS_mphi_DM,cl95)
mphi_DM_CL99 = get_CL_vals(TS_mphi_DM,cl99)

mx_DM_CL90 = get_CL_vals(TS_mx_DM,cl90)
mx_DM_CL95 = get_CL_vals(TS_mx_DM,cl95)
mx_DM_CL99 = get_CL_vals(TS_mx_DM,cl99)

# ========================================================

# MC=np.load('../mc/mese_cascades_MC_2013_dnn.npy') # GeV
#
# surv_prob_g = np.array([BinnedLikelihoodFunction(g,1e7,1e8,interaction).AttenuatedFlux(g,1e7,1e8,interaction)[0] for g in g_list])
# surv_prob_mphi = np.array([BinnedLikelihoodFunction(3e-1,mphi,1e8,interaction).AttenuatedFlux(3e-1,mphi,1e8,interaction)[0] for mphi in mphi_list])
# surv_prob_mx = np.array([BinnedLikelihoodFunction(3e-1,1e7,mx,interaction).AttenuatedFlux(3e-1,1e7,mx,interaction)[0]for mx in mx_list])
#
#
# plt.figure(1,dpi=150,figsize=(8,6))
# ax = plt.gca()
# plt.plot(g_list,surv_prob_g)
# plt.tight_layout()
# plt.title(r'E = %0.0e GeV, '%(MC['energy'][0])+r'$m_\phi$ = %0.0e eV'%(1e7)+r', $m_\chi$ = %0.0e eV'%(1e8)) #r'g = %0.0e eV, '%(3e-1)+
# plt.ylabel(r'$\mathcal{P}_{surv}$')
# plt.xlabel('g (eV)')
# plt.loglog()
# plt.tick_params(axis='both', which='minor')
# # plt.show()
# plt.savefig('plots/surv_prob_g.png',dpi=200)
# # plt.close()
#
# plt.figure(2,dpi=150,figsize=(8,6))
# plt.plot(mphi_list,surv_prob_mphi)
# plt.tight_layout()
# plt.title(r'E = %0.0e GeV, '%(MC['energy'][0])+r'g = %0.0e eV, '%(3e-1)+r', $m_\chi$ = %0.0e eV'%(1e8)) #r'g = %0.0e eV, '%(3e-1)+
# plt.ylabel(r'$\mathcal{P}_{surv}$')
# plt.xlabel(r' $m_\phi$  (eV)')
# plt.loglog()
# plt.tick_params(axis='both', which='minor')
# # plt.show()
# plt.savefig('plots/surv_prob_mphi.png',dpi=200)
# # plt.close()
#
# plt.figure(3,dpi=150,figsize=(8,6))
# plt.plot(mx_list,surv_prob_mx)
# plt.tight_layout()
# plt.title(r'E = %0.0e GeV, '%(MC['energy'][0])+r'g = %0.0e eV, '%(3e-1)+r'$m_\phi$ = %0.0e eV'%(1e7)) #r'g = %0.0e eV, '%(3e-1)+
# plt.ylabel(r'$\mathcal{P}_{surv}$')
# plt.xlabel(r' $m_\chi$  (eV)')
# plt.loglog()
# plt.tick_params(axis='both', which='minor')
# # plt.show()
# plt.savefig('plots/surv_prob_mx.png',dpi=200)
# # plt.close()
# quit()
#
#
# # ===========  sensitivity plots ============
#
# g = 3e-1
# cl_null_list = []
# cl_DM_list = []
# for mphi in mphi_list:
#     for mx in mx_list:
#         ts = BinnedLikelihoodFunction(g,mphi,mx,interaction)()
#         ts_DM = BinnedLikelihoodFunction(g,mphi,mx,interaction)(3e-1,1e7,1e8,interaction)
#         if ts[0] <= cl90:
#             cl_null_list.append([mphi,mx,ts[0]])
#         if ts_DM[0] <= cl90:
#             cl_DM_list.append([mphi,mx,ts[0]])
# cl_null_list = np.array(cl_null_list)
# cl_DM_list = np.array(cl_DM_list)
#
# plt.figure(1)
# plt.plot(cl_null_list[:,0]/GeV,cl_null_list[:,1]/GeV, label='Null')
# plt.plot(cl_DM_list[:,0]/GeV,cl_DM_list[:,1]/GeV, label='DM')
# plt.legend()
# plt.loglog()
# plt.xlabel(r'$m_\phi$ (GeV)')
# plt.ylabel(r'$m_\chi$ (GeV)')
# plt.title(r'$g = 3 \times 10^{-10}$ (GeV)')
# # plt.show()
# plt.savefig('plots/CL_combined_mphi_mx.png',dpi=200)
# # plt.close()
#
#
# mphi = 1e7
# cl_null_list = []
# cl_DM_list = []
# for g in g_list:
#     for mx in mx_list:
#         ts = BinnedLikelihoodFunction(g,mphi,mx,interaction)()
#         ts_DM = BinnedLikelihoodFunction(g,mphi,mx,interaction)(3e-1,1e7,1e8,interaction)
#         if ts[0] <= cl90:
#             cl_null_list.append([g,mx,ts[0]])
#         if ts_DM[0] <= cl90:
#             cl_DM_list.append([g,mx,ts[0]])
# cl_null_list = np.array(cl_null_list)
# cl_DM_list = np.array(cl_DM_list)
#
# plt.figure(2)
# plt.plot(cl_null_list[:,0]/GeV,cl_null_list[:,1]/GeV, label='Null')
# plt.plot(cl_DM_list[:,0]/GeV,cl_DM_list[:,1]/GeV, label='DM')
# plt.legend()
# plt.loglog()
# plt.xlabel(r'$g$ (GeV)')
# plt.ylabel(r'$m_\chi$ (GeV)')
# plt.title(r'$m_\phi = 10^{-2}$ (GeV)')
# # plt.show()
# plt.savefig('plots/CL_combined_g_mx.png',dpi=200)
# # plt.close()
#
#
# mx = 1e8
# cl_null_list = []
# cl_DM_list = []
# for g in g_list:
#     for mphi in mphi_list:
#         ts = BinnedLikelihoodFunction(g,mphi,mx,interaction)()
#         ts_DM = BinnedLikelihoodFunction(g,mphi,mx,interaction)(3e-1,1e7,1e8,interaction)
#         if ts[0] <= cl90:
#             cl_null_list.append([g,mphi,ts[0]])
#         if ts_DM[0] <= cl90:
#             cl_DM_list.append([g,mphi,ts[0]])
# cl_null_list = np.array(cl_null_list)
# cl_DM_list = np.array(cl_DM_list)
#
# plt.figure(3)
# plt.plot(cl_null_list[:,0]/GeV,cl_null_list[:,1]/GeV, label='Null')
# plt.plot(cl_DM_list[:,0]/GeV,cl_DM_list[:,1]/GeV, label='DM')
# plt.legend()
# plt.loglog()
# plt.xlabel(r'$g$ (GeV)')
# plt.ylabel(r'$m_\phi$ (GeV)')
# plt.title(r'$m_\chi = 10^{-1}$ (GeV)')
# # plt.show()
# plt.savefig('plots/CL_combined_g_mphi.png',dpi=200)
# # plt.close()
#
#
# quit()

# ============= CL levels for TS ===============
# plt.figure(1,dpi=150,figsize=(8,6))
# ax = plt.gca()
# plt.plot(g_list/GeV,[val[0] for val in TS_g_null],'-r')
# plt.fill_between(g_list[g_null_CL90[0]]/GeV, g_null_CL90[1],color='red',alpha=0.1,label=r'90\% CL')
# plt.fill_between(g_list[g_null_CL95[0]]/GeV, g_null_CL95[1] ,color='red',alpha=0.3,label=r'95\% CL')
# plt.fill_between(g_list[g_null_CL99[0]]/GeV, g_null_CL99[1] ,color='red',alpha=0.5,label=r'99\% CL')
# plt.tight_layout()
# plt.title(r'$m_\phi$ = %0.0e eV'%(1e7/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
# plt.ylabel(r'$-2\Delta LLH$')
# plt.xlabel('g (GeV)')
# # plt.loglog()
# plt.semilogx()
# plt.tight_layout()
# legend = plt.legend(fontsize=14)
# legend.set_title('Null')
# plt.tick_params(axis='both', which='minor')
# # plt.show()
# plt.savefig('plots/TS_g_null_CL.png',dpi=200)
# # plt.close()
#
# plt.figure(2,dpi=150,figsize=(8,6))
# ax = plt.gca()
# plt.plot(mphi_list/GeV,[val[0] for val in TS_mphi_null],'-r')
# plt.fill_between(mphi_list[mphi_null_CL90[0]]/GeV, mphi_null_CL90[1],color='red',alpha=0.1,label=r'90\% CL')
# plt.fill_between(mphi_list[mphi_null_CL95[0]]/GeV, mphi_null_CL95[1],color='red',alpha=0.3,label=r'95\% CL')
# plt.fill_between(mphi_list[mphi_null_CL99[0]]/GeV, mphi_null_CL99[1],color='red',alpha=0.5,label=r'99\% CL')
# plt.tight_layout()
# plt.title(r'g = %0.0e eV, '%(3e-1/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
# plt.ylabel(r'$-2\Delta LLH$')
# plt.xlabel(r'$m_\phi$ (GeV)')
# # plt.loglog()
# plt.semilogx()
# plt.tight_layout()
# legend = plt.legend(fontsize=14)
# legend.set_title('Null')
# plt.tick_params(axis='both', which='minor')
# # plt.show()
# plt.savefig('plots/TS_mphi_null_CL.png',dpi=200)
# # plt.close()
#
# plt.figure(3,dpi=150,figsize=(8,6))
# ax = plt.gca()
# plt.plot(mx_list/GeV,[val[0] for val in TS_mx_null],'-r')
# plt.fill_between(mx_list[mx_null_CL90[0]]/GeV, mx_null_CL90[1],color='red',alpha=0.1,label=r'90\% CL')
# plt.fill_between(mx_list[mx_null_CL95[0]]/GeV, mx_null_CL95[1],color='red',alpha=0.3,label=r'95\% CL')
# plt.fill_between(mx_list[mx_null_CL99[0]]/GeV, mx_null_CL99[1],color='red',alpha=0.5,label=r'99\% CL')
# plt.tight_layout()
# plt.title(r'g = %0.0e eV, '%(3e-1/GeV)+r'$m_\phi$ = %0.0e eV'%(1e7/GeV))
# plt.ylabel(r'$-2\Delta LLH$')
# plt.xlabel(r'$m_\chi$ (GeV)')
# # plt.loglog()
# plt.semilogx()
# plt.tight_layout()
# legend = plt.legend(fontsize=14)
# legend.set_title('Null')
# plt.tick_params(axis='both', which='minor')
# # plt.show()
# plt.savefig('plots/TS_mx_null_CL.png',dpi=200)
# # plt.close()

plt.figure(4,dpi=150,figsize=(8,6))
ax = plt.gca()
plt.axvline(3e-1/GeV,c='b',ls='--',label='True')
plt.plot(g_list/GeV,[val[0] for val in TS_g_DM],'-b')
plt.fill_between(g_list[g_DM_CL90[0]]/GeV, g_DM_CL90[1],color='blue',alpha=0.1,label=r'90\% CL')
plt.fill_between(g_list[g_DM_CL95[0]]/GeV, g_DM_CL95[1],color='blue',alpha=0.3,label=r'95\% CL')
plt.fill_between(g_list[g_DM_CL99[0]]/GeV, g_DM_CL99[1],color='blue',alpha=0.5,label=r'99\% CL')
plt.tight_layout()
plt.title(r'$m_\phi$ = %0.0e eV'%(1e7/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
plt.ylabel(r'$-2\Delta LLH$')
plt.xlabel('g (GeV)')
plt.loglog()
# plt.semilogx()
plt.tight_layout()
legend = plt.legend(fontsize=14)
legend.set_title('DM')
plt.tick_params(axis='both', which='minor')
# plt.show()
plt.savefig('plots/TS_g_DM_CL.png',dpi=200)
# plt.close()

plt.figure(5,dpi=150,figsize=(8,6))
ax = plt.gca()
plt.axvline(1e7/GeV,c='b',ls='--',label='True')
plt.plot(mphi_list/GeV,[val[0] for val in TS_mphi_DM],'-b')
plt.fill_between(mphi_list[mphi_DM_CL90[0]]/GeV, mphi_DM_CL90[1],color='blue',alpha=0.1,label=r'90\% CL')
plt.fill_between(mphi_list[mphi_DM_CL95[0]]/GeV, mphi_DM_CL95[1],color='blue',alpha=0.3,label=r'95\% CL')
plt.fill_between(mphi_list[mphi_DM_CL99[0]]/GeV, mphi_DM_CL99[1],color='blue',alpha=0.5,label=r'99\% CL')
plt.tight_layout()
plt.title(r'g = %0.0e eV, '%(3e-1/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
plt.ylabel(r'$-2\Delta LLH$')
plt.xlabel(r'$m_\phi$ (GeV)')
plt.loglog()
# plt.semilogx()
legend = plt.legend(fontsize=14)
legend.set_title('DM')
plt.tick_params(axis='both', which='minor')
# plt.show()
plt.savefig('plots/TS_mphi_DM_CL.png',dpi=200)
# plt.close()

plt.figure(6,dpi=150,figsize=(8,6))
ax = plt.gca()
plt.axvline(1e8/GeV,c='b',ls='--',label='True')
plt.plot(mx_list/GeV,[val[0] for val in TS_mx_DM],'-b')
plt.fill_between(mx_list[mx_DM_CL90[0]]/GeV, mx_DM_CL90[1],color='blue',alpha=0.1,label=r'90\% CL')
plt.fill_between(mx_list[mx_DM_CL95[0]]/GeV, mx_DM_CL95[1],color='blue',alpha=0.3,label=r'95\% CL')
plt.fill_between(mx_list[mx_DM_CL99[0]]/GeV, mx_DM_CL99[1],color='blue',alpha=0.5,label=r'99\% CL')
plt.tight_layout()
plt.title(r'g = %0.0e eV, '%(3e-1/GeV)+r'$m_\phi$ = %0.0e eV'%(1e7/GeV))
plt.ylabel(r'$-2\Delta LLH$')
plt.xlabel(r'$m_\chi$ (GeV)')
plt.loglog()
# plt.semilogx()
plt.tight_layout()
legend = plt.legend(fontsize=14)
legend.set_title('DM')
plt.tick_params(axis='both', which='minor')
# plt.show()
plt.savefig('plots/TS_mx_DM_CL.png',dpi=200)
# plt.close()



quit()
