from unbinned_loglikelihood import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('../paper.mplstyle')


# ========= TS scan over one DM parameter =========
inters = ['scalar','fermion','vector','fermscal']
inter_title = ['Scalar-Scalar DM','Scalar-Fermion DM','Vector-Fermion DM','Fermion-Scalar DM']
for i,interaction in enumerate(inters):
    g_vals = np.logspace(-4,0)
    g_coords = [[np.log10(g),7.,8.] for g in g_vals]
    mphi_vals = np.logspace(6,11)
    mphi_coords = [[np.log10(3e-1),np.log10(mphi),8.] for mphi in mphi_vals]
    mx_vals = np.logspace(6,10)
    mx_coords = [[np.log10(3e-1),7., np.log10(mx)] for mx in mx_vals]

    TS_g_null = np.array([UnbinnedLikelihoodFunction(theta,interaction).TestStatistics() for theta in g_coords])
    TS_mphi_null = np.array([UnbinnedLikelihoodFunction(theta,interaction).TestStatistics() for theta in mphi_coords])
    TS_mx_null = np.array([UnbinnedLikelihoodFunction(theta,interaction).TestStatistics() for theta in mx_coords])

    TS_g_DM = np.array([UnbinnedLikelihoodFunction(theta,interaction).TestStatistics(np.log10(3e-1),7,8) for theta in mx_coords])
    TS_mphi_DM = np.array([UnbinnedLikelihoodFunction(theta,interaction).TestStatistics(np.log10(3e-1),7,8) for theta in mx_coords])
    TS_mx_DM = np.array([UnbinnedLikelihoodFunction(theta,interaction).TestStatistics(np.log10(3e-1),7,8) for theta in mx_coords])

    plt.figure(1,figsize=(5,6))
    ax = plt.gca()
    plt.plot(g_vals/GeV,TS_g_null,'-r', label='Null ')
    plt.plot(g_vals/GeV,TS_g_DM,'-b', label='DM')
    plt.title(r'g = %0.0e eV, '%(3e-1/GeV)+r'$m_\phi$ = %0.0e eV'%(1e7/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
    plt.ylabel(r'$\sum TS$')
    plt.xlabel('g (GeV)')
    plt.loglog()
    legend = plt.legend(fontsize='small',fancybox=True)
    legend.set_title(inter_title[i])
    plt.tick_params(axis='both', which='minor')
    plt.tight_layout()
    # plt.show()
    plt.savefig('plots/TS_g_'+interaction+'unbinned.png',dpi=200)
    plt.close()
    
    plt.figure(2,figsize=(5,6))
    ax = plt.gca()
    plt.plot(mphi_vals/GeV,TS_mphi_null,'-r', label='Null')
    plt.plot(mphi_vals/GeV,TS_mphi_DM,'-b', label='DM')
    plt.title(r'g = %0.0e eV, '%(3e-1/GeV)+r'$m_\phi$ = %0.0e eV'%(1e7/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
    plt.ylabel(r'$\sum TS$')
    plt.xlabel(r'$m_\phi$ (GeV)')
    plt.loglog()
    legend = plt.legend(fontsize='small',fancybox=True)
    legend.set_title(inter_title[i])
    plt.tick_params(axis='both', which='minor')
    plt.tight_layout()
    # plt.show()
    plt.savefig('plots/TS_mphi_'+interaction+'unbinned.png',dpi=200)
    plt.close()

    plt.figure(3,figsize=(5,6))
    ax = plt.gca()
    plt.plot(mx_vals/GeV,TS_mx_null,'-r', label='Null - Energy')
    plt.plot(mx_vals/GeV,TS_mx_DM,'-b', label='DM')
    plt.title(r'g = %0.0e eV, '%(3e-1/GeV)+r'$m_\phi$ = %0.0e eV'%(1e7/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
    plt.ylabel(r'$\sum TS$')
    plt.xlabel(r'$m_\chi$ (GeV)')
    plt.loglog()
    legend = plt.legend()
    legend.set_title(inter_title[i])
    plt.tick_params(axis='both', which='minor')
    plt.tight_layout()
    # plt.show()
    plt.savefig('plots/TS_mx_'+interaction+'unbinned.png',dpi=200)
    plt.close()

# ========= class testing =========
# theta = [-1,7,8]
# lfn_scalar = UnbinnedLikelihoodFunction(theta,'scalar')
# ts_null_scalar = lfn_scalar.TestStatistics() # E
# ts_DM_scalar = lfn_scalar.TestStatistics(-1,7,8) # E
# logprob_scalar = lfn_scalar.LogProbability()
# print(ts_null_scalar,ts_DM_scalar,logprob_scalar)

# ========= TS scan over one DM parameter =========
# g_vals = np.logspace(-3,0)
# g_coords = [[np.log10(g),7.,8.] for g in g_vals]
# g_TS_null = np.array([UnbinnedLikelihoodFunction(theta,'scalar').TestStatistics() for theta in g_coords])
# g_TS_DM = np.array([UnbinnedLikelihoodFunction(theta,'scalar').TestStatistics(-1,7,8) for theta in g_coords])
#
# mphi_vals = np.logspace(3,8)
# mphi_coords = [[np.log10(3e-1),np.log10(mphi),8.] for mphi in mphi_vals]
# mphi_TS_null = np.array([UnbinnedLikelihoodFunction(theta,'scalar').TestStatistics() for theta in mphi_coords])
# mphi_TS_DM = np.array([UnbinnedLikelihoodFunction(theta,'scalar').TestStatistics(-1,7,8) for theta in mphi_coords])
#
# mx_vals = np.logspace(3,8)
# mx_coords = [[np.log10(3e-1),7., np.log10(mx)] for mx in mx_vals]
# mx_TS_null = np.array([UnbinnedLikelihoodFunction(theta,'scalar').TestStatistics() for theta in mx_coords])
# mx_TS_DM = np.array([UnbinnedLikelihoodFunction(theta,'scalar').TestStatistics(-1,7,8) for theta in mx_coords])
#
#
# plt.figure(1,dpi=150)
# plt.plot(g_vals, g_TS_null, label='Null Hypo')
# plt.plot(g_vals, g_TS_DM, label='DM Hypo (g = %0.0e eV)'%(3e-1))
# plt.xlabel('g (GeV)',fontsize=18)
# plt.ylabel(r'$-2\Delta$LLH',fontsize=18)
# title = r'$m_\phi = $  '+ '{:.0e} GeV,   '.format(10**g_coords[0][1]) + r'  $m_\chi = $  '+ '{:.0e} GeV'.format(10**g_coords[0][2])
# plt.title(title,fontsize=18)
# plt.tick_params(axis='both', which='minor')
# plt.tight_layout()
# plt.loglog()
# plt.legend()
# plt.show()
#
#
# plt.figure(2,dpi=150)
# plt.plot(mphi_vals, mphi_TS_null, label='Null Hypo')
# plt.plot(mphi_vals, mphi_TS_DM, label='DM Hypo ($m_\phi$ = %0.0e eV)'%(1e7))
# plt.xlabel(r'$m_\phi$ (GeV)',fontsize=18)
# plt.ylabel(r'$-2\Delta$LLH',fontsize=18)
# title = r'$g = $  '+ '{:.0e},   '.format(10**mphi_coords[0][0]) + r'  $m_\chi = $  '+ '{:.0e} GeV'.format(10**mphi_coords[0][2])
# plt.title(title,fontsize=18)
# plt.tick_params(axis='both', which='minor')
# plt.tight_layout()
# plt.loglog()
# plt.legend()
# plt.show()
#
# plt.figure(3,dpi=150)
# plt.plot(mx_vals, mx_TS_null, label='Null Hypo')
# plt.plot(mx_vals, mx_TS_DM, label='DM Hypo ($m_\chi$ = %0.0e eV)'%(1e8))
# plt.xlabel(r'$m_\chi$ (GeV)',fontsize=18)
# plt.ylabel(r'$-2\Delta$LLH',fontsize=18)
# title =  r'$g = $  '+ '{:.0e},   '.format(10**mx_coords[0][0]) + r'$m_\phi = $  '+ '{:.0e} GeV,   '.format(10**mx_coords[0][1])
# plt.title(title,fontsize=18)
# plt.tick_params(axis='both', which='minor')
# plt.tight_layout()
# plt.loglog()
# plt.legend()
# plt.show()
