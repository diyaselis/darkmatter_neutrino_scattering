from binned_loglikelihood import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('../paper.mplstyle')

# class testing
g,mphi,mx = [3e-1,1e7,1e8]
lfn_scalar = BinnedLikelihoodFunction(g,mphi,mx,'scalar')
ts_null_scalar = lfn_scalar()
print(ts_null_scalar)

# TS scan over one DM parameter
g_list = np.logspace(-4,0)
mphi_list = np.logspace(6,11)
mx_list = np.logspace(6,10)

TS_g_null = [BinnedLikelihoodFunction(g,1e7,1e8,'scalar')() for g in g_list]
TS_mphi_null = [BinnedLikelihoodFunction(3e-1,mphi,1e8,'scalar')() for mphi in mphi_list]
TS_mx_null = [BinnedLikelihoodFunction(3e-1,1e7,mx,'scalar')() for mx in mx_list]

TS_g_DM = [BinnedLikelihoodFunction(g,1e7,1e8,'scalar')(3e-1,1e7,1e8,'scalar') for g in g_list]
TS_mphi_DM = [BinnedLikelihoodFunction(3e-1,mphi,1e8,'scalar')(3e-1,1e7,1e8,'scalar') for mphi in mphi_list]
TS_mx_DM = [BinnedLikelihoodFunction(3e-1,1e7,mx,'scalar')(3e-1,1e7,1e8,'scalar') for mx in mx_list]

plt.figure(1)
ax = plt.gca()
plt.plot(g_list/GeV,TS_g_null, label='Null Hypo')
plt.plot(g_list/GeV,TS_g_DM, label='DM Hypo (g = %0.0e eV)'%(3e-1/GeV))
plt.title(r'$m_\phi$ = %0.0e eV'%(1e7/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
plt.ylabel(r'$\sum TS$')
plt.xlabel('g (GeV)')
plt.loglog()
plt.legend()
plt.tick_params(axis='both', which='minor')
plt.tight_layout()
plt.show()

plt.figure(2)
ax = plt.gca()
plt.plot(mphi_list/GeV,TS_mphi_null, label='Null Hypo')
plt.plot(mphi_list/GeV,TS_mphi_DM, label=r'DM Hypo ($m_\phi$ = %0.0e eV)'%(1e7/GeV))
plt.title('g = %0.0e eV'%(3e-1/GeV)+r', $m_\chi$ = %0.0e eV'%(1e8/GeV))
plt.ylabel(r'$\sum TS$')
plt.xlabel(r'$m_\phi$ (GeV)')
plt.loglog()
plt.legend()
plt.tick_params(axis='both', which='minor')
plt.tight_layout()
plt.show()

plt.figure(3)
ax = plt.gca()
plt.plot(mx_list/GeV,TS_mx_null, label='Null Hypo')
plt.plot(mx_list/GeV,TS_mx_DM, label='DM Hypo ($m_\chi$ = %0.0e eV)'%(1e8/GeV))
plt.title('g = %0.0e eV'%(3e-1/GeV)+r', $m_\phi$ = %0.0e eV'%(1e7/GeV))
plt.ylabel(r'$\sum TS$')
plt.xlabel(r'$m_\chi$ (GeV)')
plt.loglog()
plt.legend()
plt.tick_params(axis='both', which='minor')
plt.tight_layout()
plt.show()


# Differents of N_events per bin
g,mphi,mx = [3e-1,1e7,1e8]
lfn_scalar = BinnedLikelihoodFunction(g,mphi,mx,'scalar')
diff_perbin = np.abs(lfn_scalar.h_null - lfn_scalar.h_DM)
plt.step(bin_centers, diff_perbin, where='mid')
plt.xlim(1e3, 1e7)
plt.xlabel('Energy (GeV)')
plt.ylabel(r'$|\Delta N_{events}|$')
plt.gca().set_yscale("log")
plt.gca().set_xscale("log")
plt.tight_layout()
plt.show()
