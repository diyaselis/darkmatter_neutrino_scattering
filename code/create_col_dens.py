import numpy as np
import NFW
from xs import *

MC=np.load('../mc/mese_cascades_MC_2013_dnn.npy')
muon_mc=np.load('../mc/mese_cascades_muongun_dnn.npy')

try:
    column_dens=np.load('column_dens.npy')
except:
    ras = np.append(MC['ra'],muon_mc['ra'])
    decs = np.append(MC['dec'],muon_mc['dec'])
    column_dens = NFW.get_t_NFW(ras,decs) * gr * Na /cm**2  # g/ cm^2 -> eV^3
    np.save('../created_files/column_dens',column_dens)

print(len(column_dens))
print(column_dens[:10])
