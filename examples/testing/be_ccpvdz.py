import ducc
import scipy
import vqe_methods
import pyscf_helper

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, molden, cc
from pyscf.cc import ccsd

import openfermion as of
from openfermion import *
from tVQE import *

import numpy as np

geometry = [('Be', (0,0,0))]
charge = 0
spin = 0
basis = 'cc-pvdz'
[n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = ducc.init(geometry,charge,spin,basis,reference='rhf')

sq_ham = ducc.SQ_Hamiltonian()
sq_ham.init(h, g, C, S)
print(" HF Energy: %21.15f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))

n_occ = n_a+n_b
n_act = 5
n_frz = 0
act_max = n_frz+n_act
shift = 2*n_frz

mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-14
mf.max_cycle = 1000
mf.verbose = 0
mf.init_guess = 'atom'
mf.kernel()
print("MO energies:")
print(mf.mo_energy)

mccsd = cc.UCCSD(mf)
mccsd.conv_tol = 1e-12
mccsd.conv_tol_normt = 1e-10
#myccsd.diis_start_cycle=10000
mccsd.max_cycle = 1000
mccsd.kernel()
t1_amps = mccsd.t1
t2_amps = mccsd.t2

fmat = sq_ham.make_f(range(n_a),range(n_b))
vmat = sq_ham.make_v()

t1_amps, t2_amps = ducc.get_t_ext(t1_amps,t2_amps,n_a,n_b,act_max)
t1_amps, t2_amps = ducc.transform_t_spatial_to_spin(t1_amps, t2_amps, n_a, n_b, n_orb)

"""
constant, one_body, two_body_anti = ducc.get_ducc_a2(fmat,vmat,t1_amps,t2_amps,n_a,n_b,n_orb,n_occ,act_max,compute_three_body=False)

umat_1 = ducc.get_u(two_body_anti,range(n_a),range(n_b))
one_body -= umat_1

fmat_spatial = ducc.get_spin_to_spatial_1(one_body,n_orb)
vmat_spatial = ducc.get_spin_to_spatial_1(two_body_anti,n_orb)
"""

#"""
umat_1 = ducc.get_u(vmat,range(n_a),range(n_b))
fmat -= umat_1

fmat_spatial = ducc.get_spin_to_spatial_1(fmat,n_orb)
vmat_spatial = ducc.get_spin_to_spatial_1(vmat,n_orb)

constant = 0
#"""

h1 = fmat_spatial[0:act_max,0:act_max]
h2 = vmat_spatial[0:act_max,0:act_max,0:act_max,0:act_max]  


cisolver = fci.direct_nosym.FCISolver()
ecore = constant
nroots = 1
nelec = n_occ


efci_orgnl,ci_orgnl = cisolver.kernel(h1, h2, h1.shape[1], nelec, max_space=450, ecore=ecore,nroots =nroots,verbose=5)
fci_dim_orgnl = ci_orgnl.shape[0]*ci_orgnl.shape[1]
print(" FCI:        %12.8f Dim:%6d"%(efci_orgnl,fci_dim_orgnl))

print(" SCF:        %12.8f"%(E_scf))

print(" Correlation energy:        %12.8f"%(efci_orgnl-E_scf))
