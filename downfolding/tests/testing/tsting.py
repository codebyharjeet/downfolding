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
ft1_mat, ft1_const = ducc.compute_ft1(f_blocks,t_blocks,n_a,n_b,n_orb)

ft2_mat_1,ft2_mat_2 = ducc.compute_ft2(f_blocks,t_blocks,n_a,n_b,n_orb)

wt1_mat_1, wt1_mat_2 = ducc.compute_wt1(v_blocks,t_blocks,n_a,n_b,n_orb)

wt2_mat_1, wt2_mat_2, wt2_mat_3, wt2 = ducc.compute_wt2(v_blocks,t_blocks,n_a,n_b,n_orb,compute_three_body=False)

ft1t1_mat_1, ft1t1 = ducc.compute_ft1t1(f_blocks,t_blocks,n_a,n_b,n_orb)

ft2t1_mat_1, ft2t1_mat_2, ft2t1 = ducc.compute_ft2t1(f_blocks,t_blocks,n_a,n_b,n_orb)

ft1t2_mat_1, ft1t2_mat_2 = ducc.compute_ft1t2(f_blocks,t_blocks,n_a,n_b,n_orb)

ft2t2_mat_1, ft2t2_mat_2, ft2t2_mat_3, ft2t2 = ducc.compute_ft2t2(f_blocks,t_blocks,n_a,n_b,n_orb,compute_three_body=False)
"""

#ducc_energy = ducc.compute_ducc_of(fmat, vmat, t1_amps, t2_amps, n_a, n_b, n_orb,n_occ, act_max, compute_three_body=False)

#print("Energy from exact diagonalisation using OpenFermion = ",ducc_energy)

def del(i, j):
  if i == j:
    return 1
  else:
    return 0


umat_1 = ducc.get_u(vmat,range(n_a),range(n_b))
fmat -= umat_1

fmat_pv = np.zeros((2*n_orb,2*n_orb))
vmat_pv = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))

# one electron integral
for A in range(2*n_occ,2*act_max):
    for B in range(2*n_occ,2*act_max):
        fmat_pv[A,B] = fmat[A,B]

for I in range(0,2*n_occ):
    for J in range(0,2*n_occ):
        fmat_pv[J,I] = fmat[J,I] 
        fmat_pv[J,I] -= del(I,J)

for I in range(0,2*n_occ):
    for A in range(2*n_occ,2*act_max):
        fmat_pv[I,A] = fmat[I,A]

for A in range(2*n_occ,2*act_max):
    for I in range(0,2*n_occ):
        fmat_pv[A,I] = fmat[A,I]

for B in range(2*n_occ,2*act_max):
    for C in range(2*n_occ,2*act_max):
        for A in range(2*n_occ,2*act_max):
            for I in range(0,2*n_occ):
                vmat_pv[B,C,I,A] = - vmat[B,C,A,I] 


for K in range(0,2*n_occ):
    for A in range(2*n_occ,2*act_max):
        for J in range(0,2*n_occ):
            for I in range(0,2*n_occ):
                vmat_pv[A,K,J,I] = - vmat[K,A,J,I] 
                fmat_pv[A,J] -= del(I,K)*fmat[A,J]

fmat_spatial = ducc.get_spin_to_spatial(fmat,n_orb)

vmat_spatial = ducc.get_spin_to_spatial(vmat,n_orb)

h1 = fmat_spatial[0:act_max,0:act_max]
h2 = vmat_spatial[0:act_max,0:act_max,0:act_max,0:act_max]  

constant = 0
cisolver = fci.direct_spin1.FCI()
ecore = constant
nroots = 1
nelec = n_occ

efci_orgnl,ci_orgnl = cisolver.kernel(h1, h2, h1.shape[1], nelec, ecore=ecore,nroots =nroots,verbose=100)
fci_dim_orgnl = ci_orgnl.shape[0]*ci_orgnl.shape[1]
print(" FCI:        %12.8f Dim:%6d"%(efci_orgnl,fci_dim_orgnl))

print(" SCF:        %12.8f"%(E_scf))

print("Correlation energy = ",efci_orgnl-E_scf)