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

ducc_energy = ducc.compute_ducc_of(fmat, vmat, t1_amps, t2_amps, n_a, n_b, n_orb,n_occ, act_max, compute_three_body=False)

print("Energy from exact diagonalisation using OpenFermion = ",ducc_energy)
