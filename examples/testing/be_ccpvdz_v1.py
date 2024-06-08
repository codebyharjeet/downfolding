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

f_blocks, v_blocks = ducc.get_integral_blocks(fmat, 0.25 * vmat, n_a, n_b, n_orb)
t_blocks = ducc.get_t_blocks(t1_amps, t2_amps)

ft1_mat, ft1_const = ducc.compute_ft1(f_blocks, t_blocks, n_a, n_b, n_orb)

ft1_mat_op = normal_ordered(ducc.one_body_to_op(ft1_mat, n_orb, n_occ))
print(ft1_mat_op)


exit()

"""
constant, one_body, two_body_anti = ducc.get_ducc_a2(fmat,vmat,t1_amps,t2_amps,n_a,n_b,n_orb,n_occ,act_max,compute_three_body=False)

umat_1 = ducc.get_u(two_body_anti,range(n_a),range(n_b))
one_body -= umat_1

fmat_spatial = ducc.get_spin_to_spatial_1(one_body,n_orb)
vmat_spatial = ducc.get_spin_to_spatial_1(two_body_anti,n_orb)
"""

"""
umat_1 = ducc.get_u(vmat,range(n_a),range(n_b))
fmat -= umat_1

fmat_spatial = ducc.get_spin_to_spatial_1(fmat,n_orb)
vmat_spatial = ducc.get_spin_to_spatial_1(vmat,n_orb)

constant = 0
"""


one_body_op = normal_ordered(ducc.one_body_to_op(fmat, act_max, n_occ))
two_body_op = normal_ordered(ducc.two_body_to_op(vmat, act_max, n_occ))
ham = one_body_op + 0.25*two_body_op
print("ham constructed as fermoperator")

import time

# this one is not fast
start = time.time()
ham_sparse_number = get_number_preserving_sparse_operator(ham, 2*n_orb, n_occ)
print("ham contructed with number preserving")
energy_3, state_3 = get_ground_state(ham_sparse_number)
print("fci energy sparse 3 number = ",energy_3)
end = time.time()
print("Computed in = ",end - start)

# this is fast
start_1 = time.time()
ham_sparse = get_sparse_operator(ham)
print("ham contructed in matrix")
energy_2, state = get_ground_state(ham_sparse)
print("fci energy sparse 2 = ",energy_2)
end_1 = time.time()
print("Computed in = ",end_1 - start_1)

"""
energy_1 = sparse_eigenspectrum(ham_sparse)
print("fci energy sparse = ",energy_1[0].real)

energy = eigenspectrum(ham)
print("fci energy = ",energy[0].real)
"""



"""
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
"""