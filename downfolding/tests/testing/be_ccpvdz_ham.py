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
#print(" HF Energy: %21.15f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))

n_occ = n_a+n_b
n_act = 5
print("Size of active space = ",n_act)

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
fdic = ducc.one_body_mat2dic(fmat,n_occ,n_act,n_orb)
fop = ducc.one_body_to_op(fmat,n_occ,n_orb)

vten = sq_ham.make_v()
vdic = ducc.two_body_ten2dic(vten,n_occ,n_act,n_orb)
vop = ducc.two_body_to_op(vten,n_occ,n_orb)

t1dic = ducc.t1_mat2dic(t1_amps,n_a,n_act,n_orb)
s1_op = ducc.t1_to_op(ducc.t1_to_ext(t1_amps,n_act))

t2dic = ducc.t2_ten2dic(t2_amps,n_a,n_act,n_orb)
s2_op = ducc.t2_to_op(ducc.t2_to_ext(t2_amps,n_act))

fop_test = ducc.as_proj(normal_ordered(fop),2*n_act)

vop_test = ducc.as_proj(normal_ordered(vop),2*n_act)

fn_s1_op_test = ducc.as_proj(normal_ordered(commutator(fop,s1_op)),2*n_act)
print("fn_s1_op_test")

fn_s2_op_test = ducc.as_proj(normal_ordered(commutator(fop,s2_op)),2*n_act)
print("fn_s2_op_test")

wn_s1_op_test = ducc.as_proj(normal_ordered(commutator(vop,s1_op)),2*n_act)
print("wn_s1_op_test")

wn_s2_op_test = ducc.as_proj(normal_ordered(commutator(vop,s2_op)),2*n_act)
print("wn_s2_op_test")

fn_s1_s1_op_test = 0.5 * ducc.as_proj(normal_ordered(commutator(commutator(fop,s1_op),s1_op)),2*n_act)
print("fn_s1_s1_op_test")

fn_s1_s2_op_test = 0.5 * ducc.as_proj(normal_ordered(commutator(commutator(fop,s1_op),s2_op)),2*n_act)
print("fn_s1_s2_op_test")

fn_s2_s1_op_test = 0.5 * ducc.as_proj(normal_ordered(commutator(commutator(fop,s2_op),s1_op)),2*n_act)
print("fn_s2_s1_op_test")

fn_s2_s2_op_test = 0.5 * ducc.as_proj(normal_ordered(commutator(commutator(fop,s2_op),s2_op)),2*n_act)
print("fn_s2_s2_op_test")

ducc_ham = fop_test + 0.25*vop_test + fn_s1_op_test + fn_s2_op_test + 0.25*wn_s1_op_test + 0.25*wn_s2_op_test + fn_s1_s1_op_test + fn_s1_s2_op_test + fn_s2_s1_op_test + fn_s2_s2_op_test
print("added")

ducc_energy, _ = get_ground_state(get_sparse_operator(ducc_ham))

print("DUCC correlation energy = ",ducc_energy)