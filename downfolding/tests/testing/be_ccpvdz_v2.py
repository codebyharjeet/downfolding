import ducc
import scipy
import vqe_methods
import pyscf_helper

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, cc
from pyscf.cc import ccsd

import openfermion as of
from openfermion import *
from tVQE import *

import numpy as np

geometry = [('H', (0,0,0)),('H', (0,0,1)),('H', (0,0,2)),('H', (0,0,3)),('H', (0,0,4)),('H', (0,0,5))]
charge = 0
spin = 0
basis = 'cc-pvdz'
[n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = ducc.init(geometry,charge,spin,basis,reference='rhf')

sq_ham = ducc.SQ_Hamiltonian()
sq_ham.init(h, g, C, S)
#print(" HF Energy: %21.15f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))

n_occ = n_a+n_b
n_act = 24
print("Size of active space = ",n_act)

mf = scf.RHF(mol)
#mf.conv_tol_grad = 1e-14
#mf.max_cycle = 1000
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

ducc_ham = ducc.get_ducc_ham_active_space(fmat,vmat,t1_amps,t2_amps,n_a,n_b,n_orb,n_occ,n_act,inc_3_body=False)

#ducc_ham += of.FermionOperator("", float(E_scf))
print(ducc_ham)
exit()
ducc_energy, _ = get_ground_state(get_sparse_operator(ducc_ham))


print("DUCC correlation energy = %.8f" %(ducc_energy))
print("DUCC energy = %.8f" %(ducc_energy+E_scf))


