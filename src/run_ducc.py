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

from ducc import *

geometry = [('Be', (0,0,0))]
charge = 0
spin = 0
basis = 'cc-pvdz'

[n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis,reference='rhf')
sq_ham = pyscf_helper.SQ_Hamiltonian()
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
vmat = 0.25*sq_ham.make_v()

#testing
#fock = sq_ham.export_FN_ph(range(n_a),range(n_b))
#wn = sq_ham.export_WN_ph(range(n_a),range(n_b))

#fmat_test = one_body_to_matrix(fock, n_orb)
#vmat_test = two_body_to_tensor(wn, n_orb)
#fock_op = normal_ordered(one_body_to_op(fmat,n_orb,n_occ))
#wn_op = normal_ordered(two_body_to_op(vmat,n_orb,n_occ))

t1_amps, t2_amps = get_t_ext(t1_amps,t2_amps,n_a,n_b,act_max)

t1_amps, t2_amps = transform_t_spatial_to_spin(t1_amps, t2_amps, n_a, n_b, n_orb)

compute_ducc(fmat,vmat,t1_amps,t2_amps,n_a,n_b,n_occ,n_orb,act_max)


#hn = fock_op + wn_op
#hn_proj = as_proj(hn,2*act_max)
#print("Energy of A1 Hamiltonian = ", eigenspectrum(hn_proj)[0].real)



#sq_ham_act = sq_ham.extract_local_hamiltonian(act_max)
#fermi_ham_act = sq_ham_act.export_FermionOperator()
#print("Ground state energy in truncated basis:")
#print(eigenspectrum(fermi_ham_act)[0]-E_scf)