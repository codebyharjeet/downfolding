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

#fmat_test = one_body_to_matrix_ph(fock, n_orb, n_occ)
#vmat_test = two_body_to_tensor_ph(wn,n_orb,n_occ)

#fock_op = normal_ordered(one_body_to_op(fmat,n_orb,n_occ))
#wn_op = normal_ordered(two_body_to_op(vmat,n_orb,n_occ))

#t1_amps, t2_amps = get_t_ext(t1_amps,t2_amps,n_a,n_b,act_max)

#t1_amps, t2_amps = transform_t_spatial_to_spin(t1_amps, t2_amps, n_a, n_b, n_orb)

#compute_ducc(fmat,vmat,t1_amps,t2_amps,n_a,n_b,n_occ,n_orb,act_max)

C = mf.mo_coeff

h  = pyscf.scf.hf.get_hcore(mf.mol)
j,k = mf.get_jk()
h = C.T @ h @ C;
j = C.T @ j @ C;
k = C.T @ k @ C;
h1 = mf.mo_coeff.T.dot(mf.get_hcore()).dot(mf.mo_coeff)

Fao = mf.get_fock()
Fmo = C.T @ Fao @ C

#hello = Fmo-j+0.5*k
#test = np.linalg.norm(hello-h1)
#print("test = ",test)


h2 = pyscf.ao2mo.kernel(mol, C, aosym="s4", compact=False)
h2.shape = (n_orb, n_orb, n_orb, n_orb)

fmat_spatial = np.zeros((n_orb,n_orb))
vmat_spatial = np.zeros((n_orb,n_orb,n_orb,n_orb))
for p in range(0,n_orb):
        pa = 2*p
        for q in range(0,n_orb):
            qa = 2*q
            qb = 2*q + 1
            fmat_spatial[p,q] = fmat[pa,qa]
print(fmat_spatial.shape)
for p in range(0,n_orb):
        pa = 2*p
        pb = 2*p+1
        for q in range(0,n_orb):
            qa = 2*q
            qb = 2*q+1
            for r in range(0,n_orb):
                ra = 2*r
                rb = 2*r+1
                for s in range(0,n_orb):
                    sa = 2*s
                    sb = 2*s+1
                    # vmat_spatial[p,q,r,s] = vmat[pa,qb,ra,sb]
                    vmat_spatial[p,r,q,s] = vmat[pa,qb,ra,sb]
                    #vmat_spatial[p,q,r,s] = vmat[pa, ra,qb,sb]
             
print(vmat_spatial.shape)



f_norm = np.linalg.norm(Fmo-fmat_spatial)
print("Norm of one electron integral = ",f_norm)
g_norm = np.linalg.norm(h2-vmat_spatial)
print("Norm of two electron integral = ",g_norm)

fmat_spatial = fmat_spatial -j+0.5*k

act_spc = 5
fmat_spatial_act = fmat_spatial[0:act_spc,0:act_spc]
vmat_spatial_act = vmat_spatial[0:act_spc,0:act_spc,0:act_spc,0:act_spc]

print("shape of fmat and vmat spatial = ",fmat_spatial.shape,vmat_spatial.shape)
#comment = """
cisolver = fci.direct_spin1.FCI()
ecore = 0
nroots = 1
nelec = n_occ
efci, ci = cisolver.kernel(fmat_spatial, vmat_spatial, fmat_spatial.shape[1], nelec, ecore=ecore,nroots =nroots,verbose=100)
fci_dim = ci.shape[0]*ci.shape[1]
print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))
#"""



#print("One electron from pyscf")
#tprint(Fmo)
#print("After conversion from spin to spatial")
#tprint(fmat_spatial)

#g_two = proj_tens_to_as(h2,2)
#print("Two electron from pyscf")
#tprint(g_two)

#print("Two electron spin to spatial transformation")
#vmat_spatial = proj_tens_to_as(vmat_spatial,2)
#tprint(vmat_spatial)









#hn = fock_op + wn_op
#hn_proj = as_proj(hn,2*act_max)
#print("Energy of A1 Hamiltonian = ", eigenspectrum(hn_proj)[0].real)

#sq_ham_act = sq_ham.extract_local_hamiltonian(act_max)
#fermi_ham_act = sq_ham_act.export_FermionOperator()
#print("Ground state energy in truncated basis:")
#print(eigenspectrum(fermi_ham_act)[0]-E_scf)