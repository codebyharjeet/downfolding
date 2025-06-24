import numpy as np
import copy as cp
import scipy 
import pyscf
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, cc, lib
from pyscf.cc import ccsd
from pyscf.tools import molden
import time
from functools import reduce
from itertools import product
from downfolding.hamiltonian import Hamiltonian
from downfolding.system import System

def check_internal_stability(meanfield):
    print("Checking the internal stability of the SCF solution.")
    new_mo = meanfield.stability()[0]
    mo_diff = 0
    j = 0
    log = lib.logger.new_logger(meanfield)

    mo_diff = np.linalg.norm(meanfield.mo_coeff - new_mo)
    while (mo_diff > 1e-5) and (j < 3):
        print("Rotating orbitals to find stable solution: attempt %d."%(j+1))
        new_dm = meanfield.make_rdm1(new_mo,meanfield.mo_occ)
        meanfield.run(new_dm)
        new_mo = meanfield.stability()[0]
        mo_diff = np.linalg.norm(meanfield.mo_coeff - new_mo)
        j += 1
    if mo_diff > 1e-5:
        print("Unable to find a stable SCF solution after %d attempts."%(j+1))
    else:
        print("SCF solution is internally stable.")

def load_pyscf_integrals(meanfield, n_frzn_occ=0,n_act=None, dm0=None, stability=False):
    """Builds the System object using the information contained within a PySCF
    mean-field object for a molecular system.

    Arguments:
    ----------
    meanFieldObj : Object -> PySCF SCF/mean-field object
    nfrozen : int -> number of frozen electrons
    Returns:
    ----------
    system: System object
    """
    time_init = time.time()
    print(" ---------------------------------------------------------")
    print("                                                          ")
    print("                      Using Pyscf:")
    print("                                                          ")
    print(" ---------------------------------------------------------")
    print("                                                          ")
    mol = meanfield.mol

    n_orb = mol.nao_nr()
    n_qubits = 2*n_orb
    n_a, n_b = mol.nelec 
    n_el = n_a + n_b

    if stability:
        check_internal_stability(meanfield)
    assert meanfield.converged == True
    hf_energy = meanfield.e_tot
    mo_occ = meanfield.mo_occ
    E_nuc = mol.energy_nuc()

    T = mol.intor('int1e_kin_sph')
    V = mol.intor('int1e_nuc_sph') 
    H_core = T + V
    S = mol.intor('int1e_ovlp_sph')
    I = mol.intor('int2e_sph')

    print("\nSystem and Method:")
    print(mol.atom)

    print("Basis set                                      :%12s" %(mol.basis))
    print("Number of Orbitals                             :%10i" %(n_orb))
    print("Number of electrons                            :%10i" %(n_el))
    print("Number of alpha electrons                      :%10i" %(n_a))
    print("Number of beta electrons                       :%10i" %(n_b))
    print("Nuclear Repulsion                              :%18.12f " %E_nuc)
    print("Electronic SCF energy                          :%18.12f " %(meanfield.e_tot-E_nuc))
    print("SCF Energy                                     :%21.15f"%(meanfield.e_tot))


    print(" AO->MO")
    if n_frzn_occ != 0:
        print("Number of frozen orbitals = ",n_frzn_occ)
        assert(n_frzn_occ <= n_b)
        n_a   -= n_frzn_occ
        n_b   -= n_frzn_occ
        n_orb -= n_frzn_occ
        print(" NElectrons: %4i %4i" %(n_a, n_b))
        C = meanfield.mo_coeff
        Cact = C[:,n_frzn_occ:n_frzn_occ+n_orb]
        Cocc = C[:,0:n_frzn_occ]
        dm = Cocc @ Cocc.T
        j, k = scf.hf.get_jk(mol, dm)
        t = H_core + 2*j - k
        h = reduce(np.dot, (Cact.conj().T, H_core + 2*j - k, Cact))
        ecore = np.trace(2*dm @ (H_core + j - .5*k))
        print(" ecore: %12.8f" %ecore) 
        E_nuc += ecore
        C_a = C_b = Cact
        H_a = H_b = h 

    else:
        C_a = C_b = meanfield.mo_coeff
        H_a = C_a.T.dot(H_core).dot(C_a)
        H_b = C_b.T.dot(H_core).dot(C_b)

    A = np.array([[1,0],[0,0]])
    B = np.array([[0,0],[0,1]])
    AA = np.einsum("pq,rs->pqrs",A,A)
    AB = np.einsum("pq,rs->pqrs",A,B)
    BA = np.einsum("pq,rs->pqrs",B,A)
    BB = np.einsum("pq,rs->pqrs",B,B)

    h = np.kron(H_a,A) + np.kron(H_b,B)
    C = np.kron(C_a,A) + np.kron(C_b,B)
    S = np.kron(S,A) + np.kron(S,B)

    nmo_a, nmo_b = C_a.shape[1], C_b.shape[1]
    eri_aa = ao2mo.general(mol, (C_a,C_a,C_a,C_a), compact=False).reshape(nmo_a,nmo_a,nmo_a,nmo_a)
    eri_bb = ao2mo.general(mol, (C_b,C_b,C_b,C_b), compact=False).reshape(nmo_b,nmo_b,nmo_b,nmo_b)
    eri_ab = ao2mo.general(mol, (C_a,C_a,C_b,C_b), compact=False).reshape(nmo_a,nmo_a,nmo_b,nmo_b)
    eri_ba = eri_ab.transpose(2,3,0,1)

    eri_so = (np.kron(eri_aa, AA) +
              np.kron(eri_ab, AB) +
              np.kron(eri_ba, BA) +
              np.kron(eri_bb, BB))
    


    print(eri_so.shape, " %12.8f Mb" %(eri_so.nbytes*1e-6))

    hamiltonian = Hamiltonian(h, eri_so, n_a, n_b, n_orb)
    system = System(
        meanfield,
        n_el,
        n_a,
        n_b,
        n_orb,
        n_qubits,
        n_frzn_occ,
        nuclear_repulsion=E_nuc,
        mo_energies=meanfield.mo_energy,
        mo_occupation=meanfield.mo_occ,
    )
    return system, hamiltonian
