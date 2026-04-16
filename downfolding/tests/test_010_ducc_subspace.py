import numpy.testing as npt
import pytest
import sys
import pyscf
from pyscf import gto, scf, fci, ao2mo
from downfolding import *
from downfolding import Driver



def _check_ducc_subspace(approx, expected):
    # common setup
    mol = gto.M(
        atom=[["Be", (0.0, 0.0, 0.0)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
    )
    mf = scf.RHF(mol)
    mf.kernel(verbose=0)
    
    n_frozen = 1
    n_act = 5

    n_e_total = int(np.sum(mf.mo_occ))
    n_e_rem = n_e_total - (2 * n_frozen) 
    n_a = n_b = n_e_rem // 2

    C = mf.mo_coeff
    S = mf.get_ovlp()

    O_env = C[:, :n_frozen]
    C_rem = C[:, n_frozen:]
    n_rem = C_rem.shape[1]

    d1_embed = 2.0 * np.dot(O_env, O_env.T)

    h_core_ao = mf.get_hcore()
    vj, vk = mf.get_jk(mol, d1_embed, hermi=1)

    # Calculate scalar core energy
    E_core = mol.energy_nuc() + np.sum(d1_embed * (h_core_ao + 0.5 * vj - 0.25 * vk))
    print(f"\nFrozen Core Energy Shift (E_core): {E_core:.15f}")


    F_core_ao = h_core_ao + vj - 0.5 * vk
    h_eff = C_rem.T @ F_core_ao @ C_rem
    print("h_eff shape (Active + Virtual):", h_eff.shape)

    g_eff = pyscf.ao2mo.kernel(mol, C_rem, aosym="s4", compact=False)
    g_eff = g_eff.reshape(n_rem, n_rem, n_rem, n_rem)
    print("g_eff shape (Active + Virtual):", g_eff.shape)    


    driver = Driver.from_mo_basis(mf, n_e=n_e_total, n_orb=C.shape[1], ecore=E_core, h1=h_eff, h2=g_eff, nfrozen=n_frozen)


    driver.run_ducc(n_act=n_act, approximation=approx, amp_type="CCSD", three_body=False, four_body=False)
    for backend in ("pyscf",):
        energy = driver.exact_diagonalize(backend=backend)
        npt.assert_allclose(energy-mf.e_tot, expected,atol=1e-8, rtol=0,err_msg=f"DUCC {approx!r} {backend} energy mismatch")




def test_010_ducc_subspace_a1():
    _check_ducc_subspace("a1", -0.024433577082)

def test_010_ducc_subspace_a4():
    _check_ducc_subspace("a4", -0.032840894814)

def test_010_ducc_subspace_a7():
    _check_ducc_subspace("a7", -0.045419332112)

test_010_ducc_subspace_a1()
# test_010_ducc_subspace_a4()
# test_010_ducc_subspace_a7()




