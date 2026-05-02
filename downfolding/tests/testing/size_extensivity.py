import numpy.testing as npt
from pyscf import gto, scf

from downfolding import Driver
from downfolding import *

def _check_size_extensivity(approx):
    h4 = gto.M(
        atom=[
            ["H", (0.0, 0.0, 0.0)],
            ["H", (0.0, 0.0, 1.0)],
            ["H", (0.0, 0.0, 2.0)],
            ["H", (0.0, 0.0, 3.0)],
        ],
        basis="cc-pvdz",
        charge=0,
        spin=0,
    )
    mf4 = scf.RHF(h4)
    mf4.kernel(verbose=0)

    h8 = gto.M(
        atom=[
            ["H", (0.0, 0.0, 0.0)],
            ["H", (0.0, 0.0, 1.0)],
            ["H", (0.0, 0.0, 2.0)],
            ["H", (0.0, 0.0, 3.0)],
            ["H", (100.0, 0.0, 0.0)],
            ["H", (100.0, 0.0, 1.0)],
            ["H", (100.0, 0.0, 2.0)],
            ["H", (100.0, 0.0, 3.0)],
        ],
        basis="cc-pvdz",
        charge=0,
        spin=0,
    )
    mf8 = scf.RHF(h8)
    mf8.kernel(verbose=0)

    npt.assert_allclose(mf8.e_tot, 2.0 * mf4.e_tot, atol=1e-8, rtol=0, err_msg="SCF is not size extensive for H4 vs H8",)

    # driver4 = Driver.from_pyscf(mf4, nfrozen=0)
    # driver4.run_ducc(n_act=4, approximation=approx, amp_type="CCSD", three_body=False, four_body=False)
    # e4 = driver4.exact_diagonalize(backend="pyscf")

    driver8 = Driver.from_pyscf(mf8, nfrozen=0)
    driver8.run_ducc(n_act=8, approximation=approx, amp_type="CCSD", three_body=False, four_body=False)
    h0, h1, h2 = driver8.H.export_pyscf()
    analyze_tensor_physics(h2)
    # e8 = driver8.exact_diagonalize(backend="pyscf")

    # npt.assert_allclose(e8, 2.0 * e4, atol=1e-8, rtol=0, err_msg=f"DUCC {approx!r} is not size extensive for H4 vs H8",)


def test_h4_h8_size_extensivity_a1():
    _check_size_extensivity("a1")


def test_h4_h8_size_extensivity_a7():
    _check_size_extensivity("a7")


test_h4_h8_size_extensivity_a1()
test_h4_h8_size_extensivity_a7()
