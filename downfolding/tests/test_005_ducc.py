import numpy.testing as npt
from pyscf import gto, scf
from downfolding import Driver

def _check_ducc(approx, expected):
    # common setup
    mol = gto.M(
        atom=[["Be", (0.0, 0.0, 0.0)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
    )
    mf = scf.RHF(mol)
    mf.kernel(verbose=0)
    driver = Driver.from_pyscf(mf, nfrozen=0)

    # run and test
    driver.run_ducc(n_act=5, approximation=approx, three_body=False, four_body=False)
    for backend in ("pyscf", "openfermion"):
        energy = driver.exact_diagonalize(backend=backend)
        npt.assert_allclose(energy, expected,atol=1e-8, rtol=0,err_msg=f"DUCC {approx!r} {backend} energy mismatch")

def test_004_ducc_a1():
    _check_ducc("a1", -0.0228297064359837)

def test_004_ducc_a2():
    _check_ducc("a2", -0.0324886370563591)

def test_004_ducc_a3():
    _check_ducc("a3", -0.0679356132060296)

def test_004_ducc_a4():
    _check_ducc("a4", -0.0315602272157471)

def test_004_ducc_a5():
    _check_ducc("a5", -0.04503049345567)

def test_004_ducc_a6():
    _check_ducc("a6", -0.045799911247402)

def test_004_ducc_a7():
    _check_ducc("a7", -0.045824248188506)

# test_004_ducc_a1()
# test_004_ducc_a2()
# test_004_ducc_a3()
# test_004_ducc_a4()
# test_004_ducc_a5()
# test_004_ducc_a6()
# test_004_ducc_a7()