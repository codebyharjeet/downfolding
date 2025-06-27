import pytest
import sys
from pyscf import gto, scf
from downfolding import *

def test_007_ducc_ccsd_t():
    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["Be", (0.0, 0.0, 0.0)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
    )
    mf = scf.RHF(mol)
    mf.kernel(verbose=0)
    driver = Driver.from_pyscf(mf, nfrozen=0)

    driver.run_ducc(n_act=5, approximation="a7", three_body=False, four_body=False)

    driver.exact_diagonalize(backend="pyscf")

    driver.exact_diagonalize(backend="openfermion")

    driver.run_ccsd_t()

    
test_007_ducc_ccsd_t()