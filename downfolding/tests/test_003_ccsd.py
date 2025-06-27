import pytest
import sys
from pyscf import gto, scf
from downfolding import *

def test_Driver():
    # build molecule using PySCF and run SCF calculation
    mol = gto.M(
        atom=[["Be", (0.0, 0.0, 0.0)]],
        basis="cc-pvdz",
        charge=0,
        spin=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    driver = Driver.from_pyscf(mf, nfrozen=0)

    driver.run_ccsd()

    
test_Driver()