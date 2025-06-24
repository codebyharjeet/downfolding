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

    driver.run_ducc(n_act=5, approximation="a7", three_body=False, four_body=False)

    
test_Driver()