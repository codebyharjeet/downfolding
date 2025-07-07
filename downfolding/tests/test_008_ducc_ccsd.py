import pytest
import sys
from pyscf import gto, scf
from downfolding import *

def test_005_ducc_ccsd():
    # build molecule using PySCF and run SCF calculation
    geom = '''
    Li      0.0      0.0     0.0
    H      0.0      0.0     1.0
    '''

    mol = gto.M(
        atom=geom,
        basis="6-31g",
        charge=0,
        spin=0,
    )
    mf = scf.RHF(mol)
    mf.kernel(verbose=0)
    driver = Driver.from_pyscf(mf, nfrozen=1)

    driver.run_ducc(n_act=5, approximation="a7", three_body=False, four_body=False)

    # driver.exact_diagonalize(backend="openfermion")

    # driver.run_ccsd(diis_size=30,optimized=False)
    driver.run_ccsd(diis_size=None,optimized=True)

    driver.exact_diagonalize(backend="pyscf")

    
test_005_ducc_ccsd()