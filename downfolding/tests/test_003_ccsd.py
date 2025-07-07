import pytest
import numpy as np
import numpy.testing as npt
import sys
from pyscf import gto, scf, cc
from downfolding import *

def test_003_ccsd():
    # build molecule using PySCF and run SCF calculation
    atom="""
        O 
        H 1 1.1
        H 1 1.1 2 104
        """
    mol = gto.M(
        atom=atom,
        basis="6-31g",
        charge=0,
        spin=0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    nfreeze = 1
    driver = Driver.from_pyscf(mf, nfrozen=nfreeze)

    ccsd_etot      = driver.run_ccsd(diis_size=30,optimized=False)

    ccsd_etot_fast = driver.run_ccsd(diis_size=None,optimized=True)

    mycc = cc.CCSD(mf, frozen=nfreeze).run()
    # print('CCSD total energy', mycc.e_tot)
    npt.assert_allclose(ccsd_etot, mycc.e_tot, atol=1e-7, rtol=0, err_msg="CCSD total energies differ!")
    npt.assert_allclose(ccsd_etot_fast, mycc.e_tot, atol=1e-7, rtol=0, err_msg="CCSD total energies differ!")

test_003_ccsd()