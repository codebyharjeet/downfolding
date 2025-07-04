import pytest
import numpy as np
import numpy.testing as npt
import sys
from pyscf import gto, scf, cc
from downfolding import *

def test_003_ccsd():
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

    ccsd_etot = driver.run_ccsd()

    mycc = cc.CCSD(mf).run()
    # print('CCSD total energy', mycc.e_tot)

    npt.assert_allclose(ccsd_etot, mycc.e_tot, atol=1e-8, rtol=0, err_msg="CCSD total energies differ!")

test_003_ccsd()