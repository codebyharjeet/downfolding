import pytest
import sys
import numpy as np
from pyscf import gto, scf, cc 
from downfolding import *
import numpy.testing as npt

def test_006_ccsd_t():
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

    ccsd_etot, ccsd_t_corr = driver.run_ccsd_t()

    mycc = cc.CCSD(mf).run()
    # print('CCSD total energy', mycc.e_tot)
    et = mycc.ccsd_t()
    # print('CCSD(T) total energy', mycc.e_tot + et)

    npt.assert_allclose(ccsd_etot, mycc.e_tot, atol=1e-8, rtol=0, err_msg="CCSD total energies differ!")
    npt.assert_allclose(ccsd_etot + ccsd_t_corr, mycc.e_tot + et, atol=1e-8, rtol=0, err_msg="CCSD(T) total energies differ!")


test_006_ccsd_t()