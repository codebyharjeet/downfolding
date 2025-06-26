from pyscf import gto, scf
from downfolding import Driver 

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

driver.run_ducc(n_act=6, approximation="a7", three_body=False, four_body=False)

driver.exact_diagonalize(backend="pyscf")

driver.exact_diagonalize(backend="openfermion")

# driver.save_integrals(format="npz")
# driver.save_integrals(format="openfermion")
