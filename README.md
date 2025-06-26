downfolding
==============================
[//]: # (Badges)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Actions Build Status](https://github.com/codebyharjeet/downfolding/workflows/CI/badge.svg)](https://github.com/codebyharjeet/downfolding/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/codebyharjeet/downfolding/branch/main/graph/badge.svg)](https://codecov.io/gh/codebyharjeet/downfolding/branch/main)


A simple python package for computing **Double Unitary Coupled Cluster** (DUCC) downfolded Hamiltonians for molecular systems.

---

## Features

* **Flexible Input Formats**: Initialize from PySCF mean-field object or FCIDUMP file.
* **Multiple DUCC Approximations**: Choose between A2–A7 methods, with optional 3‑ and 4‑body terms.
* **Exact Diagonalization**: Support for PySCF FCI solver and sparse eigensolvers via OpenFermion.
* **Exportable Integrals**: Save downfolded integrals in `.npz` or OpenFermion operator format.

---

## Installation

Clone the repository and install via pip:

```bash
git clone https://github.com/codebyharjeet/downfolding.git
cd downfolding
python -m pip install .
```

> This will compile the package and install all dependencies: NumPy, opt\_einsum, PySCF, OpenFermion, and PyTorch.

---

## Quickstart Example

```python
from pyscf import gto, scf
from downfolding import Driver

# 1) Build molecule and run SCF
mol = gto.M(atom=[["Be", (0.0, 0.0, 0.0)]], basis="cc-pvdz")
mf  = scf.RHF(mol)
mf.kernel(verbose=0)

# 2) Initialize DUCC driver (nfrozen = number of frozen orbitals)
driver = Driver.from_pyscf(mf, nfrozen=0)

# 3) Compute DUCC A7 Hamiltonian in active space of size n_act
driver.run_ducc(
    n_act=6,
    approximation="a7",
    three_body=False,
    four_body=False,
)

# 4) Exact diagonalization using PySCF FCI solver
driver.exact_diagonalize(backend="pyscf")

# 5) Or in the 2^N many-body space via OpenFermion
driver.exact_diagonalize(backend="openfermion")
```

---

## Saving Integrals

```python
# Save in NumPy .npz format
driver.save_integrals(format="npz")

# Save as an OpenFermion FermionOperator
driver.save_integrals(format="openfermion")
```

> This will write files into the current directory.


### Copyright

Copyright (c) 2025, Harjeet Singh


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
