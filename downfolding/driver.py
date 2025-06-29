import sys
from pathlib import Path
from typing import List, Tuple, Dict, Literal
import numpy as np
import scipy
from downfolding.interfaces import load_pyscf_integrals
from downfolding.hamiltonian import HamFormat, Hamiltonian
from downfolding.printing import get_timestamp, ducc_summary, ccsd_summary, ccsd_t_summary
import pyscf
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, cc, lib
from pyscf.cc import ccsd
import openfermion as of 
from openfermion.utils import save_operator

class Driver:

    @classmethod
    def from_pyscf(cls, meanfield, nfrozen):
        get_timestamp()
        return cls(*load_pyscf_integrals(meanfield, nfrozen))

    # @classmethod
    # def from_fcidump(cls, fcidump, nfrozen, data_type=np.float64):
    #     return cls(*load_fcidump_integrals(fcidump, nfrozen, data_type=data_type))
    
    def __init__(self, system, hamiltonian, hf_energy):
        """
        Parameters
        ----------
        system : System
        hamiltonian : Integral
        """
        self.system = system
        self.H = hamiltonian
        self.hf_energy = hf_energy
        self.correlation_energy = 0

    
    def run_hf(self):
        """
        Compute the Hartree-Fock energy.
        """
        from downfolding.hf import calc_hf
        self.hf_energy = calc_hf(self.system, self.H)

    def run_ccsd(self, diis_size=10, diis_start_cycle=0):
        """
        Compute the CCSD energy.
        """
        from downfolding.ccsd import ccsd_main 
        ccsd_etot = ccsd_main(self.system, self.H, diis_size, diis_start_cycle)
        self.correlation_energy = ccsd_etot - self.hf_energy
        ccsd_summary(ccsd_etot, self.correlation_energy)
        return ccsd_etot

    def run_ccsd_t(self):
        """
        Compute the CCSD(T) energy.
        """
        from downfolding.ccsd_t import ccsd_t_main 
        ccsd_etot, ccsd_t_corr = ccsd_t_main(self.system, self.H)
        ccsd_t_summary(ccsd_etot, ccsd_t_corr)
        # self.correlation_energy = ccsd_etot - self.hf_energy
        return ccsd_etot, ccsd_t_corr

    def run_ducc(self, n_act, approximation, three_body, four_body):
        """
        Compute the DUCC energy.
        """
        from downfolding.ducc import calc_ducc

        ham = calc_ducc(self.system, self.H, n_act, approximation, three_body=three_body, four_body=four_body)
        setattr(self, "H", ham)


    def exact_diagonalize(self, backend: str="pyscf") -> None:
        """
        Perform exact diagonalization using specified backend.
        
        Parameters
        ----------
        backend : str, optional
            Computational backend to use for diagonalization.
            Options: "pyscf" (default), "openfermion"
            
        Raises
        ------
        ValueError
            If unsupported backend is specified.
        """
        if backend == "pyscf":
            n_act = self.H.n_act
            n_a = self.H.n_a
            n_b = self.H.n_b
            constant, h, g = self.H(HamFormat.SPATORB_PV)
            p = fci.direct_nosym.FCISolver()
            e, fcivec = p.kernel(h, g, n_act, (n_a,n_b), max_space=450, nroots=1, verbose=0)      
            energy = e+constant

        elif backend == "openfermion":
            ham_mat = self.H(HamFormat.HILBERT)
            evals, evecs = scipy.sparse.linalg.eigsh(ham_mat, k=1, which="SA")
            energy = evals[0]
            
        else:
            raise ValueError(f"Unsupported backend '{backend}'. "
                            "Available options: 'pyscf', 'openfermion'")
        
        ducc_summary(energy, backend, self.hf_energy)
        return energy

    def save_integrals(self, format: Literal["npz", "openfermion"] = "npz", filename: str | None = None, directory: Path | str | None = None,) -> Path:
        """
        Save DUCC integrals to disk in the specified format.

        Parameters
        ----------
        format : {"npz","openfermion"}
            Output format: NumPy .npz archive or OpenFermion operator file.
        filename : str, optional
            Base filename. Defaults to the running script's name.
        directory : Path or str, optional
            Target directory. Defaults to the current working directory.

        Returns
        -------
        Path
            Full path to the saved file.
        """
        dir_path = Path(directory) if directory is not None else Path.cwd()
        dir_path.mkdir(parents=True, exist_ok=True)

        if filename:
            base = filename
        else:
            script = Path(sys.argv[0]).stem
            base = script if script else "ducc_integrals"

        if format.lower() == "npz":
            out_path = dir_path / f"{base}.npz"
            n_occ = self.H.n_a + self.H.n_b
            constant, h, g = self.H(HamFormat.SPATORB_PV)
            np.savez(out_path, H1=h, H2=g, ECORE=constant, NORB=self.H.n_act, NELEC=n_occ)

        elif format.lower() == "openfermion":            
            op = self.H(HamFormat.OPENFERMION)
            file_name = f"{base}"
            save_operator(op, file_name=file_name, data_directory=str(dir_path), allow_overwrite=True)
            out_path = dir_path / f"{file_name}.data"

        else:
            raise ValueError(f"Unsupported format '{format}'. Use 'npz' or 'openfermion'.")

        print(f"Integrals saved to {out_path}")

        return out_path


