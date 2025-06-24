from typing import List, Tuple, Dict
import numpy as np
from downfolding.interfaces import load_pyscf_integrals
import pyscf
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, cc, lib
from pyscf.cc import ccsd

class Driver:

    @classmethod
    def from_pyscf(cls, meanfield, nfrozen):
        return cls(*load_pyscf_integrals(meanfield, nfrozen))

    # @classmethod
    # def from_fcidump(cls, fcidump, nfrozen, data_type=np.float64):
    #     return cls(*load_fcidump_integrals(fcidump, nfrozen, data_type=data_type))
    
    def __init__(self, system, hamiltonian):
        """
        Parameters
        ----------
        system : System
        hamiltonian : Integral
        """
        self.system = system
        self.H = hamiltonian
        self.hf_energy = 0
        self.correlation_energy = 0

    
    def run_hf(self):
        """
        Compute the Hartree-Fock energy.
        """
        from downfolding.hf import calc_hf
        self.hf_energy = calc_hf(self.system, self.H)

    def run_ducc(self, n_act, approximation, three_body, four_body):
        """
        Compute the DUCC energy.
        """
        from downfolding.ducc import calc_ducc
        ducc_energy = calc_ducc(self.system, self.H, n_act, approximation, three_body=three_body, four_body=four_body)
    

