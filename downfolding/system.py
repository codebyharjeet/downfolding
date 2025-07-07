

class System():
    """
    An system object.

    Attributes
    ----------
    eref : float
        the energy of the reference wave function (including nuclear repulsion contribution)
    nfzc : int
        the number of frozen core orbitals
    no : int
        the number of active occupied orbitals
    nv : int
        the number of active virtual orbitals
    nmo : int
        the number of active orbitals
    H : Hamiltonian object
        the normal-ordered Hamiltonian, which includes the Fock matrix, the ERIs, the spin-adapted ERIs (L), and various property integrals
    o : NumPy slice
        occupied orbital subspace
    v : NumPy slice
        virtual orbital subspace

    Methods
    -------

    """

    def __init__(
        self,
        mf,
        nelectrons,
        n_a,
        n_b,
        norbitals,
        nqubits,
        nfrozen,
        nuclear_repulsion=0.0,
        mo_energies=None,
        mo_occupation=None,
    ):
        # basic information
        self.meanfield = mf 
        self.nelectrons = nelectrons
        self.n_a = n_a
        self.n_b = n_b
        self.norbitals = norbitals
        self.nqubits = nqubits 
        self.nfrozen = nfrozen
        self.nuclear_repulsion = nuclear_repulsion
        self.mo_energies = mo_energies
        self.mo_occupation = mo_occupation
    

    def print(self):
        print("\n   HF Calculation Summary")
        print("   -------------------------------------")
        print("System                                         :")
        mol = self.meanfield.mol 
        print(mol.atom)
        print("Basis set                                      :%12s" %(mol.basis))

        if self.nfrozen == 0:
            print("Number of Orbitals                             :%10i" %(self.norbitals))
            print("Number of electrons                            :%10i" %(self.nelectrons))
            print("Number of alpha electrons                      :%10i" %(self.n_a))
            print("Number of beta electrons                       :%10i" %(self.n_b))
      
        elif self.nfrozen != 0:
            print("\n ~~ Before freezing ~~")
            print("Number of Orbitals                             :%10i" %(self.norbitals+self.nfrozen))
            print("Number of electrons                            :%10i" %(self.nelectrons+2*self.nfrozen))
            print("Number of alpha electrons                      :%10i" %(self.n_a+self.nfrozen))
            print("Number of beta electrons                       :%10i" %(self.n_b+self.nfrozen))

            print("\n ~~ After freezing ~~")
            print("Number of frozen core orbitals                 :%10i" %(self.nfrozen))
            print("Number of Orbitals                             :%10i" %(self.norbitals))
            print("Number of electrons                            :%10i" %(self.nelectrons))
            print("Number of alpha electrons                      :%10i" %(self.n_a))
            print("Number of beta electrons                       :%10i" %(self.n_b))

        print("Nuclear Repulsion                              :%18.12f " %self.nuclear_repulsion)
        print("Electronic SCF energy                          :%18.12f " %(self.meanfield.e_tot-self.nuclear_repulsion))
        print("SCF Energy                                     :%18.12f " %(self.meanfield.e_tot))  