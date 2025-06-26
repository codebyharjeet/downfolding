from __future__ import annotations          # enables | union in Python ≤3.9
import numpy as np
import copy as cp 
import torch
from enum import Enum, auto
from functools import lru_cache
from typing import Dict
from downfolding.helper import one_body_mat2dic, two_body_ten2dic, one_body_to_op, two_body_to_op
import openfermion as of 
from openfermion import *
from opt_einsum import contract
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermionpyscf import run_pyscf


Array = np.ndarray | torch.Tensor          # simple alias for type hints

class HamFormat(Enum):
    """Output formats the Hamiltonian can be converted to."""
    SPINORB_FV  = auto()     # (f, v_as) in the fermi vacuum spin–orbital basis
    SPATORB_PV  = auto()     # (h, v) in the physical vacuum spatial-orbital basis
    HILBERT     = auto()     # 2ⁿ×2ⁿ many-body matrix 
    BLOCK_DICT  = auto()     # SO blocks as a dict 
    OPENFERMION = auto()     # openfermion.FermionOperator


class Hamiltonian:
    """
    Molecular Hamiltonian 

    Internally we keep only the raw one- and two-electron integrals.
    Views in other formats are
    built lazily and memoised with @lru_cache.
    """

    def __init__(
        self,
        f: Array,            # 1-electron integrals (n_so × n_so)
        v: Array,            # 2-electron integrals (n_so⁴)
        n_a: int,            # number of alpha electrons
        n_b: int,            # number of beta  electrons
        n_orb: int,          # number of spatial orbitals
        constant: float = 0.0,    # optional scalar
        *,
        n_act: int | None = None,
        w: Array | None = None,   # optional 3-electron integrals (n_so⁶)
        x: Array | None = None,   # optional 4-electron integrals (n_so⁸)
    ):
        self._f = f
        self._v = v
        self.n_a = n_a
        self.n_b = n_b      
        self.n_orb = n_orb
        self.constant = constant 

        self.n_act = n_act
        self._w = w 
        self._x = x 


    def to(self, fmt: HamFormat | str, **kwargs):
        """Return the Hamiltonian in the requested *fmt*."""
        if not isinstance(fmt, HamFormat):
            fmt = HamFormat[fmt.upper()]

        if fmt is HamFormat.SPINORB_FV:
            return self._as_spinorb_fv()
        if fmt is HamFormat.SPATORB_PV:
            return self._as_spatorb_pv()
        if fmt is HamFormat.HILBERT:
            return self._as_hilbert(**kwargs)
        if fmt is HamFormat.BLOCK_DICT:
            return self._as_block_dict(**kwargs)
        if fmt is HamFormat.OPENFERMION:
            return self._as_openfermion(**kwargs)

        raise ValueError(f"Unsupported format {fmt!r}")

    __call__ = to                          # H("block_dict")

    # For f-strings:  f"{H:spinorb}"
    def __format__(self, spec: str):
        spec = (spec or "spinorb").upper()
        return str(self.to(spec))

    @classmethod
    def from_physical_vacuum(cls, h: Array, v: Array, n_a: int, n_b: int, n_orb: int, constant: float = 0.0, *, n_act: int | None = None, w: Array | None = None, x: Array | None = None,) -> Hamiltonian:
        # convert physical vacuum(h, v) to fermi vacuum (f, v)
        f, v_fv = cls.move_to_fermi_vacuum(h, v, n_a, n_b, n_orb)
        return cls(f, v_fv, n_a, n_b, n_orb,constant,n_act=n_act, w=w, x=x)
    
    def _as_spatorb_pv(self):
        return self.export_pyscf()

    @lru_cache
    def _as_block_dict(self) -> Dict[str, Array]:
        """Return a dict with the familiar (oo|oo), (oo|ov), … blocks."""
        o, v = occ, virt
        V = self._v
        return {
            "oo,oo": V[o, o, o, o],
            "oo,ov": V[o, o, o, v],
            "oo,vv": V[o, o, v, v],
            "ov,ov": V[o, v, o, v],
            "ov,vv": V[o, v, v, v],
            "vv,vv": V[v, v, v, v],
        }

    @lru_cache
    def _as_hilbert(self):
        """
        Many-body Hamiltonian matrix in the
        occupation-number basis
        """
        ham_op = self.export_FermionOperator()
        number_preserving = True  
        if number_preserving:
            return of.linalg.get_number_preserving_sparse_operator(ham_op, 2*self.n_act, self.n_a+self.n_b, spin_preserving=True)
        else:
            return of.get_sparse_operator(ham_op, n_qubits=2*self.n_act)

    @lru_cache
    def _as_openfermion(self):
        """Return an openfermion.QubitOperator (lazy import)."""
        return self.export_FermionOperator()

    def __repr__(self):
        n_so = self._f.shape[0]
        return f"<Hamiltonian | {n_so} spin orbitals>"

    @staticmethod
    def move_to_fermi_vacuum(h: Array, v_pv: Array, n_a, n_b, n_orb) -> tuple[Array,Array]:
        """
        f_{pq} = h_{pq} + \sum_{i \in occ} <pi||qi>
        v^{pq}_{rs} = <pq||rs>
        """
        o = slice(None, n_a+n_b)        
        v = contract('prqs', v_pv) - contract('psqr', v_pv)
        f = h + contract('piqi->pq', v[:, o, :, o])
        return f, 0.25*v 

    def export_pyscf(self):
        # Move back to physical vacuum 
        n_occ = self.n_a+self.n_b
        n_act = self.n_act

        fdic = one_body_mat2dic(self._f, n_occ, n_act, n_act)
        vdic = two_body_ten2dic(4*self._v, n_occ, n_act, n_act)
    
        fdic_pv = cp.deepcopy(fdic)
        vdic_pv = cp.deepcopy(vdic)
        constant_pv = cp.deepcopy(self.constant)

        fdic_pv["oo"] -= np.einsum("ipiq->pq", vdic["oooo"]) 
        fdic_pv["ov"] -= np.einsum("ipiq->pq", vdic["ooov"])
        fdic_pv["vo"] -= np.einsum("ipiq->pq", vdic["ovoo"]) 
        fdic_pv["vv"] -= np.einsum("ipiq->pq", vdic["ovov"]) 

        constant_pv -= np.einsum("ii->", fdic_pv["oo"]) 
        constant_pv -= np.einsum("ijij->", vdic_pv["oooo"])/2	

        # 1) Export spin orbital integrals to spatial orbital integrals
        # 2) Convert from physicist's notation to chemist's
        # 3) Assume RHF orbitals

        n = n_act 
        nv = 2*n_act-n_occ
        h = np.zeros((n, n))
        g = np.zeros((n, n, n, n))

        idx_map = {}
        idx_map["o"] = [(i,2*i,2*i+1) for i in range(n_occ//2)]
        idx_map["v"] = [(i+n_occ//2, 2*i, 2*i+1) for i in range(nv//2)]

        for block in fdic_pv.keys():
            for p,pa,pb in idx_map[block[0]]:
                for q,qa,qb in idx_map[block[1]]:
                    h[p,q] = fdic_pv[block][pa,qa]

        block = "oooo" 
        for p,pa,pb in idx_map[block[0]]:
            for q,qa,qb in idx_map[block[1]]:
                for r,ra,rb in idx_map[block[2]]:
                    for s,sa,sb in idx_map[block[3]]:
                        g[p,q,r,s] =  vdic_pv[block][pa,qb,ra,sb] # oooo


        block = "ooov" 
        for p,pa,pb in idx_map[block[0]]:
            for q,qa,qb in idx_map[block[1]]:
                for r,ra,rb in idx_map[block[2]]:
                    for s,sa,sb in idx_map[block[3]]:
                        g[p,q,r,s] =  vdic_pv[block][pa,qb,ra,sb] # ooov
                        g[p,q,s,r] = -vdic_pv[block][pa,qb,rb,sa] # oovo


        block = "ovoo" 
        for p,pa,pb in idx_map[block[0]]:
            for q,qa,qb in idx_map[block[1]]:
                for r,ra,rb in idx_map[block[2]]:
                    for s,sa,sb in idx_map[block[3]]:
                        g[p,q,r,s] =  vdic_pv[block][pa,qb,ra,sb] # ovoo
                        g[q,p,r,s] = -vdic_pv[block][pa,qb,rb,sa] # vooo


        block = "ovvv" 
        for p,pa,pb in idx_map[block[0]]:
            for q,qa,qb in idx_map[block[1]]:
                for r,ra,rb in idx_map[block[2]]:
                    for s,sa,sb in idx_map[block[3]]:
                        g[p,q,r,s] =  vdic_pv[block][pa,qb,ra,sb] # ovvv
                        g[q,p,r,s] = -vdic_pv[block][pa,qb,rb,sa] # vovv


        block = "vvov" 
        for p,pa,pb in idx_map[block[0]]:
            for q,qa,qb in idx_map[block[1]]:
                for r,ra,rb in idx_map[block[2]]:
                    for s,sa,sb in idx_map[block[3]]:
                        g[p,q,r,s] =  vdic_pv[block][pa,qb,ra,sb] # vvov
                        g[p,q,s,r] = -vdic_pv[block][pa,qb,rb,sa] # vvvo



        block = "oovv" 
        for p,pa,pb in idx_map[block[0]]:
            for q,qa,qb in idx_map[block[1]]:
                for r,ra,rb in idx_map[block[2]]:
                    for s,sa,sb in idx_map[block[3]]:
                        g[p,q,r,s] = vdic_pv[block][pa,qb,ra,sb] # oovv

        block = "vvoo" 
        for p,pa,pb in idx_map[block[0]]:
            for q,qa,qb in idx_map[block[1]]:
                for r,ra,rb in idx_map[block[2]]:
                    for s,sa,sb in idx_map[block[3]]:
                        g[p,q,r,s] = vdic_pv[block][pa,qb,ra,sb] # vvoo

        block = "ovov" 
        for p,pa,pb in idx_map[block[0]]:
            for q,qa,qb in idx_map[block[1]]:
                for r,ra,rb in idx_map[block[2]]:
                    for s,sa,sb in idx_map[block[3]]:
                        g[p,q,r,s] =  vdic_pv[block][pa,qb,ra,sb] # ovov
                        g[q,p,s,r] =  vdic_pv[block][pa,qb,ra,sb] # vovo
                        g[p,q,s,r] = -vdic_pv[block][pa,qb,rb,sa] # vovo
                        g[q,p,r,s] = -vdic_pv[block][pa,qb,rb,sa] # ovvo

        block = "vvvv" 
        for p,pa,pb in idx_map[block[0]]:
            for q,qa,qb in idx_map[block[1]]:
                for r,ra,rb in idx_map[block[2]]:
                    for s,sa,sb in idx_map[block[3]]:
                        g[p,q,r,s] =  vdic_pv[block][pa,qb,ra,sb] # vvvv

        g = np.einsum("pqrs->prqs",g)

        # print("a: ", g[0,1,2,1]) 
        # print("b: ", g[1,0,1,2]) 
        # Check for proper symmetries
        assert(np.allclose(h, h.T.conj()))

        # for p in range(g.shape[0]):
        #     for q in range(g.shape[1]):
        #         for r in range(g.shape[2]):
        #             for s in range(g.shape[3]):
        #                 if min([p,q,r,s]) <2: continue
        #                 if max([p,q,r,s]) >1: continue
        #                 if not np.isclose(g[p,q,r,s], g[q,p,s,r]):
        #                     print("p,q,r,s: ", p,q,r,s,g[p,q,r,s],g[q,p,s,r])

        return constant_pv, h, g       

    def export_FermionOperator(self):
        n_occ = self.n_a + self.n_b 
        n_act = self.n_act

        ham_op = of.FermionOperator("", self.constant)
        ham_op += normal_ordered(one_body_to_op(self._f,n_occ,n_act))
        ham_op += normal_ordered(two_body_to_op(self._v,n_occ,n_act))        

        if self._w is not None:
            ham_op += normal_ordered(three_body_to_op(self._w,n_occ,n_act))
        if self._x is not None:
            ham_op += normal_ordered(four_body_to_op(self._x,n_occ,n_act))
        return ham_op


"""
## Use case 
H = Hamiltonian(h, v, n_a, n_b, n_orb)

# 1. Explicit converter call
blocks = H.to('block_dict')

# 2. Callable shortcut
matrix_form = H(HamFormat.MATRIX)

# 3. Pythonic string formatting
print(f"{H:block_dict}")         
"""

