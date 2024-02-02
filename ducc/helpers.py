import ducc
import scipy
#import vqe_methods
#import pyscf_helper

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, molden, cc
from pyscf.cc import ccsd

import openfermion as of
from openfermion import *
#from tVQE import *

import numpy as np


def print_quote():
    message = "Hello world!"
    return print(message)


class SQ_Hamiltonian:
    """
    General hamiltonian operator:
    H = H(pq){p'q} + V(pqrs){p'q'sr} + A(p){p'} + B(p){p} + C(pq){p'q'} + D(pq){pq} + ...
    """

    def __init__(self):
        # operator integrals tensors
        self.int_H = np.array(())
        self.int_V = np.array(())
        self.int_A = np.array(())
        self.int_B = np.array(())
        self.int_C = np.array(())
        self.int_D = np.array(())

        # MO basis : really, just any transformation from AO->Current basis such that C'SC = I
        self.C = np.array(())

        # AO overlap
        self.S = np.array(())

        # number of spatial orbitals
        self.n_orb = 0

    def init(self, h, v, C, S):
        # molecule is a pyscf molecule object from gto.Mole()

        self.S = cp.deepcopy(S)
        self.C = cp.deepcopy(C)
        self.n_orb = self.C.shape[1] // 2

        self.int_H = cp.deepcopy(h)
        self.int_V = cp.deepcopy(v)

    def energy_of_determinant(self, config_a, config_b):
        """This only returns electronic energy"""
        e1 = 0
        e2 = 0
        for i in config_a:
            e1 += self.int_H[2 * i, 2 * i]
        for i in config_b:
            e1 += self.int_H[2 * i + 1, 2 * i + 1]
        for i in config_a:
            for j in config_a:
                if i >= j:
                    continue
                e2 += self.int_V[2 * i, 2 * i, 2 * j, 2 * j]
                e2 -= self.int_V[2 * i, 2 * j, 2 * j, 2 * i]
        for i in config_b:
            for j in config_b:
                if i >= j:
                    continue
                e2 += self.int_V[2 * i + 1, 2 * i + 1, 2 * j + 1, 2 * j + 1]
                e2 -= self.int_V[2 * i + 1, 2 * j + 1, 2 * j + 1, 2 * i + 1]
        for i in config_a:
            for j in config_b:
                e2 += self.int_V[2 * i, 2 * i, 2 * j + 1, 2 * j + 1]
        print("One-body = %16.10f" % e1)
        print("Two-body = %16.10f" % e2)
        e = e1 + e2
        return e

    def make_f(self, config_a, config_b):
        n_orb = self.n_orb
        f = cp.deepcopy(self.int_H)
        for p in range(0, n_orb):
            pa = 2 * p
            pb = 2 * p + 1
            for q in range(0, n_orb):
                qa = 2 * q
                qb = 2 * q + 1
                for i in config_a:
                    f[pa, qa] += (
                        self.int_V[pa, qa, 2 * i, 2 * i]
                        - self.int_V[pa, 2 * i, 2 * i, qa]
                    )
                    f[pb, qb] += self.int_V[pb, qb, 2 * i, 2 * i]
                for j in config_b:
                    f[pa, qa] += self.int_V[pa, qa, 2 * j + 1, 2 * j + 1]
                    f[pb, qb] += (
                        self.int_V[pb, qb, 2 * j + 1, 2 * j + 1]
                        - self.int_V[pb, 2 * j + 1, 2 * j + 1, qb]
                    )
        return f

    def make_v(self):
        """
        v^{pq}_{rs} = <pq||rs>
        """
        n_orb = self.n_orb
        v = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
        for p in range(0, n_orb):
            pa = 2 * p
            pb = 2 * p + 1
            for q in range(0, n_orb):
                qa = 2 * q
                qb = 2 * q + 1
                for r in range(0, n_orb):
                    ra = 2 * r
                    rb = 2 * r + 1
                    for s in range(0, n_orb):
                        sa = 2 * s
                        sb = 2 * s + 1
                        v[pa, qa, ra, sa] = (
                            self.int_V[pa, ra, qa, sa] - self.int_V[pa, sa, qa, ra]
                        )
                        v[pa, qb, ra, sb] = self.int_V[pa, ra, qb, sb]
                        v[pa, qb, rb, sa] = -self.int_V[pa, sa, qb, rb]
                        v[pb, qa, rb, sa] = self.int_V[pb, rb, qa, sa]
                        v[pb, qa, ra, sb] = -self.int_V[pb, sb, qa, ra]
                        v[pb, qb, rb, sb] = (
                            self.int_V[pb, rb, qb, sb] - self.int_V[pb, sb, qb, rb]
                        )
        return v


def init(
    molecule,
    charge,
    spin,
    basis,
    reference="rhf",
    n_frzn_occ=0,
    n_act=None,
    mo_order=None,
):
    # {{{
    # PYSCF inputs
    mol = gto.Mole()
    mol.atom = molecule

    # this is needed to prevent openblas - openmp clash for some reason
    # todo: take out
    # lib.num_threads(1)

    mol.max_memory = 8e3  # MB
    mol.charge = charge
    mol.spin = spin
    mol.basis = basis
    mol.symmetry = False
    mol.build()

    # orbitals and electrons
    n_orb = mol.nao_nr()
    n_a, n_b = mol.nelec
    n_el = n_a + n_b

    if n_act == None:
        n_act = n_orb

    # SCF
    if reference == "rhf":
        mf = scf.RHF(mol)
    elif reference == "rohf":
        mf = scf.ROHF(mol)
    elif reference == "uhf":
        mf = scf.UHF(mol)
    else:
        print("Please specify a proper reference (rhf/rohf/uhf).")
    mf.conv_tol_grad = 1e-14
    mf.max_cycle = 1000
    mf.verbose = 4
    mf.init_guess = "atom"
    # mf = scf.newton(mf).set(conv_tol=1e-12)

    hf_energy = mf.kernel()

    hf_energy = mf.e_tot

    assert mf.converged == True
    mo_occ = mf.mo_occ
    C = mf.mo_coeff
    occ_a = 0
    occ_b = 0
    virt_a = 0
    virt_b = 0

    # dump orbitals for viewing
    # molden.from_mo(mol, 'orbitals_canon.molden', C)

    if reference != "uhf":
        C_a = C_b = mf.mo_coeff
        mo_a = np.zeros(len(mo_occ))
        mo_b = np.zeros(len(mo_occ))
        for i in range(0, len(mo_occ)):
            if mo_occ[i] > 0:
                mo_a[i] = 1
                occ_a += 1
            else:
                virt_a += 1
            if mo_occ[i] > 1:
                mo_b[i] = 1
                occ_b += 1
            else:
                virt_b += 1

    else:
        C_a = mf.mo_coeff[0]
        C_b = mf.mo_coeff[1]
        mo_a = np.zeros(len(mo_occ[0]))
        mo_b = np.zeros(len(mo_occ[1]))
        for i in range(0, len(mo_occ[0])):
            if mo_occ[0][i] == 1:
                mo_a[i] = 1
                occ_a += 1
            else:
                virt_a += 1
        for i in range(0, len(mo_occ[1])):
            if mo_occ[1][i] == 1:
                mo_b[i] = 1
                occ_b += 1
            else:
                virt_b += 1

    P_a = np.diag(mo_a)
    P_b = np.diag(mo_b)

    E_nuc = mol.energy_nuc()

    #
    # if mo_order != None:
    #    print(len(mo_order) , mf.mo_coeff.shape[1])
    #    assert(len(mo_order) == mf.mo_coeff.shape[1])
    #    mf.mo_coeff = mf.mo_coeff[:,mo_order]

    # C = mf.mo_coeff #MO coeffs
    # S = mf.get_ovlp()

    ##READING INTEGRALS FROM PYSCF
    E_nuc = gto.Mole.energy_nuc(mol)
    T = mol.intor("int1e_kin_sph")
    V = mol.intor("int1e_nuc_sph")
    H_core = T + V
    S = mol.intor("int1e_ovlp_sph")
    I = mol.intor("int2e_sph")

    print("\nSystem and Method:")
    print(mol.atom)

    print("Basis set                                      :%12s" % (mol.basis))
    print("Number of Orbitals                             :%10i" % (n_orb))
    print("Number of electrons                            :%10i" % (n_el))
    print("Number of alpha electrons                      :%10i" % (n_a))
    print("Number of beta electrons                       :%10i" % (n_b))
    print("Nuclear Repulsion                              :%18.12f " % E_nuc)
    print(
        "Electronic SCF energy                          :%18.12f " % (mf.e_tot - E_nuc)
    )
    print("SCF Energy                                     :%21.15f" % (mf.e_tot))

    print(" AO->MO")
    # convert from AO to MO representation
    H_a = C_a.T.dot(H_core).dot(C_a)
    H_b = C_b.T.dot(H_core).dot(C_b)

    I_aa = np.einsum("pqrs,pi->iqrs", I, C_a)
    I_aa = np.einsum("iqrs,qj->ijrs", I_aa, C_a)
    I_aa = np.einsum("ijrs,rk->ijks", I_aa, C_a)
    I_aa = np.einsum("ijks,sl->ijkl", I_aa, C_a)

    I_ab = np.einsum("pqrs,pi->iqrs", I, C_a)
    I_ab = np.einsum("iqrs,qj->ijrs", I_ab, C_a)
    I_ab = np.einsum("ijrs,rk->ijks", I_ab, C_b)
    I_ab = np.einsum("ijks,sl->ijkl", I_ab, C_b)

    I_ba = np.einsum("pqrs,pi->iqrs", I, C_b)
    I_ba = np.einsum("iqrs,qj->ijrs", I_ba, C_b)
    I_ba = np.einsum("ijrs,rk->ijks", I_ba, C_a)
    I_ba = np.einsum("ijks,sl->ijkl", I_ba, C_a)

    I_bb = np.einsum("pqrs,pi->iqrs", I, C_b)
    I_bb = np.einsum("iqrs,qj->ijrs", I_bb, C_b)
    I_bb = np.einsum("ijrs,rk->ijks", I_bb, C_b)
    I_bb = np.einsum("ijks,sl->ijkl", I_bb, C_b)

    J_a = np.einsum("pqrs,rs->pq", I_aa, P_a) + np.einsum("pqrs,rs->pq", I_ab, P_b)
    J_b = np.einsum("pqrs,rs->pq", I_bb, P_b) + np.einsum("pqrs,rs->pq", I_ba, P_a)
    K_a = np.einsum("pqrs,rq->ps", I_aa, P_a)
    K_b = np.einsum("pqrs,rq->ps", I_bb, P_b)

    F_a = H_a + J_a - K_a
    F_b = H_b + J_b - K_b
    manual_energy = (
        E_nuc
        + 0.5 * np.einsum("pq,pq", H_a + F_a, P_a)
        + 0.5 * np.einsum("pq,pq", H_b + F_b, P_b)
    )
    print("Manual HF energy = %21.15f" % (manual_energy))
    onebody = np.einsum("pq,pq", H_a, P_a) + np.einsum("pq,pq", H_b, P_b)
    twobody = 0.5 * np.einsum("pq,pq", J_a - K_a, P_a) + 0.5 * np.einsum(
        "pq,pq", J_b - K_b, P_b
    )
    # print("One-body energy = %16.10f"%onebody)
    # print("Two-body energy = %16.10f"%twobody)

    # group terms to be exported in {a0,b0,a1,b2,...} MO ordering
    A = np.array([[1, 0], [0, 0]])
    B = np.array([[0, 0], [0, 1]])
    AA = np.einsum("pq,rs->pqrs", A, A)
    AB = np.einsum("pq,rs->pqrs", A, B)
    BA = np.einsum("pq,rs->pqrs", B, A)
    BB = np.einsum("pq,rs->pqrs", B, B)

    h = np.kron(H_a, A) + np.kron(H_b, B)
    p = np.kron(P_a, A) + np.kron(P_b, B)
    # f = np.kron(F_a,A) + np.kron(F_b,B)
    C = np.kron(C_a, A) + np.kron(C_b, B)
    S = np.kron(S, A) + np.kron(S, B)
    g = np.kron(I_aa, AA) + np.kron(I_ab, AB) + np.kron(I_ba, BA) + np.kron(I_bb, BB)

    # FCI
    if False:
        cisolver = fci.FCI(mf)
        print("FCI energy = %21.15f" % cisolver.kernel()[0])

    return (n_orb, n_a, n_b, h, g, mol, E_nuc, mf.e_tot, C, S)


thresh = 1e-15


def tprint(tens):
    if np.ndim(tens) == 0:
        print(tens)
    elif np.ndim(tens) == 1:
        for i in range(0, len(tens)):
            if abs(tens[i]) > thresh:
                print("[%d] : %e" % (i, tens[i]))
    elif np.ndim(tens) == 2:
        for i in range(0, tens.shape[0]):
            for j in range(0, tens.shape[1]):
                if abs(tens[i, j]) > thresh:
                    print("[%d,%d] : %e" % (i, j, tens[i, j]))
    elif np.ndim(tens) == 4:
        for i in range(0, tens.shape[0]):
            for j in range(0, tens.shape[1]):
                for k in range(0, tens.shape[2]):
                    for l in range(0, tens.shape[3]):
                        if abs(tens[i, j, k, l]) > thresh:
                            print("[%d,%d,%d,%d] : %e" % (i, j, k, l, tens[i, j, k, l]))
    elif np.ndim(tens) == 6:
        for i in range(0, tens.shape[0]):
            for j in range(0, tens.shape[1]):
                for k in range(0, tens.shape[2]):
                    for l in range(0, tens.shape[3]):
                        for m in range(0, tens.shape[4]):
                            for n in range(0, tens.shape[5]):
                                if abs(tens[i, j, k, l, m, n]) > thresh:
                                    print(
                                        "[%d,%d,%d,%d,%d,%d] : %e"
                                        % (i, j, k, l, m, n, tens[i, j, k, l, m, n])
                                    )
    else:
        print("TODO: implement a printing for a %dD tensor." % (np.ndim(tens)))
        exit()


def get_t_ext(t1_amps, t2_amps, n_a, n_b, act_max):
    # getting t1_external_a
    for i in range(0, n_a):
        for a in range(n_a, act_max):
            t1_amps[0][i, a - n_a] = 0
    # getting t1_external_b
    for i in range(0, n_b):
        for a in range(n_b, act_max):
            t1_amps[1][i, a - n_b] = 0
    # print(t1_amps)

    # getting t2_external_aa
    for i in range(0, n_a):
        for j in range(0, n_a):
            for a in range(n_a, act_max):
                for b in range(n_a, act_max):
                    t2_amps[0][i, j, a - n_a, b - n_a] = 0

    # getting t2_external_ab
    for i in range(0, n_a):
        for j in range(0, n_b):
            for a in range(n_a, act_max):
                for b in range(n_b, act_max):
                    t2_amps[1][i, j, a - n_a, b - n_b] = 0

    # getting t2_external_bb
    for i in range(0, n_b):
        for j in range(0, n_b):
            for a in range(n_b, act_max):
                for b in range(n_b, act_max):
                    t2_amps[2][i, j, a - n_b, b - n_b] = 0

    # print(t2_amps)
    return t1_amps, t2_amps


def get_many_body_terms(operator):
    constant = of.FermionOperator()
    one_body = of.FermionOperator()
    two_body = of.FermionOperator()
    three_body = of.FermionOperator()
    terms = operator.terms
    for term in terms:
        if len(term) == 0:
            constant += of.FermionOperator(term, terms.get(term))
        elif len(term) == 2:
            one_body += of.FermionOperator(term, terms.get(term))
        elif len(term) == 4:
            two_body += of.FermionOperator(term, terms.get(term))
        elif len(term) == 6:
            three_body += of.FermionOperator(term, terms.get(term))
        else:
            print("Unexpected number of terms: %d" % len(term))
    return (constant, one_body, two_body, three_body)


def as_proj(operator, act_max):
    proj_op = of.FermionOperator()
    const, one_body, two_body, three_body = get_many_body_terms(operator)
    # constant terms
    proj_op += const
    # one-body terms
    terms1 = one_body.terms
    for term in terms1:
        if term[0][0] < act_max:
            if term[1][0] < act_max:
                proj_op += of.FermionOperator(term, terms1.get(term))
    # two-body terms
    terms2 = two_body.terms
    for term in terms2:
        if term[0][0] < act_max:
            if term[1][0] < act_max:
                if term[2][0] < act_max:
                    if term[3][0] < act_max:
                        proj_op += of.FermionOperator(term, terms2.get(term))
    # three-body terms
    terms3 = three_body.terms
    for term in terms3:
        if term[0][0] < act_max:
            if term[1][0] < act_max:
                if term[2][0] < act_max:
                    if term[3][0] < act_max:
                        if term[4][0] < act_max:
                            if term[5][0] < act_max:
                                proj_op += of.FermionOperator(term, terms3.get(term))
    return proj_op


def proj_tens_to_as(term, act_max):
    if np.ndim(term) == 0:
        print("It is a constant term.")
        exit()
    elif np.ndim(term) == 2:
        proj_term = term[0 : 2 * act_max, 0 : 2 * act_max]
    elif np.ndim(term) == 4:
        proj_term = term[
            0 : 2 * act_max, 0 : 2 * act_max, 0 : 2 * act_max, 0 : 2 * act_max
        ]
    elif np.ndim(term) == 6:
        proj_term = term[
            0 : 2 * act_max,
            0 : 2 * act_max,
            0 : 2 * act_max,
            0 : 2 * act_max,
            0 : 2 * act_max,
            0 : 2 * act_max,
        ]
    else:
        print("TODO: implement a projection for a %dD tensor." % (np.ndim(term)))
        exit()
    return proj_term


def get_u(two_elec, config_a, config_b):
    """
    u_{pq} = \sum_{i} <pi||qi>
    """
    n_orb = int(two_elec.shape[0] / 2)
    u = np.zeros((2 * n_orb, 2 * n_orb))
    # two_elec = get_spin_to_spatial(two_elec,n_orb)
    print("Running get_u function and its shape = ", u.shape)
    for p in range(0, n_orb):
        pa = 2 * p
        pb = 2 * p + 1
        for q in range(0, n_orb):
            qa = 2 * q
            qb = 2 * q + 1
            for i in config_a:
                u[pa, qa] += two_elec[pa, 2 * i, qa, 2 * i]
                u[pb, qb] += two_elec[pb, 2 * i, qb, 2 * i]
            for j in config_b:
                u[pa, qa] += two_elec[pa, 2 * j + 1, qa, 2 * j + 1]
                u[pb, qb] += two_elec[pb, 2 * j + 1, qb, 2 * j + 1]

    return u


def antisymmetrize_residual(g_ijab, n_occ, n_orb):
    # antisymmetrize the oovv residual
    Rijab_anti = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    Rijab_anti += np.einsum("ijab->ijab", g_ijab)
    Rijab_anti -= np.einsum("ijab->jiab", g_ijab)
    Rijab_anti -= np.einsum("ijab->ijba", g_ijab)
    Rijab_anti += np.einsum("ijab->jiba", g_ijab)
    return 0.25 * Rijab_anti


def get_integral_blocks(fmat, vmat, n_a, n_b, n_orb):
    fmat_1 = np.array(fmat)
    vmat_1 = np.array(vmat)

    f = {}
    f["oo"] = fmat_1[0 : n_a + n_b, 0 : n_a + n_b]

    f["ov"] = fmat_1[0 : n_a + n_b, n_a + n_b : 2 * n_orb]

    f["vo"] = fmat_1[n_a + n_b : 2 * n_orb, 0 : n_a + n_b]

    f["vv"] = fmat_1[n_a + n_b : 2 * n_orb, n_a + n_b : 2 * n_orb]

    v = {}
    v["oooo"] = vmat_1[0 : n_a + n_b, 0 : n_a + n_b, 0 : n_a + n_b, 0 : n_a + n_b]

    v["ooov"] = vmat_1[
        0 : n_a + n_b, 0 : n_a + n_b, 0 : n_a + n_b, n_a + n_b : 2 * n_orb
    ]

    v["ovoo"] = vmat_1[
        0 : n_a + n_b, n_a + n_b : 2 * n_orb, 0 : n_a + n_b, 0 : n_a + n_b
    ]

    v["ovov"] = vmat_1[
        0 : n_a + n_b, n_a + n_b : 2 * n_orb, 0 : n_a + n_b, n_a + n_b : 2 * n_orb
    ]

    v["oovv"] = vmat_1[
        0 : n_a + n_b, 0 : n_a + n_b, n_a + n_b : 2 * n_orb, n_a + n_b : 2 * n_orb
    ]

    v["vvoo"] = vmat_1[
        n_a + n_b : 2 * n_orb, n_a + n_b : 2 * n_orb, 0 : n_a + n_b, 0 : n_a + n_b
    ]

    v["ovvv"] = vmat_1[
        0 : n_a + n_b,
        n_a + n_b : 2 * n_orb,
        n_a + n_b : 2 * n_orb,
        n_a + n_b : 2 * n_orb,
    ]

    v["vvov"] = vmat_1[
        n_a + n_b : 2 * n_orb,
        n_a + n_b : 2 * n_orb,
        0 : n_a + n_b,
        n_a + n_b : 2 * n_orb,
    ]

    v["vvvv"] = vmat_1[
        n_a + n_b : 2 * n_orb,
        n_a + n_b : 2 * n_orb,
        n_a + n_b : 2 * n_orb,
        n_a + n_b : 2 * n_orb,
    ]

    return f, v


def get_t_blocks(t1_amps, t2_amps):
    t = {}
    t["ov"] = t1_amps
    t["vo"] = t1_amps.transpose()
    t["oovv"] = t2_amps
    t["vvoo"] = t2_amps.transpose()
    return t
