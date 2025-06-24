import ducc
import scipy

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, cc
from pyscf.cc import ccsd

import openfermion as of
from openfermion import *

import numpy as np
import copy as cp


def one_body_to_matrix(operator, n_orb):
    """
    Converts normal-ordered one-body fermionic operator to dense matrix
    F = f_{pq} p^ q
    """
    one_body_mat = np.zeros((2 * n_orb, 2 * n_orb))
    terms = operator.terms
    for term in terms:
        one_body_mat[term[0][0], term[1][0]] = terms.get(term)
    return one_body_mat

def one_body_to_matrix_ph(operator, n_orb, n_occ):
    """
    Converts particle-hole normal-ordered one-body fermionic operator to dense matrix
    """
    one_body_mat = np.zeros((2 * n_orb, 2 * n_orb))
    terms = operator.terms
    for term in terms:
        # OO
        if (term[0][0] < n_occ) and (term[1][0] < n_occ):
            one_body_mat[term[1][0], term[0][0]] += -terms.get(term)
        # OV
        elif not (term[0][1]):
            one_body_mat[term[1][0], term[0][0]] += -terms.get(term)
        # VO
        elif term[1][1]:
            one_body_mat[term[0][0], term[1][0]] += terms.get(term)
        # VV
        elif (term[0][0] >= n_occ) and (term[1][0] >= n_occ):
            one_body_mat[term[0][0], term[1][0]] += terms.get(term)
    return one_body_mat

def two_body_to_tensor(operator, n_orb):
    """
    Converts normal-ordered two-body fermionic operator to dense tensor
    V = v_pqrs p^ q^ s r
    """
    # if (operator.many_body_order != 4):
    # 	print("Error: not a two-body operator")
    # 	exit()
    two_body_tensor = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    terms = operator.terms
    for term in terms:
        two_body_tensor[
            term[0][0], term[1][0], term[3][0], term[2][0]
        ] = 0.25 * terms.get(term)
        two_body_tensor[
            term[1][0], term[0][0], term[3][0], term[2][0]
        ] = -0.25 * terms.get(term)
        two_body_tensor[
            term[0][0], term[1][0], term[2][0], term[3][0]
        ] = -0.25 * terms.get(term)
        two_body_tensor[
            term[1][0], term[0][0], term[2][0], term[3][0]
        ] = 0.25 * terms.get(term)
    return two_body_tensor

def two_body_to_tensor_ph(operator, n_orb, n_occ):
    """
    Converts particle-hole normal-ordered two-body fermionic operator to dense tensor
    """
    two_body_tens = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    terms = operator.terms
    for term in terms:
        p = term[0][0]
        q = term[1][0]
        s = term[2][0]
        r = term[3][0]
        # OOOO
        if (p < n_occ) and (q < n_occ) and (s < n_occ) and (r < n_occ):
            two_body_tens[s, r, q, p] += 0.25 * terms.get(term)
            two_body_tens[s, r, p, q] += -0.25 * terms.get(term)
            two_body_tens[r, s, q, p] += -0.25 * terms.get(term)
            two_body_tens[r, s, p, q] += 0.25 * terms.get(term)
        # OOOV and OOVO
        elif (p < n_occ) and (q >= n_occ) and (s < n_occ) and (r < n_occ):
            two_body_tens[s, r, p, q] += -0.25 * terms.get(term)
            two_body_tens[s, r, q, p] += 0.25 * terms.get(term)
            two_body_tens[r, s, p, q] += 0.25 * terms.get(term)
            two_body_tens[r, s, q, p] += -0.25 * terms.get(term)
        # OVOO and VOOO
        elif (p >= n_occ) and (q < n_occ) and (s < n_occ) and (r < n_occ):
            two_body_tens[r, p, s, q] += -0.25 * terms.get(term)
            two_body_tens[r, p, q, s] += 0.25 * terms.get(term)
            two_body_tens[p, r, s, q] += 0.25 * terms.get(term)
            two_body_tens[p, r, q, s] += -0.25 * terms.get(term)
        # OOVV and VVOO
        elif (p >= n_occ) and (q >= n_occ) and (s < n_occ) and (r < n_occ):
            two_body_tens[s, r, q, p] += 0.125 * terms.get(term)
            two_body_tens[s, r, p, q] += -0.125 * terms.get(term)
            two_body_tens[r, s, q, p] += -0.125 * terms.get(term)
            two_body_tens[r, s, p, q] += 0.125 * terms.get(term)

            two_body_tens[p, q, r, s] += 0.125 * terms.get(term)
            two_body_tens[p, q, s, r] += -0.125 * terms.get(term)
            two_body_tens[q, p, r, s] += -0.125 * terms.get(term)
            two_body_tens[q, p, s, r] += 0.125 * terms.get(term)
        # OVOV and VOOV and OVVO and VOVO
        elif (p >= n_occ) and (q < n_occ) and (s >= n_occ) and (r < n_occ):
            two_body_tens[r, p, q, s] += 0.25 * terms.get(term)
            two_body_tens[p, r, q, s] += -0.25 * terms.get(term)
            two_body_tens[r, p, s, q] += -0.25 * terms.get(term)
            two_body_tens[p, r, s, q] += 0.25 * terms.get(term)
        # VVVO and VVOV
        elif (p >= n_occ) and (q >= n_occ) and (s < n_occ) and (r >= n_occ):
            two_body_tens[p, q, r, s] += 0.25 * terms.get(term)
            two_body_tens[q, p, r, s] += -0.25 * terms.get(term)
            two_body_tens[p, q, s, r] += -0.25 * terms.get(term)
            two_body_tens[q, p, s, r] += 0.25 * terms.get(term)
        # VOVV and OVVV
        elif (p >= n_occ) and (q >= n_occ) and (s >= n_occ) and (r < n_occ):
            two_body_tens[p, r, s, q] += 0.25 * terms.get(term)
            two_body_tens[p, r, q, s] += -0.25 * terms.get(term)
            two_body_tens[r, p, s, q] += -0.25 * terms.get(term)
            two_body_tens[r, p, q, s] += 0.25 * terms.get(term)
        # VVVV
        elif (p >= n_occ) and (q >= n_occ) and (s >= n_occ) and (r >= n_occ):
            two_body_tens[p, q, r, s] += 0.25 * terms.get(term)
            two_body_tens[p, q, s, r] += -0.25 * terms.get(term)
            two_body_tens[q, p, r, s] += -0.25 * terms.get(term)
            two_body_tens[q, p, s, r] += 0.25 * terms.get(term)
    return two_body_tens

def three_body_to_tensor(operator, n_orb):
    """
    Converts normal_ordered three-body fermionic operator to dense tensor
    W = w_pqrstu p^ q^ r^ u t s
    """
    # if (operator.many_body_order != 6):
    # 	print("Error: not a three-body operator")
    # 	exit()
    three_body_tensor = np.zeros(
        (2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb)
    )
    terms = operator.terms
    for term in terms:
        coeff = terms.get(term)
        p = term[0][0]
        q = term[1][0]
        r = term[2][0]
        s = term[5][0]
        t = term[4][0]
        u = term[3][0]

        three_body_tensor[p, q, r, s, t, u] = (1.0 / 36.0) * coeff
        three_body_tensor[p, q, r, s, u, t] = -(1.0 / 36.0) * coeff
        three_body_tensor[p, q, r, t, s, u] = -(1.0 / 36.0) * coeff
        three_body_tensor[p, q, r, t, u, s] = (1.0 / 36.0) * coeff
        three_body_tensor[p, q, r, u, s, t] = (1.0 / 36.0) * coeff
        three_body_tensor[p, q, r, u, t, s] = -(1.0 / 36.0) * coeff

        three_body_tensor[p, r, q, s, t, u] = -(1.0 / 36.0) * coeff
        three_body_tensor[p, r, q, s, u, t] = (1.0 / 36.0) * coeff
        three_body_tensor[p, r, q, t, s, u] = (1.0 / 36.0) * coeff
        three_body_tensor[p, r, q, t, u, s] = -(1.0 / 36.0) * coeff
        three_body_tensor[p, r, q, u, s, t] = -(1.0 / 36.0) * coeff
        three_body_tensor[p, r, q, u, t, s] = (1.0 / 36.0) * coeff

        three_body_tensor[q, p, r, s, t, u] = -(1.0 / 36.0) * coeff
        three_body_tensor[q, p, r, s, u, t] = (1.0 / 36.0) * coeff
        three_body_tensor[q, p, r, t, s, u] = (1.0 / 36.0) * coeff
        three_body_tensor[q, p, r, t, u, s] = -(1.0 / 36.0) * coeff
        three_body_tensor[q, p, r, u, s, t] = -(1.0 / 36.0) * coeff
        three_body_tensor[q, p, r, u, t, s] = (1.0 / 36.0) * coeff

        three_body_tensor[q, r, p, s, t, u] = (1.0 / 36.0) * coeff
        three_body_tensor[q, r, p, s, u, t] = -(1.0 / 36.0) * coeff
        three_body_tensor[q, r, p, t, s, u] = -(1.0 / 36.0) * coeff
        three_body_tensor[q, r, p, t, u, s] = (1.0 / 36.0) * coeff
        three_body_tensor[q, r, p, u, s, t] = (1.0 / 36.0) * coeff
        three_body_tensor[q, r, p, u, t, s] = -(1.0 / 36.0) * coeff

        three_body_tensor[r, p, q, s, t, u] = (1.0 / 36.0) * coeff
        three_body_tensor[r, p, q, s, u, t] = -(1.0 / 36.0) * coeff
        three_body_tensor[r, p, q, t, s, u] = -(1.0 / 36.0) * coeff
        three_body_tensor[r, p, q, t, u, s] = (1.0 / 36.0) * coeff
        three_body_tensor[r, p, q, u, s, t] = (1.0 / 36.0) * coeff
        three_body_tensor[r, p, q, u, t, s] = -(1.0 / 36.0) * coeff

        three_body_tensor[r, q, p, s, t, u] = -(1.0 / 36.0) * coeff
        three_body_tensor[r, q, p, s, u, t] = (1.0 / 36.0) * coeff
        three_body_tensor[r, q, p, t, s, u] = (1.0 / 36.0) * coeff
        three_body_tensor[r, q, p, t, u, s] = -(1.0 / 36.0) * coeff
        three_body_tensor[r, q, p, u, s, t] = -(1.0 / 36.0) * coeff
        three_body_tensor[r, q, p, u, t, s] = (1.0 / 36.0) * coeff

    return three_body_tensor

def make_full_one(big, small, n_a, n_b, n_orb):
    if n_a + n_b == small.shape[0]:
        ind_0 = slice(0, n_a + n_b)
    else:
        ind_0 = slice(n_a + n_b, 2 * n_orb)
    if n_a + n_b == small.shape[1]:
        ind_1 = slice(0, n_a + n_b)
    else:
        ind_1 = slice(n_a + n_b, 2 * n_orb)
    big[ind_0, ind_1] = small
    return big

def make_full_two(big, small, n_a, n_b, n_orb):
    if n_a + n_b == small.shape[0]:
        ind_0 = slice(0, n_a + n_b)
    else:
        ind_0 = slice(n_a + n_b, 2 * n_orb)
    if n_a + n_b == small.shape[1]:
        ind_1 = slice(0, n_a + n_b)
    else:
        ind_1 = slice(n_a + n_b, 2 * n_orb)
    if n_a + n_b == small.shape[2]:
        ind_2 = slice(0, n_a + n_b)
    else:
        ind_2 = slice(n_a + n_b, 2 * n_orb)
    if n_a + n_b == small.shape[3]:
        ind_3 = slice(0, n_a + n_b)
    else:
        ind_3 = slice(n_a + n_b, 2 * n_orb)
    big[ind_0, ind_1, ind_2, ind_3] = small
    return big

def make_full_three(big, small, n_a, n_b, n_orb):
    if n_a + n_b == small.shape[0]:
        ind_0 = slice(0, n_a + n_b)
    else:
        ind_0 = slice(n_a + n_b, 2 * n_orb)
    if n_a + n_b == small.shape[1]:
        ind_1 = slice(0, n_a + n_b)
    else:
        ind_1 = slice(n_a + n_b, 2 * n_orb)
    if n_a + n_b == small.shape[2]:
        ind_2 = slice(0, n_a + n_b)
    else:
        ind_2 = slice(n_a + n_b, 2 * n_orb)
    if n_a + n_b == small.shape[3]:
        ind_3 = slice(0, n_a + n_b)
    else:
        ind_3 = slice(n_a + n_b, 2 * n_orb)
    if n_a + n_b == small.shape[4]:
        ind_4 = slice(0, n_a + n_b)
    else:
        ind_4 = slice(n_a + n_b, 2 * n_orb)
    if n_a + n_b == small.shape[5]:
        ind_5 = slice(0, n_a + n_b)
    else:
        ind_5 = slice(n_a + n_b, 2 * n_orb)
    big[ind_0, ind_1, ind_2, ind_3, ind_4, ind_5] = small
    return big

def transform_t_spatial_to_spin(t1_amps, t2_amps, n_a, n_b, n_orb):
    # expanding the t1 amplitude into alpha and beta space
    t1 = np.zeros((n_a + n_b, 2 * n_orb - n_a - n_b))
    for i in range(0, n_a):
        for a in range(n_a, n_orb):
            ia = 2 * i
            aa = 2 * a
            t1[ia, aa - (n_a + n_b)] = t1_amps[0][i, a - n_a]
    for i in range(0, n_b):
        for a in range(n_b, n_orb):
            ib = 2 * i + 1
            ab = 2 * a + 1
            t1[ib, ab - (n_a + n_b)] = t1_amps[1][i, a - n_b]

    # print(t1.shape)
    # print(t1)

    # expanding the t2 amplitude_aa
    t2 = np.zeros((n_a + n_b, n_a + n_b, 2 * n_orb - n_a - n_b, 2 * n_orb - n_a - n_b))
    for i in range(0, n_a):
        ia = 2 * i
        for j in range(0, n_a):
            ja = 2 * j
            for a in range(n_a, n_orb):
                aa = 2 * a
                for b in range(n_a, n_orb):
                    ba = 2 * b
                    t2[ia, ja, aa - (n_a + n_b), ba - (n_a + n_b)] = t2_amps[0][
                        i, j, a - n_a, b - n_a
                    ]

    # expanding the t2 amplitude_ab
    for i in range(0, n_a):
        ia = 2 * i
        for j in range(0, n_b):
            jb = 2 * j + 1
            for a in range(n_a, n_orb):
                aa = 2 * a
                for b in range(n_b, n_orb):
                    bb = 2 * b + 1
                    t2[ia, jb, aa - (n_a + n_b), bb - (n_a + n_b)] = t2_amps[1][
                        i, j, a - n_a, b - n_b
                    ]
                    t2[jb, ia, bb - (n_a + n_b), aa - (n_a + n_b)] = t2_amps[1][
                        i, j, a - n_a, b - n_b
                    ]
                    t2[ia, jb, bb - (n_a + n_b), aa - (n_a + n_b)] = -t2_amps[1][
                        i, j, a - n_a, b - n_b
                    ]
                    t2[jb, ia, aa - (n_a + n_b), bb - (n_a + n_b)] = -t2_amps[1][
                        i, j, a - n_a, b - n_b
                    ]

    # expanding the t2 amplitude_bb
    for i in range(0, n_b):
        ib = 2 * i + 1
        for j in range(0, n_b):
            jb = 2 * j + 1
            for a in range(n_b, n_orb):
                ab = 2 * a + 1
                for b in range(n_b, n_orb):
                    bb = 2 * b + 1
                    t2[ib, jb, ab - (n_a + n_b), bb - (n_a + n_b)] = t2_amps[2][
                        i, j, a - n_b, b - n_b
                    ]

    # print(t2.shape)
    return t1, t2

def get_spin_to_spatial(integral, n_orb):
    if np.ndim(integral) == 0:
        print("It is a constant term.")
        exit()
    elif np.ndim(integral) == 2:
        integral_spatial = np.zeros((n_orb, n_orb))
        for p in range(0, n_orb):
            pa = 2 * p
            for q in range(0, n_orb):
                qa = 2 * q
                integral_spatial[p, q] = integral[pa, qa]
    elif np.ndim(integral) == 4:
        integral_spatial = np.zeros((n_orb, n_orb, n_orb, n_orb))
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
                        # vmat_spatial[p,q,r,s] = vmat[pa,qb,ra,sb]
                        integral_spatial[p, r, q, s] = integral[pa, qb, ra, sb]
                        # integral_spatial[p,r,q,s] = integral[pb,qa,rb,sa]
                        # vmat_spatial[p,q,r,s] = vmat[pa, ra,qb,sb]

    return integral_spatial

def get_spin_to_spatial_1(integral, n_orb):
    if np.ndim(integral) == 0:
        print("It is a constant term.")
        exit()

    elif np.ndim(integral) == 2:
        integral_spatial = np.zeros((n_orb, n_orb))
        for p in range(0, n_orb):
            pa = 2 * p
            for q in range(0, n_orb):
                qa = 2 * q
                integral_spatial[p, q] = integral[pa, qa]
                if p != q:
                    integral_spatial[q, p] = integral[pa, qa]
                    
    elif np.ndim(integral) == 4:
        integral_spatial = np.zeros((n_orb, n_orb, n_orb, n_orb))
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
                        #prqs
                        integral_spatial[p, r, q, s] = integral[pa, qb, ra, sb]
                        #qspr
                        if integral_spatial[q, s, p, r] == 0:
                            integral_spatial[q, s, p, r] = integral[pa, qb, ra, sb]
                        #prsq
                        if integral_spatial[p, r, s, q] == 0:
                            integral_spatial[p, r, s, q] = integral[pa, qb, ra, sb]
                        #sqpr
                        if integral_spatial[s, q, p, r] == 0:
                            integral_spatial[s, q, p, r] = integral[pa, qb, ra, sb]
                        #rpqs
                        if integral_spatial[r, p, q, s] == 0:
                            integral_spatial[r, p, q, s] = integral[pa, qb, ra, sb]
                        #qsrp
                        if integral_spatial[q, s, r, p] == 0:
                            integral_spatial[q, s, r, p] = integral[pa, qb, ra, sb]
                        #rpsq
                        if integral_spatial[r, p, s, q] == 0:
                            integral_spatial[r, p, s, q] = integral[pa, qb, ra, sb]
                        #sqrp
                        if integral_spatial[s, q, r, p] == 0:
                            integral_spatial[s, q, r, p] = integral[pa, qb, ra, sb]

    return integral_spatial

def get_spatial_to_spin(integral, n_orb):
    A = np.array([[1, 0], [0, 0]])
    B = np.array([[0, 0], [0, 1]])
    AA = np.einsum("pq,rs->pqrs", A, A)
    AB = np.einsum("pq,rs->pqrs", A, B)
    BA = np.einsum("pq,rs->pqrs", B, A)
    BB = np.einsum("pq,rs->pqrs", B, B)

    if np.ndim(integral) == 0:
        print("It is a constant term.")
        exit()

    elif np.ndim(integral) == 2:
        integral_spin = np.kron(integral, A) + np.kron(integral, B)

    elif np.ndim(integral) == 4:
        integral_spin = (
            np.kron(integral, AA)
            + np.kron(integral, AB)
            + np.kron(integral, BA)
            + np.kron(integral, BB)
        )
    return integral_spin

def t1_to_op(t1_amps):
	n_a = t1_amps[0].shape[0]
	n_virt_a = t1_amps[0].shape[1]
	s1_op = of.FermionOperator()
	for i in range(0,n_a):
		ia = 2*i 
		ib = 2*i+1 
		for a in range(0,n_virt_a):
			aa = 2*a + 2*n_a
			ab = 2*a+1 + 2*n_a
			s1_op += of.FermionOperator(((aa,1),(ia,0)),  t1_amps[0][i,a])
			s1_op += of.FermionOperator(((ia,1),(aa,0)), -t1_amps[0][i,a])
			s1_op += of.FermionOperator(((ab,1),(ib,0)),  t1_amps[1][i,a])
			s1_op += of.FermionOperator(((ib,1),(ab,0)), -t1_amps[1][i,a])
	return s1_op

def t2_to_op(t2_amps):
	n_a = t2_amps[0].shape[0]
	n_virt_a = t2_amps[0].shape[2]
	s2_op = of.FermionOperator()
	# aaaa/bbbb
	for i in range(0,n_a):
		ia = 2*i 
		ib = 2*i+1 
		for j in range(i+1,n_a):
			ja = 2*j 
			jb = 2*j+1 
			for a in range(0,n_virt_a):
				aa = 2*a + 2*n_a
				ab = 2*a+1 + 2*n_a
				for b in range(a+1,n_virt_a):
					ba = 2*b + 2*n_a
					bb = 2*b+1 + 2*n_a 
					s2_op += of.FermionOperator(((aa,1),(ba,1),(ja,0),(ia,0)),  t2_amps[0][i,j,a,b])
					s2_op += of.FermionOperator(((ia,1),(ja,1),(ba,0),(aa,0)), -t2_amps[0][i,j,a,b])
					s2_op += of.FermionOperator(((ab,1),(bb,1),(jb,0),(ib,0)),  t2_amps[2][i,j,a,b])
					s2_op += of.FermionOperator(((ib,1),(jb,1),(bb,0),(ab,0)), -t2_amps[2][i,j,a,b])
	# abab
	for i in range(0,n_a):
		ia = 2*i 
		for j in range(0,n_a):
			jb = 2*j+1 
			for a in range(0,n_virt_a):
				aa = 2*a + 2*n_a 
				for b in range(0,n_virt_a):
					bb = 2*b+1 + 2*n_a 
					s2_op += of.FermionOperator(((aa,1),(bb,1),(jb,0),(ia,0)),  t2_amps[1][i,j,a,b])
					s2_op += of.FermionOperator(((ia,1),(jb,1),(bb,0),(aa,0)), -t2_amps[1][i,j,a,b])
	return s2_op 

def t1_to_ext(t1,n_act):
	n_a = t1[0].shape[0]
	n_virt_a = t1[0].shape[1]
	n_virt_int_a = n_act - n_a 
	for i in range(0,n_a):
		for a in range(0,n_virt_int_a):
			t1[0][i,a] = 0
			t1[1][i,a] = 0
	return t1 

def t2_to_ext(t2,n_act):
	n_a = t2[0].shape[0]
	n_virt_a = t2[0].shape[2]
	n_virt_int_a = n_act-n_a 
	for i in range(0,n_a):
		for j in range(0,n_a):
			for a in range(0,n_virt_int_a):
				for b in range(0,n_virt_int_a):
					t2[0][i,j,a,b] = 0
					t2[1][i,j,a,b] = 0
					t2[2][i,j,a,b] = 0
	return t2 

def one_body_to_op(one_body_mat,n_occ,n_orb):
	one_body_op = of.FermionOperator()
	for p in range(0,n_occ):
		for q in range(0,n_occ):
			# O|O
			one_body_op += of.FermionOperator(((q,0),(p,1)), -one_body_mat[p,q])
	for p in range(0,n_occ):
		for q in range(n_occ,2*n_orb):
			# O|V
			one_body_op += of.FermionOperator(((p,1),(q,0)),  one_body_mat[p,q])
			# V|O
			one_body_op += of.FermionOperator(((q,1),(p,0)),  one_body_mat[q,p])
	for p in range(n_occ,2*n_orb):
		for q in range(n_occ,2*n_orb):
			# V|V
			one_body_op += of.FermionOperator(((p,1),(q,0)),  one_body_mat[p,q])
	return one_body_op 

def two_body_to_op(two_body_tens,n_occ,n_orb):
	two_body_op = of.FermionOperator()
	for p in range(0,n_occ):
		for q in range(0,n_occ):
			for r in range(0,n_occ):
				for s in range(0,n_occ):
					# OO|OO
					two_body_op += of.FermionOperator(((s,0),(r,0),(p,1),(q,1)),  two_body_tens[p,q,r,s])
	for p in range(0,n_occ):
		for q in range(0,n_occ):
			for r in range(0,n_occ):
				for s in range(n_occ,2*n_orb):
					# OO|OV
					two_body_op += of.FermionOperator(((r,0),(s,0),(p,1),(q,1)), -two_body_tens[p,q,r,s])
					# OO|VO
					two_body_op += of.FermionOperator(((r,0),(s,0),(p,1),(q,1)),  two_body_tens[p,q,s,r])
					# OV|OO
					two_body_op += of.FermionOperator(((s,1),(q,0),(r,0),(p,1)), -two_body_tens[p,s,r,q])
					# VO|OO
					two_body_op += of.FermionOperator(((s,1),(p,0),(r,0),(q,1)),  two_body_tens[s,q,r,p])
	for p in range(0,n_occ):
		for q in range(0,n_occ):
			for r in range(n_occ,2*n_orb):
				for s in range(n_occ,2*n_orb):
					# OO|VV
					two_body_op += of.FermionOperator(((s,0),(r,0),(p,1),(q,1)),  two_body_tens[p,q,r,s])
					# OV|OV
					two_body_op += of.FermionOperator(((r,1),(q,0),(s,0),(p,1)),  two_body_tens[p,r,q,s])
					# VO|OV
					two_body_op += of.FermionOperator(((r,1),(p,0),(s,0),(q,1)), -two_body_tens[r,q,p,s])
					# OV|VO
					two_body_op += of.FermionOperator(((s,1),(q,0),(r,0),(p,1)), -two_body_tens[p,s,r,q])
					# VO|VO
					two_body_op += of.FermionOperator(((s,1),(p,0),(r,0),(q,1)),  two_body_tens[s,q,r,p])
					# VV|OO
					two_body_op += of.FermionOperator(((r,1),(s,1),(q,0),(p,0)),  two_body_tens[r,s,p,q])
	for p in range(0,n_occ):
		for q in range(n_occ,2*n_orb):
			for r in range(n_occ,2*n_orb):
				for s in range(n_occ,2*n_orb):
					# OV|VV
					two_body_op += of.FermionOperator(((q,1),(s,0),(r,0),(p,1)), -two_body_tens[p,q,r,s])
					# VO|VV
					two_body_op += of.FermionOperator(((q,1),(s,0),(r,0),(p,1)),  two_body_tens[q,p,r,s])
					# VV|OV
					two_body_op += of.FermionOperator(((r,1),(q,1),(p,0),(s,0)), -two_body_tens[r,q,p,s])
					# VV|VO
					two_body_op += of.FermionOperator(((s,1),(q,1),(p,0),(r,0)),  two_body_tens[s,q,r,p])
	for p in range(n_occ,2*n_orb):
		for q in range(n_occ,2*n_orb):
			for r in range(n_occ,2*n_orb):
				for s in range(n_occ,2*n_orb):
					# VVVV
					two_body_op += of.FermionOperator(((p,1),(q,1),(s,0),(r,0)),  two_body_tens[p,q,r,s])
	return two_body_op

def three_body_to_op(three_body_tens,n_occ,n_orb):
	"""
	Untested, use at your own risk
	"""
	three_body_op = of.FermionOperator()
	for p in range(0,n_occ):
		for q in range(0,n_occ):
			for r in range(0,n_occ):
				for s in range(0,n_occ):
					for t in range(0,n_occ):
						for u in range(0,n_occ):
							# OOO|OOO
							three_body_op += of.FermionOperator(((u,0),(t,0),(s,0),(p,1),(q,1),(r,1)), -three_body_tens[p,q,r,s,t,u])
	for p in range(0,n_occ):
		for q in range(0,n_occ):
			for r in range(0,n_occ):
				for s in range(0,n_occ):
					for t in range(0,n_occ):
						for u in range(n_occ,2*n_orb):
							# OOO|OOV
							three_body_op += of.FermionOperator(((t,0),(s,0),(u,0),(p,1),(q,1),(r,1)), -three_body_tens[p,q,r,s,t,u])
							# OOO|OVO
							three_body_op += of.FermionOperator(((t,0),(s,0),(u,0),(p,1),(q,1),(r,1)),  three_body_tens[p,q,r,s,u,t])
							# OOO|VOO
							three_body_op += of.FermionOperator(((s,0),(t,0),(u,0),(p,1),(q,1),(r,1)), -three_body_tens[p,q,r,u,t,s])
							# OOV|OOO
							three_body_op += of.FermionOperator(((u,1),(r,0),(t,0),(s,0),(p,1),(q,1)),  three_body_tens[p,q,u,s,t,r])
							# OVO|OOO
							three_body_op += of.FermionOperator(((u,1),(q,0),(t,0),(s,0),(p,1),(r,1)), -three_body_tens[p,u,r,s,t,q])
							# VOO|OOO
							three_body_op += of.FermionOperator(((u,1),(p,0),(t,0),(s,0),(q,1),(r,1)),  three_body_tens[u,q,r,s,t,p])
	for p in range(0,n_occ):
		for q in range(0,n_occ):
			for r in range(0,n_occ):
				for s in range(0,n_occ):
					for t in range(n_occ,2*n_orb):
						for u in range(n_occ,2*n_orb):
							# OOO|OVV
							three_body_op += of.FermionOperator(((s,0),(u,0),(t,0),(p,1),(q,1),(r,1)), -three_body_tens[p,q,r,s,t,u])
							# OOO|VOV
							three_body_op += of.FermionOperator(((s,0),(u,0),(t,0),(p,1),(q,1),(r,1)),  three_body_tens[p,q,r,t,s,u])
							# OOV|OOV
							three_body_op += of.FermionOperator(((t,1),(r,0),(s,0),(u,0),(p,1),(q,1)),  three_body_tens[p,q,t,s,r,u])
							# OVO|OOV
							three_body_op += of.FermionOperator(((t,1),(q,0),(s,0),(u,0),(p,1),(r,1)), -three_body_tens[p,t,r,s,q,u])
							# VOO|OOV
							three_body_op += of.FermionOperator(((t,1),(p,0),(s,0),(u,0),(q,1),(r,1)),  three_body_tens[t,q,r,s,p,u])
							# OOO|VVO
							three_body_op += of.FermionOperator(((s,0),(t,0),(u,0),(p,1),(q,1),(r,1)), -three_body_tens[p,q,r,u,t,s])
							# OOV|OVO
							three_body_op += of.FermionOperator(((u,1),(r,0),(s,0),(t,0),(p,1),(q,1)), -three_body_tens[p,q,u,s,t,r])
							# OVO|OVO
							three_body_op += of.FermionOperator(((u,1),(q,0),(s,0),(t,0),(p,1),(r,1)),  three_body_tens[p,u,r,s,t,q])
							# VOO|OVO
							three_body_op += of.FermionOperator(((u,1),(p,0),(s,0),(t,0),(q,1),(r,1)), -three_body_tens[u,q,r,s,t,p])
							# OOV|VOO
							three_body_op += of.FermionOperator(((t,1),(s,0),(r,0),(u,0),(p,1),(q,1)),  three_body_tens[p,q,t,u,r,s])
							# OVO|VOO
							three_body_op += of.FermionOperator(((t,1),(s,0),(q,0),(u,0),(p,1),(r,1)), -three_body_tens[p,t,r,u,q,s])
							# VOO|VOO
							three_body_op += of.FermionOperator(((t,1),(s,0),(p,0),(u,0),(q,1),(r,1)),  three_body_tens[t,q,r,u,p,s])
							# OVV|OOO
							three_body_op += of.FermionOperator(((t,1),(u,1),(r,0),(q,0),(s,0),(p,1)), -three_body_tens[p,t,u,s,q,r])
							# VOV|OOO
							three_body_op += of.FermionOperator(((t,1),(u,1),(r,0),(p,0),(s,0),(q,1)),  three_body_tens[t,q,u,s,p,r])
							# VVO|OOO
							three_body_op += of.FermionOperator(((t,1),(u,1),(q,0),(p,0),(s,0),(r,1)), -three_body_tens[t,u,r,s,p,q])
	for i in range(0,n_occ):
		for j in range(0,n_occ):
			for k in range(0,n_occ):
				for a in range(n_occ,2*n_orb):
					for b in range(n_occ,2*n_orb):
						for c in range(n_occ,2*n_orb):
							# OOO|VVV
							three_body_op += of.FermionOperator(((c,0),(b,0),(a,0),(i,1),(j,1),(k,1)), -three_body_tens[i,j,k,a,b,c])
							# OOV|OVV
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)),  three_body_tens[i,j,a,k,b,c])
							# OVO|OVV
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)), -three_body_tens[i,a,j,k,b,c])
							# VOO|OVV
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)),  three_body_tens[a,i,j,k,b,c])
							# OOV|VOV
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)), -three_body_tens[i,j,a,b,k,c])
							# OVO|VOV
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)),  three_body_tens[i,a,j,b,k,c])
							# VOO|VOV
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)), -three_body_tens[a,i,j,b,k,c])
							# OOV|VVO
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)),  three_body_tens[i,j,a,b,c,k])
							# OVO|VVO
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)), -three_body_tens[i,a,j,b,c,k])
							# VOO|VVO
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)),  three_body_tens[a,i,j,b,c,k])
							# OVV|OOV
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)),  three_body_tens[i,a,b,j,k,c])
							# VOV|OOV
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)), -three_body_tens[a,i,b,j,k,c])
							# VVO|OOV
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)),  three_body_tens[a,b,i,j,k,c])
							# OVV|OVO
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)), -three_body_tens[i,a,b,j,c,k])
							# VOV|OVO
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)),  three_body_tens[a,i,b,j,c,k])
							# VVO|OVO
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)), -three_body_tens[a,b,i,j,c,k])
							# OVV|VOO
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)),  three_body_tens[i,a,b,c,j,k])
							# VOV|VOO
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)), -three_body_tens[a,i,b,c,j,k])
							# VVO|VOO
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)),  three_body_tens[a,b,i,c,j,k])
							# VVV|OOO
							three_body_op += of.FermionOperator(((a,1),(b,1),(c,1),(k,0),(j,0),(i,0)),  three_body_tens[a,b,c,i,j,k])
	for p in range(0,n_occ):
		for q in range(0,n_occ):
			for r in range(n_occ,2*n_orb):
				for s in range(n_occ,2*n_orb):
					for t in range(n_occ,2*n_orb):
						for u in range(n_occ,2*n_orb):
							# OOV|VVV
							three_body_op += of.FermionOperator(((r,1),(u,0),(t,0),(s,0),(p,1),(q,1)),  three_body_tens[p,q,r,s,t,u])
							# OVO|VVV
							three_body_op += of.FermionOperator(((r,1),(u,0),(t,0),(s,0),(p,1),(q,1)), -three_body_tens[p,r,q,s,t,u])
							# VOO|VVV
							three_body_op += of.FermionOperator(((r,1),(u,0),(t,0),(s,0),(q,1),(p,1)),  three_body_tens[r,q,p,s,t,u])
							# OVV|OVV
							three_body_op += of.FermionOperator(((s,1),(r,1),(q,0),(u,0),(t,0),(p,1)), -three_body_tens[p,s,r,q,t,u])
							# VOV|OVV
							three_body_op += of.FermionOperator(((s,1),(r,1),(p,0),(u,0),(t,0),(q,1)),  three_body_tens[s,q,r,p,t,u])
							# VVO|OVV
							three_body_op += of.FermionOperator(((s,1),(r,1),(p,0),(u,0),(t,0),(q,1)), -three_body_tens[s,r,q,p,t,u])
							# OVV|VOV
							three_body_op += of.FermionOperator(((t,1),(r,1),(q,0),(u,0),(s,0),(p,1)),  three_body_tens[p,t,r,s,q,u])
							# VOV|VOV
							three_body_op += of.FermionOperator(((t,1),(r,1),(p,0),(u,0),(s,0),(q,1)), -three_body_tens[t,q,r,s,p,u])
							# VVO|VOV
							three_body_op += of.FermionOperator(((t,1),(r,1),(p,0),(u,0),(s,0),(q,1)),  three_body_tens[t,r,q,s,p,u])
							# OVV|VVO
							three_body_op += of.FermionOperator(((u,1),(r,1),(q,0),(t,0),(s,0),(p,1)), -three_body_tens[p,u,r,s,t,q])
							# VOV|VVO
							three_body_op += of.FermionOperator(((u,1),(r,1),(p,0),(t,0),(s,0),(q,1)),  three_body_tens[u,q,r,s,t,p])
							# VVO|VVO
							three_body_op += of.FermionOperator(((u,1),(r,1),(p,0),(t,0),(s,0),(q,1)), -three_body_tens[u,r,q,s,t,p])
							# VVV|OOV
							three_body_op += of.FermionOperator(((s,1),(t,1),(r,1),(q,0),(p,0),(u,0)),  three_body_tens[s,t,r,p,q,u])
							# VVV|OVO
							three_body_op += of.FermionOperator(((s,1),(u,1),(r,1),(q,0),(p,0),(t,0)), -three_body_tens[s,u,r,p,t,q])
							# VVV|VOO
							three_body_op += of.FermionOperator(((t,1),(u,1),(r,1),(q,0),(p,0),(s,0)),  three_body_tens[t,u,r,s,p,q])
	for p in range(0,n_occ):
		for q in range(n_occ,2*n_orb):
			for r in range(n_occ,2*n_orb):
				for s in range(n_occ,2*n_orb):
					for t in range(n_occ,2*n_orb):
						for u in range(n_occ,2*n_orb):
							# OVV|VVV
							three_body_op += of.FermionOperator(((q,1),(r,1),(u,0),(t,0),(s,0),(p,1)), -three_body_tens[p,q,r,s,t,u])
							# VOV|VVV
							three_body_op += of.FermionOperator(((q,1),(r,1),(u,0),(t,0),(s,0),(p,1)),  three_body_tens[q,p,r,s,t,u])
							# VVO|VVV
							three_body_op += of.FermionOperator(((r,1),(q,1),(u,0),(t,0),(s,0),(p,1)), -three_body_tens[r,q,p,s,t,u])
							# VVV|OVV
							three_body_op += of.FermionOperator(((s,1),(q,1),(r,1),(p,0),(u,0),(t,0)),  three_body_tens[s,q,r,p,t,u])
							# VVV|VOV
							three_body_op += of.FermionOperator(((t,1),(q,1),(r,1),(p,0),(u,0),(s,0)), -three_body_tens[t,q,r,s,p,u])
							# VVV|VVO
							three_body_op += of.FermionOperator(((u,1),(q,1),(r,1),(p,0),(t,0),(s,0)),  three_body_tens[u,q,r,s,t,p])
	for p in range(n_occ,2*n_orb):
		for q in range(n_occ,2*n_orb):
			for r in range(n_occ,2*n_orb):
				for s in range(n_occ,2*n_orb):
					for t in range(n_occ,2*n_orb):
						for u in range(n_occ,2*n_orb):
							# VVV|VVV
							three_body_op += of.FermionOperator(((p,1),(q,1),(r,1),(u,0),(t,0),(s,0)),  three_body_tens[p,q,r,s,t,u])
	return three_body_op 

def one_body_mat2dic(mat,n_occ,n_act,n_orb):
	dic = {
		"oo": mat[0:n_occ,0:n_occ],
		"ov": mat[0:n_occ,n_occ:2*n_act],
		"vo": mat[n_occ:2*n_act,0:n_occ],
		"vv": mat[n_occ:2*n_act,n_occ:2*n_act]
	}
	if(n_orb > n_act):
		dic["oV"] = mat[0:n_occ,2*n_act:2*n_orb]
		dic["Vo"] = mat[2*n_act:2*n_orb,0:n_occ]
		dic["vV"] = mat[n_occ:2*n_act,2*n_act:2*n_orb]
		dic["Vv"] = mat[2*n_act:2*n_orb,n_occ:2*n_act]
		dic["VV"] = mat[2*n_act:2*n_orb,2*n_act:2*n_orb]
	return dic 

def one_body_dic2mat(dic,n_occ,n_act,n_orb):
	if(n_orb > n_act):
		mat = np.zeros((2*n_orb,2*n_orb))
		for key in dic.keys():
			if key == "oo":
				mat[0:n_occ,0:n_occ] = dic["oo"]
			elif key == "oo":
				mat[0:n_occ,0:n_occ] = dic["oo"]
			elif key == "ov":
				mat[0:n_occ,n_occ:2*n_act] = dic["ov"]
			elif key == "vo":
				mat[n_occ:2*n_act,0:n_occ] = dic["vo"]
			elif key == "vv":
				mat[n_occ:2*n_act,n_occ:2*n_act] = dic["vv"]
			elif key == "oV":
				mat[0:n_occ,2*n_act:2*n_orb] = dic["oV"]
			elif key == "Vo":
				mat[2*n_act:2*n_orb,0:n_occ] = dic["Vo"]
			elif key == "vV":  
				mat[n_occ:2*n_act,2*n_act:2*n_orb] = dic["vV"] 
			elif key == "Vv":
				mat[2*n_act:2*n_orb,n_occ:2*n_act] = dic["Vv"] 
			elif key == "VV":
				mat[2*n_act:2*n_orb,2*n_act:2*n_orb] = dic["VV"]
		return mat 
	else:
		mat = np.zeros((2*n_act,2*n_act))
		for key in dic.keys():
			if key == "oo":
				mat[0:n_occ,0:n_occ] = dic["oo"]
			elif key == "ov":
				mat[0:n_occ,n_occ:2*n_act] = dic["ov"]
			elif key == "vo":
				mat[n_occ:2*n_act,0:n_occ] = dic["vo"]
			elif key == "vv":
				mat[n_occ:2*n_act,n_occ:2*n_act] = dic["vv"]
		return mat 

def two_body_ten2dic(ten,n_occ,n_act,n_orb):
	dic = {
		"oooo": ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ],
		"ooov": ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act],
		"oovv": ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act],
		"ovoo": ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ],
		"ovov": ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act],
		"ovvv": ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
		"vvoo": ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ],
		"vvov": ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act],
		"vvvv": ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
	}
	if(n_orb > n_act):
		dic["oooV"] = ten[0:n_occ,0:n_occ,0:n_occ,2*n_act:2*n_orb]
		dic["oovV"] = ten[0:n_occ,0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb]
		dic["ooVV"] = ten[0:n_occ,0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb]

		dic["ovoV"] = ten[0:n_occ,n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb]
		dic["ovvV"] = ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb]
		dic["ovVV"] = ten[0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb]

		dic["oVoo"] = ten[0:n_occ,2*n_act:2*n_orb,0:n_occ,0:n_occ]
		dic["oVov"] = ten[0:n_occ,2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act]
		dic["oVoV"] = ten[0:n_occ,2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb]
		dic["oVvv"] = ten[0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act]
		dic["oVvV"] = ten[0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb]
		dic["oVVV"] = ten[0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb]

		dic["vvoV"] = ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb]
		dic["vvvV"] = ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb]
		dic["vvVV"] = ten[n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb]

		dic["vVoo"] = ten[n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ,0:n_occ]
		dic["vVov"] = ten[n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act]
		dic["vVoV"] = ten[n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb]
		dic["vVvv"] = ten[n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act]
		dic["vVvV"] = ten[n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb]
		dic["vVVV"] = ten[n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb]

		dic["VVoo"] = ten[2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ,0:n_occ]
		dic["VVov"] = ten[2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act]
		dic["VVoV"] = ten[2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb]
		dic["VVvv"] = ten[2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act]
		dic["VVvV"] = ten[2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb]
		dic["VVVV"] = ten[2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb]
	return dic 

def two_body_dic2ten(dic,n_occ,n_act,n_orb):
	if(n_orb > n_act):
		ten = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
		for key in dic.keys():
			if key == "oooo":
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ] = dic["oooo"]
			elif key == "ooov":
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] = dic["ooov"]
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijka->ijak",dic["ooov"],optimize="optimal")
			elif key == "oovv":
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = dic["oovv"]
			elif key == "ovoo":
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] = dic["ovoo"]
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("iajk->aijk",dic["ovoo"],optimize="optimal")
			elif key == "ovov":
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  dic["ovov"]
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("iajb->iabj",dic["ovov"],optimize="optimal")
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("iajb->aijb",dic["ovov"],optimize="optimal")
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("iajb->aibj",dic["ovov"],optimize="optimal")
			elif key == "ovvv":
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["ovvv"]
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iabc->aibc",dic["ovvv"],optimize="optimal") 
			elif key == "vvoo":
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  dic["vvoo"]
			elif key == "vvov":
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  dic["vvov"]
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("abic->abci",dic["vvov"],optimize="optimal")
			elif key == "vvvv":
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["vvvv"]	
			elif key == "oooV":
				ten[0:n_occ,0:n_occ,0:n_occ,2*n_act:2*n_orb] =  dic["oooV"]
				ten[0:n_occ,0:n_occ,2*n_act:2*n_orb,0:n_occ] = -np.einsum("ijkA->ijAk",dic["oooV"],optimize="optimal")
			elif key == "oovV":
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb] =  dic["oovV"]
				ten[0:n_occ,0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act] = -np.einsum("ijaA->ijAa",dic["oovV"],optimize="optimal")
			elif key == "ooVV":
				ten[0:n_occ,0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb] =  dic["ooVV"]
			elif key == "ovoV":
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb] =  dic["ovoV"] 
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,2*n_act:2*n_orb] = -np.einsum("iajA->aijA",dic["ovoV"],optimize="optimal")
				ten[0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ] = -np.einsum("iajA->iaAj",dic["ovoV"],optimize="optimal")
				ten[n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb,0:n_occ] =  np.einsum("iajA->aiAj",dic["ovoV"],optimize="optimal")
			elif key == "ovvV":
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb] =  dic["ovvV"]
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb] = -np.einsum("iabA->aibA",dic["ovvV"],optimize="optimal")
				ten[0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act] = -np.einsum("iabA->iaAb",dic["ovvV"],optimize="optimal")
				ten[n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act] =  np.einsum("iabA->aiAb",dic["ovvV"],optimize="optimal")
			elif key == "ovVV":
				ten[0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb] =  dic["ovVV"]
				ten[n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb] = -np.einsum("iaAB->aiAB",dic["ovVV"],optimize="optimal")
			elif key == "oVoo":
				ten[0:n_occ,2*n_act:2*n_orb,0:n_occ,0:n_occ] =  dic["oVoo"]
				ten[2*n_act:2*n_orb,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("iAjk->Aijk",dic["oVoo"],optimize="optimal")
			elif key == "oVov":
				ten[0:n_occ,2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act] =  dic["oVov"]
				ten[2*n_act:2*n_orb,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("iAja->Aija",dic["oVov"],optimize="optimal")
				ten[0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act,0:n_occ] = -np.einsum("iAja->iAaj",dic["oVov"],optimize="optimal") 
				ten[2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("iAja->Aiaj",dic["oVov"],optimize="optimal") 
			elif key == "oVoV":
				ten[0:n_occ,2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb] =  dic["oVoV"]
				ten[2*n_act:2*n_orb,0:n_occ,0:n_occ,2*n_act:2*n_orb] = -np.einsum("iAjB->AijB",dic["oVoV"],optimize="optimal")
				ten[0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ] = -np.einsum("iAjB->iABj",dic["oVoV"],optimize="optimal")
				ten[2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb,0:n_occ] =  np.einsum("iAjB->AiBj",dic["oVoV"],optimize="optimal")
			elif key == "oVvv":
				ten[0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act] =  dic["oVvv"]
				ten[2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iAab->Aiab",dic["oVvv"],optimize="optimal")
			elif key == "oVvV":
				ten[0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb] =  dic["oVvV"]
				ten[2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb] = -np.einsum("iAaB->AiaB",dic["oVvV"],optimize="optimal") 
				ten[0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act] = -np.einsum("iAaB->iABa",dic["oVvV"],optimize="optimal") 
				ten[2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act] =  np.einsum("iAaB->AiBa",dic["oVvV"],optimize="optimal")
			elif key == "oVVV":
				ten[0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb] =  dic["oVVV"]
				ten[2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb] = -np.einsum("iABC->AiBC",dic["oVVV"],optimize="optimal")
			elif key == "vvoV":
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb] =  dic["vvoV"]
				ten[n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ] = -np.einsum("abiA->abAi",dic["vvoV"],optimize="optimal")
			elif key == "vvvV":
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb] =  dic["vvvV"]
				ten[n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act] = -np.einsum("abcA->abAc",dic["vvvV"],optimize="optimal")
			elif key == "vvVV":
				ten[n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb] =  dic["vvVV"] 
			elif key == "vVoo":
				ten[n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ,0:n_occ] =  dic["vVoo"]
				ten[2*n_act:2*n_orb,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("aAij->Aaij",dic["vVoo"],optimize="optimal") 
			elif key == "vVov":
				ten[n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act] =  dic["vVov"]
				ten[2*n_act:2*n_orb,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("aAib->Aaib",dic["vVov"],optimize="optimal")
				ten[n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act,0:n_occ] = -np.einsum("aAib->aAbi",dic["vVov"],optimize="optimal")
				ten[2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("aAib->Aabi",dic["vVov"],optimize="optimal")
			elif key == "vVoV":
				ten[n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb] =  dic["vVoV"]
				ten[2*n_act:2*n_orb,n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb] = -np.einsum("aAiB->AaiB",dic["vVoV"],optimize="optimal")
				ten[n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ] = -np.einsum("aAiB->aABi",dic["vVoV"],optimize="optimal")
				ten[2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ] =  np.einsum("aAiB->AaBi",dic["vVoV"],optimize="optimal")
			elif key == "vVvv":
				ten[n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act] =  dic["vVvv"] 
				ten[2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("aAbc->Aabc",dic["vVvv"],optimize="optimal") 
			elif key == "vVvV":
				ten[n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb] =  dic["vVvV"]
				ten[2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb] = -np.einsum("aAbB->AabB",dic["vVvV"],optimize="optimal")
				ten[n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act] = -np.einsum("aAbB->aABb",dic["vVvV"],optimize="optimal") 
				ten[2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act] =  np.einsum("aAbB->AaBb",dic["vVvV"],optimize="optimal") 
			elif key == "vVVV":
				ten[n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb] =  dic["vVVV"]
				ten[2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb] = -np.einsum("aABC->AaBC",dic["vVVV"],optimize="optimal") 
			elif key == "VVoo":
				ten[2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ,0:n_occ] =  dic["VVoo"]
			elif key == "VVov":
				ten[2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act] =  dic["VVov"] 
				ten[2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act,0:n_occ] = -np.einsum("ABia->ABai",dic["VVov"],optimize="optimal") 
			elif key == "VVoV":
				ten[2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb] =  dic["VVoV"] 
				ten[2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ] = -np.einsum("ABiC->ABCi",dic["VVoV"],optimize="optimal")
			elif key == "VVvv":
				ten[2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act] =  dic["VVvv"]
			elif key == "VVvV":
				ten[2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb] =  dic["VVvV"]
				ten[2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act] = -np.einsum("ABaC->ABCa",dic["VVvV"],optimize="optimal")
			elif key == "VVVV":
				ten[2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb] = dic["VVVV"]
	else:
		ten = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act))
		for key in dic.keys():
			if key == "oooo":
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ] = dic["oooo"]
			elif key == "ooov":
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] = dic["ooov"]
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijka->ijak",dic["ooov"],optimize="optimal")
			elif key == "oovv":
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = dic["oovv"]
			elif key == "ovoo":
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] = dic["ovoo"]
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("iajk->aijk",dic["ovoo"],optimize="optimal")
			elif key == "ovov":
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  dic["ovov"]
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("iajb->iabj",dic["ovov"],optimize="optimal")
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("iajb->aijb",dic["ovov"],optimize="optimal")
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("iajb->aibj",dic["ovov"],optimize="optimal")
			elif key == "ovvv":
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["ovvv"]
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iabc->aibc",dic["ovvv"],optimize="optimal") 
			elif key == "vvoo":
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  dic["vvoo"]
			elif key == "vvov":
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  dic["vvov"]
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("abic->abci",dic["vvov"],optimize="optimal")
			elif key == "vvvv":
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["vvvv"]	
	return ten 

def three_body_ten2dic(ten,n_occ,n_act):
	dic = {
		"oooooo": ten[0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ],
		"ooooov": ten[0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act],
		"oooovv": ten[0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act], 
		"ooovvv": ten[0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act], 
		"oovooo": ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ], 
		"oovoov": ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act], 
		"oovovv": ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act],
		"oovvvv": ten[0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act], 
		"ovvooo": ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ], 
		"ovvoov": ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act],
		"ovvovv": ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act],
		"ovvvvv": ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act],
		"vvvooo": ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ],
		"vvvoov": ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act],
		"vvvovv": ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act],
		"vvvvvv": ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act]
	}
	return dic

def three_body_dic2ten(dic,n_occ,n_act):
	ten = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))
	for key in dic.keys():
		if key == "oooooo":
			ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ] =  dic["oooooo"]
		elif key == "ooooov":
			ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] =  dic["ooooov"]
			ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijklma->ijklam",dic["ooooov"],optimize="optimal")
			ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijklma->ijkalm",dic["ooooov"],optimize="optimal")
		elif key == "oooovv":
			ten[0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] =  dic["oooovv"]
			ten[0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] = -np.einsum("ijklab->ijkalb",dic["oooovv"],optimize="optimal")
			ten[0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] =  np.einsum("ijklab->ijkabl",dic["oooovv"],optimize="optimal")
		elif key == "ooovvv":
			ten[0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] =  dic["ooovvv"]
		elif key == "oovooo":
			ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ] =  dic["oovooo"]
			ten[0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ] = -np.einsum("ijaklm->iajklm",dic["oovooo"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ] =  np.einsum("ijaklm->aijklm",dic["oovooo"],optimize="optimal")
		elif key == "oovoov":
			ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act] =  dic["oovoov"]
			ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ] = -np.einsum("ijaklb->ijakbl",dic["oovoov"],optimize="optimal")
			ten[0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ] =  np.einsum("ijaklb->ijabkl",dic["oovoov"],optimize="optimal")
			ten[0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act] = -np.einsum("ijaklb->iajklb",dic["oovoov"],optimize="optimal")
			ten[0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ] =  np.einsum("ijaklb->iajkbl",dic["oovoov"],optimize="optimal")
			ten[0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ] = -np.einsum("ijaklb->iajbkl",dic["oovoov"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act] =  np.einsum("ijaklb->aijklb",dic["oovoov"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ] = -np.einsum("ijaklb->aijkbl",dic["oovoov"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ] =  np.einsum("ijaklb->aijbkl",dic["oovoov"],optimize="optimal")
		elif key == "oovovv":
			ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] =  dic["oovovv"]
			ten[0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] = -np.einsum("ijakbc->ijabkc",dic["oovovv"],optimize="optimal")
			ten[0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] =  np.einsum("ijakbc->ijabck",dic["oovovv"],optimize="optimal")
			ten[0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] = -np.einsum("ijakbc->iajkbc",dic["oovovv"],optimize="optimal")
			ten[0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] =  np.einsum("ijakbc->iajbkc",dic["oovovv"],optimize="optimal")
			ten[0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] = -np.einsum("ijakbc->iajbck",dic["oovovv"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] =  np.einsum("ijakbc->aijkbc",dic["oovovv"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] = -np.einsum("ijakbc->aijbkc",dic["oovovv"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] =  np.einsum("ijakbc->aijbck",dic["oovovv"],optimize="optimal")
		elif key == "oovvvv":
			ten[0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] =  dic["oovvvv"]
			ten[0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] = -np.einsum("ijabcd->iajbcd",dic["oovvvv"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] =  np.einsum("ijabcd->aijbcd",dic["oovvvv"],optimize="optimal")
		elif key == "ovvooo":
			ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ] =  dic["ovvooo"]
			ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ] = -np.einsum("iabjkl->aibjkl",dic["ovvooo"],optimize="optimal")
			ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ] =  np.einsum("iabjkl->abijkl",dic["ovvooo"],optimize="optimal")
		elif key == "ovvoov":
			ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act] =  dic["ovvoov"]
			ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ] = -np.einsum("iabjkc->iabjck",dic["ovvoov"],optimize="optimal")
			ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ] =  np.einsum("iabjkc->iabcjk",dic["ovvoov"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act] = -np.einsum("iabjkc->aibjkc",dic["ovvoov"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ] =  np.einsum("iabjkc->aibjck",dic["ovvoov"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ] = -np.einsum("iabjkc->aibcjk",dic["ovvoov"],optimize="optimal")
			ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act] =  np.einsum("iabjkc->abijkc",dic["ovvoov"],optimize="optimal")
			ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ] = -np.einsum("iabjkc->abijck",dic["ovvoov"],optimize="optimal")
			ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ] =  np.einsum("iabjkc->abicjk",dic["ovvoov"],optimize="optimal")
		elif key == "ovvovv":
			ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] =  dic["ovvovv"]
			ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] = -np.einsum("iabjcd->iabcjd",dic["ovvovv"],optimize="optimal")
			ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] =  np.einsum("iabjcd->iabcdj",dic["ovvovv"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] = -np.einsum("iabjcd->aibjcd",dic["ovvovv"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] =  np.einsum("iabjcd->aibcjd",dic["ovvovv"],optimize="optimal")
			ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] = -np.einsum("iabjcd->aibcdj",dic["ovvovv"],optimize="optimal")
			ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] =  np.einsum("iabjcd->abijcd",dic["ovvovv"],optimize="optimal")
			ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] = -np.einsum("iabjcd->abicjd",dic["ovvovv"],optimize="optimal")
			ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] =  np.einsum("iabjcd->abicdj",dic["ovvovv"],optimize="optimal")
		elif key == "ovvvvv":
			ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] =  dic["ovvvvv"]
			ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] = -np.einsum("iabcde->aibcde",dic["ovvvvv"],optimize="optimal")
			ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] =  np.einsum("iabcde->abicde",dic["ovvvvv"],optimize="optimal")
		elif key == "vvvooo":
			ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ] =  dic["vvvooo"]
		elif key == "vvvoov":
			ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act] =  dic["vvvoov"]
			ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ] = -np.einsum("abcijd->abcidj",dic["vvvoov"],optimize="optimal")
			ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ] =  np.einsum("abcijd->abcdij",dic["vvvoov"],optimize="optimal")
		elif key == "vvvovv":
			ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] =  dic["vvvovv"]
			ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] = -np.einsum("abcide->abcdie",dic["vvvovv"],optimize="optimal")
			ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] =  np.einsum("abcide->abcdei",dic["vvvovv"],optimize="optimal")
		elif key == "vvvvvv":
			ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] =  dic["vvvvvv"]
	return ten

def t1_mat2dic(t1_amps,n_a,n_act,n_orb):
	n_virt_int_a = n_act - n_a 
	n_virt_ext_a = n_orb - n_act  
	n_occ = 2*n_a 
	n_virt_int = 2*n_virt_int_a 
	n_virt_ext = 2*n_virt_ext_a
	t1 = {
		"oV": np.zeros((n_occ,n_virt_ext)),
		"Vo": np.zeros((n_virt_ext,n_occ))
	}
	for i in range(0,n_a):
		ia = 2*i 
		ib = 2*i+1 
		for A in range(0,n_virt_ext_a):
			Aa = 2*A 
			Ab = 2*A+1 
			t1["oV"][ia,Aa] = t1_amps[0][i,A+n_virt_int_a]
			t1["Vo"][Aa,ia] = t1_amps[0][i,A+n_virt_int_a]
			t1["oV"][ib,Ab] = t1_amps[1][i,A+n_virt_int_a]
			t1["Vo"][Ab,ib] = t1_amps[1][i,A+n_virt_int_a] 
	return t1  

def t2_ten2dic(t2_amps,n_a,n_act,n_orb):
	n_virt_int_a = n_act - n_a 
	n_virt_ext_a = n_orb - n_act  
	n_occ = 2*n_a 
	n_virt_int = 2*n_virt_int_a 
	n_virt_ext = 2*n_virt_ext_a		

	t2 = {
		"oovV": np.zeros((n_occ,n_occ,n_virt_int,n_virt_ext)),
		"vVoo": np.zeros((n_virt_int,n_virt_ext,n_occ,n_occ)),
		"ooVV": np.zeros((n_occ,n_occ,n_virt_ext,n_virt_ext)),
		"VVoo": np.zeros((n_virt_ext,n_virt_ext,n_occ,n_occ))
	}

	# t_{ia,ja}^{aa,Ba}/t_{ib,jb}^{ab,Bb}
	for i in range(0,n_a):
		ia = 2*i
		ib = 2*i+1 
		for j in range(i+1,n_a):
			ja = 2*j
			jb = 2*j+1  
			for a in range(0,n_virt_int_a):
				aa = 2*a 
				ab = 2*a+1
				for B in range(0,n_virt_ext_a):
					Ba = 2*B
					Bb = 2*B+1  
					tijaB = t2_amps[0][i,j,a,B+n_virt_int_a]
					t2["oovV"][ia,ja,aa,Ba] =  tijaB
					t2["oovV"][ja,ia,aa,Ba] = -tijaB
					t2["vVoo"][aa,Ba,ja,ia] = -tijaB
					t2["vVoo"][aa,Ba,ia,ja] =  tijaB
					tijaB = t2_amps[2][i,j,a,B+n_virt_int_a]
					t2["oovV"][ib,jb,ab,Bb] =  tijaB
					t2["oovV"][jb,ib,ab,Bb] = -tijaB
					t2["vVoo"][ab,Bb,jb,ib] = -tijaB
					t2["vVoo"][ab,Bb,ib,jb] =  tijaB
	# t_{ia,ja}^{Aa,Ba}/t_{ib,jb}^{Ab,Bb}
	for i in range(0,n_a):
		ia = 2*i 
		ib = 2*i+1 
		for j in range(i+1,n_a):
			ja = 2*j
			jb = 2*j+1  
			for A in range(0,n_virt_ext_a):
				Aa = 2*A
				Ab = 2*A+1  
				for B in range(A+1,n_virt_ext_a):
					Ba = 2*B
					Bb = 2*B+1  
					tijAB = t2_amps[0][i,j,A+n_virt_int_a,B+n_virt_int_a]
					t2["ooVV"][ia,ja,Aa,Ba] =  tijAB
					t2["ooVV"][ja,ia,Aa,Ba] = -tijAB
					t2["ooVV"][ia,ja,Ba,Aa] = -tijAB
					t2["ooVV"][ja,ia,Ba,Aa] =  tijAB
					t2["VVoo"][Ba,Aa,ja,ia] =  tijAB
					t2["VVoo"][Aa,Ba,ja,ia] = -tijAB
					t2["VVoo"][Ba,Aa,ia,ja] = -tijAB
					t2["VVoo"][Aa,Ba,ia,ja] =  tijAB
					tijAB = t2_amps[2][i,j,A+n_virt_int_a,B+n_virt_int_a]
					t2["ooVV"][ib,jb,Ab,Bb] =  tijAB
					t2["ooVV"][jb,ib,Ab,Bb] = -tijAB
					t2["ooVV"][ib,jb,Bb,Ab] = -tijAB
					t2["ooVV"][jb,ib,Bb,Ab] =  tijAB
					t2["VVoo"][Bb,Ab,jb,ib] =  tijAB
					t2["VVoo"][Ab,Bb,jb,ib] = -tijAB
					t2["VVoo"][Bb,Ab,ib,jb] = -tijAB
					t2["VVoo"][Ab,Bb,ib,jb] =  tijAB
	# t_{ia,jb}^{aa,Bb}
	for i in range(0,n_a):
		ia = 2*i 
		for j in range(0,n_a):
			jb = 2*j+1 
			for a in range(0,n_virt_int_a):
				aa = 2*a 
				ab = 2*a+1 
				for B in range(0,n_virt_ext_a):
					Ba = 2*B 
					Bb = 2*B+1 
					tijaB = t2_amps[1][i,j,a,B+n_virt_int_a]
					t2["oovV"][ia,jb,aa,Bb] =  tijaB
					t2["oovV"][jb,ia,aa,Bb] = -tijaB
					t2["vVoo"][aa,Bb,ia,jb] =  tijaB
					t2["vVoo"][aa,Bb,jb,ia] = -tijaB
					tijBa = t2_amps[1][i,j,B+n_virt_int_a,a]
					t2["oovV"][ia,jb,ab,Ba] = -tijBa
					t2["oovV"][jb,ia,ab,Ba] =  tijBa
					t2["vVoo"][ab,Ba,ia,jb] = -tijBa
					t2["vVoo"][ab,Ba,jb,ia] =  tijBa
	# t_{ia,jb}^{Aa,Bb}
	for i in range(0,n_a):
		ia = 2*i 
		for j in range(0,n_a):
			jb = 2*j+1 
			for A in range(0,n_virt_ext_a):
				Aa = 2*A 
				for B in range(0,n_virt_ext_a):
					Bb = 2*B+1 
					tijAB = t2_amps[1][i,j,A+n_virt_int_a,B+n_virt_int_a]
					t2["ooVV"][ia,jb,Aa,Bb] =  tijAB
					t2["ooVV"][jb,ia,Aa,Bb] = -tijAB
					t2["ooVV"][ia,jb,Bb,Aa] = -tijAB
					t2["ooVV"][jb,ia,Bb,Aa] =  tijAB
					t2["VVoo"][Aa,Bb,ia,jb] =  tijAB
					t2["VVoo"][Aa,Bb,jb,ia] = -tijAB
					t2["VVoo"][Bb,Aa,ia,jb] = -tijAB
					t2["VVoo"][Bb,Aa,jb,ia] =  tijAB
	return t2  

def get_many_body_terms(operator):
	constant = of.FermionOperator()
	one_body = of.FermionOperator()
	two_body = of.FermionOperator()
	three_body = of.FermionOperator()
	four_body = of.FermionOperator()
	terms = operator.terms 
	for term in terms:
		if(len(term) == 0):
			constant += of.FermionOperator(term,terms.get(term))
		elif(len(term) == 2):
			one_body += of.FermionOperator(term,terms.get(term))
		elif(len(term) == 4):
			two_body += of.FermionOperator(term,terms.get(term))
		elif(len(term) == 6):
			three_body += of.FermionOperator(term,terms.get(term))
		elif(len(term) == 8):
			four_body += of.FermionOperator(term,terms.get(term))
		else:
			print("Unexpected number of terms: %d"%len(term))
	return(constant,one_body,two_body,three_body,four_body)

def as_proj(operator,act_max):
	proj_op = of.FermionOperator()
	const, one_body, two_body, three_body, four_body = get_many_body_terms(operator)
	# constant terms
	proj_op += const 
	# one-body terms
	terms1 = one_body.terms
	for term in terms1:
		if(term[0][0] < act_max):
			if(term[1][0] < act_max):
				proj_op += of.FermionOperator(term,terms1.get(term))
	# two-body terms
	terms2 = two_body.terms
	for term in terms2:
		if(term[0][0] < act_max):
			if(term[1][0] < act_max):
				if(term[2][0] < act_max):
					if (term[3][0] < act_max):
						proj_op += of.FermionOperator(term,terms2.get(term))
	# three-body terms
	terms3 = three_body.terms
	for term in terms3:
		if(term[0][0] < act_max):
			if(term[1][0] < act_max):
				if(term[2][0] < act_max):
					if (term[3][0] < act_max):
						if(term[4][0] < act_max):
							if(term[5][0] < act_max):
								proj_op += of.FermionOperator(term,terms3.get(term))

	# four-body terms
	terms4 = four_body.terms  
	for term in terms4:
		if(term[0][0] < act_max):
			if(term[1][0] < act_max):
				if(term[2][0] < act_max):
					if (term[3][0] < act_max):
						if(term[4][0] < act_max):
							if(term[5][0] < act_max):
								if(term[6][0] < act_max):
									if(term[7][0] < act_max):
										proj_op += of.FermionOperator(term,terms4.get(term))

	return proj_op

