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
import copy as cp


def compute_ducc_of(
    fmat,
    vmat,
    t1_amps,
    t2_amps,
    n_a,
    n_b,
    n_orb,
    n_occ,
    act_max,
    compute_three_body=False,
):
    one_body = np.zeros((2 * n_orb, 2 * n_orb))
    two_body = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    three_body = np.zeros(
        (2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb)
    )
    constant = 0

    f_blocks, v_blocks = ducc.get_integral_blocks(fmat, 0.25 * vmat, n_a, n_b, n_orb)
    t_blocks = ducc.get_t_blocks(t1_amps, t2_amps)

    ft1_mat, ft1_const = ducc.compute_ft1(f_blocks, t_blocks, n_a, n_b, n_orb)
    one_body += ft1_mat
    constant += ft1_const

    ft2_mat_1, ft2_mat_2 = ducc.compute_ft2(f_blocks, t_blocks, n_a, n_b, n_orb)
    two_body += ft2_mat_2
    one_body += ft2_mat_1

    wt1_mat_1, wt1_mat_2 = ducc.compute_wt1(v_blocks, t_blocks, n_a, n_b, n_orb)
    two_body += 4 * wt1_mat_2
    one_body += 4 * wt1_mat_1

    wt2_mat_1, wt2_mat_2, wt2_mat_3, wt2 = ducc.compute_wt2(
        v_blocks, t_blocks, n_a, n_b, n_orb, compute_three_body=False
    )
    three_body += 4 * wt2_mat_3
    two_body += 4 * wt2_mat_2
    one_body += 4 * wt2_mat_1
    constant += 4 * wt2

    ft1t1_mat_1, ft1t1 = ducc.compute_ft1t1(f_blocks, t_blocks, n_a, n_b, n_orb)
    one_body += 0.5 * ft1t1_mat_1
    constant += 0.5 * ft1t1

    ft2t1_mat_1, ft2t1_mat_2, ft2t1 = ducc.compute_ft2t1(
        f_blocks, t_blocks, n_a, n_b, n_orb
    )
    two_body += 0.5 * ft2t1_mat_2
    one_body += 0.5 * ft2t1_mat_1
    constant += 0.5 * ft2t1

    ft1t2_mat_1, ft1t2_mat_2 = ducc.compute_ft1t2(f_blocks, t_blocks, n_a, n_b, n_orb)
    two_body += 0.5 * ft1t2_mat_2
    one_body += 0.5 * ft1t2_mat_1

    ft2t2_mat_1, ft2t2_mat_2, ft2t2_mat_3, ft2t2 = ducc.compute_ft2t2(
        f_blocks, t_blocks, n_a, n_b, n_orb, compute_three_body=False
    )
    three_body += 0.5 * ft2t2_mat_3
    two_body += 0.5 * ft2t2_mat_2
    one_body += 0.5 * ft2t2_mat_1
    constant += 0.5 * ft2t2

    constant_op = of.FermionOperator("", float(constant))

    one_body += fmat
    one_body_op = normal_ordered(ducc.one_body_to_op(one_body, act_max, n_occ))

    two_body_anti = ducc.antisymmetrize_residual(two_body, n_occ, n_orb)
    two_body_anti += 0.25 * vmat
    two_body_op = normal_ordered(ducc.two_body_to_op(two_body_anti, act_max, n_occ))

    three_body_op = normal_ordered(ducc.three_body_to_op(three_body, act_max, n_occ))

    a4_3_ham = constant_op + one_body_op + two_body_op + three_body_op

    # a4_3_expected = -0.03162960
    # a4_expected = -0.03156022

    return eigenspectrum(a4_3_ham)[0].real
