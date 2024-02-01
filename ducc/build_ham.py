import ducc
import scipy
import vqe_methods
import pyscf_helper

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, molden, cc
from pyscf.cc import ccsd

import openfermion as of
from openfermion import *
from tVQE import *

import numpy as np


def compute_ducc_of(fmat, vmat, t1_amps, t2_amps, n_a, n_b, n_orb,n_occ, act_max, compute_three_body=False):
    
    one_body = np.zeros((2*n_orb,2*n_orb))
    two_body = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
    three_body = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb))
    constant = 0

    f_blocks, v_blocks = ducc.get_integral_blocks(fmat,0.25*vmat,n_a,n_b,n_orb)
    t_blocks = ducc.get_t_blocks(t1_amps, t2_amps)

    ft1_mat, ft1_const = ducc.compute_ft1(f_blocks,t_blocks,n_a,n_b,n_orb)
    one_body += ft1_mat
    constant += ft1_const

    ft2_mat_1,ft2_mat_2 = ducc.compute_ft2(f_blocks,t_blocks,n_a,n_b,n_orb)
    two_body += ft2_mat_2
    one_body += ft2_mat_1

    wt1_mat_1, wt1_mat_2 = ducc.compute_wt1(v_blocks,t_blocks,n_a,n_b,n_orb)
    two_body += 4*wt1_mat_2
    one_body += 4*wt1_mat_1

    wt2_mat_1, wt2_mat_2, wt2_mat_3, wt2 = ducc.compute_wt2(v_blocks,t_blocks,n_a,n_b,n_orb,compute_three_body=False)
    two_body += 4*wt2_mat_2
    one_body += 4*wt2_mat_1
    constant += 4*wt2


    constant_op = of.FermionOperator('', float(constant))
    
    one_body += fmat 
    one_body_op = normal_ordered(ducc.one_body_to_op(one_body,act_max,n_occ))

    two_body_anti = ducc.antisymmetrize_residual(two_body,n_occ,n_orb)
    two_body_anti += 0.25*vmat
    two_body_op = normal_ordered(ducc.two_body_to_op(two_body_anti,act_max,n_occ))

    a3_ham = constant_op + one_body_op + two_body_op

    return eigenspectrum(a3_ham)[0].real