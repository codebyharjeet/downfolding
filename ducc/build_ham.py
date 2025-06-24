import ducc
import scipy
#import vqe_methods
#import pyscf_helper

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, cc
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
    compute_three_body,
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
        v_blocks, t_blocks, n_a, n_b, n_orb, compute_three_body
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
        f_blocks, t_blocks, n_a, n_b, n_orb, compute_three_body
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


def get_ducc_ham(
    fmat,
    vmat,
    t1_amps,
    t2_amps,
    n_a,
    n_b,
    n_orb,
    n_occ,
    act_max,
    compute_three_body,
):
    one_body = np.zeros((2 * n_orb, 2 * n_orb))
    two_body = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    three_body = 0
    if compute_three_body == True:
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
        v_blocks, t_blocks, n_a, n_b, n_orb, compute_three_body
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
        f_blocks, t_blocks, n_a, n_b, n_orb, compute_three_body
    )
    three_body += 0.5 * ft2t2_mat_3
    two_body += 0.5 * ft2t2_mat_2
    one_body += 0.5 * ft2t2_mat_1
    constant += 0.5 * ft2t2

    constant_op = of.FermionOperator("", float(constant))

    one_body += fmat
    one_body_op = normal_ordered(ducc.one_body_to_op(one_body, n_occ, act_max))

    two_body_anti = ducc.antisymmetrize_residual(two_body, n_occ, n_orb)
    two_body_anti += 0.25 * vmat
    two_body_op = normal_ordered(ducc.two_body_to_op(two_body_anti, n_occ, act_max))

    a4_3_ham = constant_op + one_body_op + two_body_op

    if compute_three_body == True:
        three_body_op = normal_ordered(ducc.three_body_to_op(three_body, n_occ, act_max))
        a4_3_ham += three_body_op


    return a4_3_ham


def get_ducc_ham_adapt(
    fmat,
    vmat,
    t1_amps,
    t2_amps,
    n_a,
    n_b,
    n_orb,
    n_occ,
    act_max,
    compute_three_body,
):
    one_body = np.zeros((2 * n_orb, 2 * n_orb))
    two_body = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    three_body = np.zeros(
        (2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb)
    )
    constant = 0

    f_blocks, v_blocks = ducc.get_integral_blocks(fmat, vmat, n_a, n_b, n_orb)
    t_blocks = ducc.get_t_blocks(t1_amps, t2_amps)

    ft1_mat, ft1_const = ducc.compute_ft1(f_blocks, t_blocks, n_a, n_b, n_orb)
    one_body += ft1_mat
    constant += ft1_const

    ft2_mat_1, ft2_mat_2 = ducc.compute_ft2(f_blocks, t_blocks, n_a, n_b, n_orb)
    two_body += ft2_mat_2
    one_body += ft2_mat_1

    wt1_mat_1, wt1_mat_2 = ducc.compute_wt1(v_blocks, t_blocks, n_a, n_b, n_orb)
    two_body += wt1_mat_2
    one_body += wt1_mat_1

    wt2_mat_1, wt2_mat_2, wt2_mat_3, wt2 = ducc.compute_wt2(
        v_blocks, t_blocks, n_a, n_b, n_orb, compute_three_body
    )
    three_body += wt2_mat_3
    two_body += wt2_mat_2
    one_body += wt2_mat_1
    constant += wt2

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
        f_blocks, t_blocks, n_a, n_b, n_orb, compute_three_body
    )
    three_body += 0.5 * ft2t2_mat_3
    two_body += 0.5 * ft2t2_mat_2
    one_body += 0.5 * ft2t2_mat_1
    constant += 0.5 * ft2t2

    constant_op = of.FermionOperator("", float(constant))

    one_body += fmat
    one_body_op = normal_ordered(ducc.one_body_to_op(one_body, act_max, n_occ))

    two_body_anti = ducc.antisymmetrize_residual(two_body, n_occ, n_orb)
    two_body_anti += vmat
    two_body_op = normal_ordered(ducc.two_body_to_op(two_body_anti, act_max, n_occ))

    three_body_op = normal_ordered(ducc.three_body_to_op(three_body, act_max, n_occ))

    a4_3_ham = constant_op + one_body_op + two_body_op + three_body_op

    return a4_3_ham


def get_ducc_a2(
    fmat,
    vmat,
    t1_amps,
    t2_amps,
    n_a,
    n_b,
    n_orb,
    n_occ,
    act_max,
    compute_three_body,
):
    one_body = np.zeros((2 * n_orb, 2 * n_orb))
    two_body = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    three_body = np.zeros(
        (2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb)
    )
    constant = 0

    f_blocks, v_blocks = ducc.get_integral_blocks(fmat, vmat, n_a, n_b, n_orb)
    t_blocks = ducc.get_t_blocks(t1_amps, t2_amps)

    ft1_mat, ft1_const = ducc.compute_ft1(f_blocks, t_blocks, n_a, n_b, n_orb)
    one_body += ft1_mat
    constant += ft1_const

    ft2_mat_1, ft2_mat_2 = ducc.compute_ft2(f_blocks, t_blocks, n_a, n_b, n_orb)
    two_body += ft2_mat_2
    one_body += ft2_mat_1

    wt1_mat_1, wt1_mat_2 = ducc.compute_wt1(v_blocks, t_blocks, n_a, n_b, n_orb)
    two_body +=  wt1_mat_2
    one_body +=  wt1_mat_1

    wt2_mat_1, wt2_mat_2, wt2_mat_3, wt2 = ducc.compute_wt2(
        v_blocks, t_blocks, n_a, n_b, n_orb, compute_three_body
    )
    three_body += wt2_mat_3
    two_body += wt2_mat_2
    one_body += wt2_mat_1
    constant += wt2



    one_body += fmat

    two_body_anti = ducc.antisymmetrize_residual(two_body, n_occ, n_orb)
    two_body_anti += vmat
   
    #two_body += vmat

    return constant, one_body, two_body_anti


def get_ducc_integrals(
    fmat,
    vmat,
    t1_amps,
    t2_amps,
    n_a,
    n_b,
    n_orb,
    n_occ,
    act_max,
    compute_three_body,
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
        v_blocks, t_blocks, n_a, n_b, n_orb, compute_three_body
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
        f_blocks, t_blocks, n_a, n_b, n_orb, compute_three_body
    )
    three_body += 0.5 * ft2t2_mat_3
    two_body += 0.5 * ft2t2_mat_2
    one_body += 0.5 * ft2t2_mat_1
    constant += 0.5 * ft2t2



    one_body += fmat
  

    two_body_anti = ducc.antisymmetrize_residual(two_body, n_occ, n_orb)
    two_body_anti += 0.25 * vmat
   


    return a4_3_ham


def get_ducc_ham_active_space(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_orb,n_occ,n_act,
    inc_3_body,):
    constant = 0
    one_body = np.zeros((2 * n_act, 2 * n_act))
    two_body = np.zeros((2 * n_act, 2 * n_act, 2 * n_act, 2 * n_act))
    three_body = 0
    if(inc_3_body):
        three_body = np.zeros((2 * n_act, 2 * n_act, 2 * n_act, 2 * n_act, 2 * n_act, 2 * n_act))
    
    fdic = ducc.one_body_mat2dic(fmat,n_occ,n_act,n_orb)
    vdic = ducc.two_body_ten2dic(vten,n_occ,n_act,n_orb)
    t1dic = ducc.t1_mat2dic(t1_amps,n_a,n_act,n_orb)
    t2dic = ducc.t2_ten2dic(t2_amps,n_a,n_act,n_orb)

    fn_s1_dic = ducc.fn_s1(fdic,t1dic)
    one_body += ducc.one_body_dic2mat(fn_s1_dic,n_occ,n_act,n_act)
    constant += fn_s1_dic['c']

    fn_s2_dic = ducc.fn_s2(fdic,t2dic)
    one_body += ducc.one_body_dic2mat(fn_s2_dic,n_occ,n_act,n_act)
    two_body += ducc.two_body_dic2ten(fn_s2_dic,n_occ,n_act,n_act)

    wn_s1_dic = ducc.wn_s1(vdic,t1dic)
    one_body += 0.250 * ducc.one_body_dic2mat(wn_s1_dic,n_occ,n_act,n_act)
    two_body += 0.250 * ducc.two_body_dic2ten(wn_s1_dic,n_occ,n_act,n_act)

    wn_s2_dic = ducc.wn_s2(vdic,t2dic,inc_3_body)
    constant += 0.250 * wn_s2_dic["c"]
    one_body += 0.250 * ducc.one_body_dic2mat(wn_s2_dic,n_occ,n_act,n_act)
    two_body += 0.250 * ducc.two_body_dic2ten(wn_s2_dic,n_occ,n_act,n_act)
    if(inc_3_body):
        three_body += 0.250 * ducc.three_body_dic2ten(wn_s2_dic,n_occ,n_act)

    fn_s1_s1_dic = ducc.fn_s1_s1(fdic,t1dic)
    constant += 0.500 * fn_s1_s1_dic['c']
    one_body += 0.500 * ducc.one_body_dic2mat(fn_s1_s1_dic,n_occ,n_act,n_act)

    fn_s1_s2_dic = ducc.fn_s1_s2(fdic,t1dic,t2dic)
    one_body += 0.500 *ducc.one_body_dic2mat(fn_s1_s2_dic,n_occ,n_act,n_act)
    two_body += 0.500 *ducc.two_body_dic2ten(fn_s1_s2_dic,n_occ,n_act,n_act)

    fn_s2_s1_dic = ducc.fn_s2_s1(fdic,t1dic,t2dic)
    constant += 0.500 * fn_s2_s1_dic['c']
    one_body += 0.500 * ducc.one_body_dic2mat(fn_s2_s1_dic,n_occ,n_act,n_act)
    two_body += 0.500 * ducc.two_body_dic2ten(fn_s2_s1_dic,n_occ,n_act,n_act)

    fn_s2_s2_dic = ducc.fn_s2_s2(fdic,t2dic,inc_3_body)
    constant += 0.500 * fn_s2_s2_dic['c']
    one_body += 0.500 * ducc.one_body_dic2mat(fn_s2_s2_dic,n_occ,n_act,n_act)
    two_body += 0.500 * ducc.two_body_dic2ten(fn_s2_s2_dic,n_occ,n_act,n_act)
    if(inc_3_body):
        three_body += 0.500 * ducc.three_body_dic2ten(fn_s2_s2_dic,n_occ,n_act)

    constant_op = of.FermionOperator("", float(constant))

    one_body += fmat[0:2*n_act,0:2*n_act]
    one_body_op = normal_ordered(ducc.one_body_to_op(one_body, n_occ, n_act))

    two_body += 0.250 * vten[0:2*n_act,0:2*n_act,0:2*n_act,0:2*n_act]  
    two_body_op = normal_ordered(ducc.two_body_to_op(two_body, n_occ, n_act))
    
    ducc_ham = constant_op + one_body_op + two_body_op

    if(inc_3_body):
        three_body_op = normal_ordered(ducc.three_body_to_op(three_body, n_occ, n_act))
        ducc_ham += three_body_op

    return ducc_ham



def get_ducc_ham_active_space_pyscf(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_orb,n_occ,n_act,
    inc_3_body,):
    constant = 0
    one_body = np.zeros((2 * n_act, 2 * n_act))
    two_body = np.zeros((2 * n_act, 2 * n_act, 2 * n_act, 2 * n_act))
    three_body = 0
    if(inc_3_body):
        three_body = np.zeros((2 * n_act, 2 * n_act, 2 * n_act, 2 * n_act, 2 * n_act, 2 * n_act))
    
    fdic = ducc.one_body_mat2dic(fmat,n_occ,n_act,n_orb)
    vdic = ducc.two_body_ten2dic(vten,n_occ,n_act,n_orb)
    t1dic = ducc.t1_mat2dic(t1_amps,n_a,n_act,n_orb)
    t2dic = ducc.t2_ten2dic(t2_amps,n_a,n_act,n_orb)

    fn_s1_dic = ducc.fn_s1(fdic,t1dic)
    one_body += ducc.one_body_dic2mat(fn_s1_dic,n_occ,n_act,n_act)
    constant += fn_s1_dic['c']

    fn_s2_dic = ducc.fn_s2(fdic,t2dic)
    one_body += ducc.one_body_dic2mat(fn_s2_dic,n_occ,n_act,n_act)
    two_body += ducc.two_body_dic2ten(fn_s2_dic,n_occ,n_act,n_act)

    wn_s1_dic = ducc.wn_s1(vdic,t1dic)
    one_body += ducc.one_body_dic2mat(wn_s1_dic,n_occ,n_act,n_act)
    two_body += ducc.two_body_dic2ten(wn_s1_dic,n_occ,n_act,n_act)

    wn_s2_dic = ducc.wn_s2(vdic,t2dic,inc_3_body)
    constant += wn_s2_dic["c"]
    one_body += ducc.one_body_dic2mat(wn_s2_dic,n_occ,n_act,n_act)
    two_body += ducc.two_body_dic2ten(wn_s2_dic,n_occ,n_act,n_act)
    if(inc_3_body):
        three_body += ducc.three_body_dic2ten(wn_s2_dic,n_occ,n_act)

    fn_s1_s1_dic = ducc.fn_s1_s1(fdic,t1dic)
    constant += 0.500 * fn_s1_s1_dic['c']
    one_body += 0.500 * ducc.one_body_dic2mat(fn_s1_s1_dic,n_occ,n_act,n_act)

    fn_s1_s2_dic = ducc.fn_s1_s2(fdic,t1dic,t2dic)
    one_body += 0.500 *ducc.one_body_dic2mat(fn_s1_s2_dic,n_occ,n_act,n_act)
    two_body += 0.500 *ducc.two_body_dic2ten(fn_s1_s2_dic,n_occ,n_act,n_act)

    fn_s2_s1_dic = ducc.fn_s2_s1(fdic,t1dic,t2dic)
    constant += 0.500 * fn_s2_s1_dic['c']
    one_body += 0.500 * ducc.one_body_dic2mat(fn_s2_s1_dic,n_occ,n_act,n_act)
    two_body += 0.500 * ducc.two_body_dic2ten(fn_s2_s1_dic,n_occ,n_act,n_act)

    fn_s2_s2_dic = ducc.fn_s2_s2(fdic,t2dic,inc_3_body)
    constant += 0.500 * fn_s2_s2_dic['c']
    one_body += 0.500 * ducc.one_body_dic2mat(fn_s2_s2_dic,n_occ,n_act,n_act)
    two_body += 0.500 * ducc.two_body_dic2ten(fn_s2_s2_dic,n_occ,n_act,n_act)
    if(inc_3_body):
        three_body += 0.500 * ducc.three_body_dic2ten(fn_s2_s2_dic,n_occ,n_act)


    one_body += fmat[0:2*n_act,0:2*n_act]
    two_body += vten[0:2*n_act,0:2*n_act,0:2*n_act,0:2*n_act]  

    return constant, one_body, two_body, three_body
