"""
Unit and regression test for the ducc package.
"""

# Import package, test suite, and other packages as needed
import sys
# import pytest
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



# @pytest.fixture(scope="module")
def beryllium_atom():
    geometry = [('Be', (0,0,0))]
    charge = 0
    spin = 0
    basis = 'cc-pvdz'
    [n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = ducc.init(geometry,charge,spin,basis,reference='rhf')
    sq_ham = ducc.SQ_Hamiltonian()
    sq_ham.init(h, g, C, S)

    n_occ = n_a+n_b
    n_act = 5
    n_frz = 0
    act_max = n_frz+n_act
    shift = 2*n_frz

    mf = scf.RHF(mol)
    mf.conv_tol_grad = 1e-14
    mf.max_cycle = 1000
    mf.verbose = 0
    mf.init_guess = 'atom'
    mf.kernel()

    mccsd = cc.UCCSD(mf)
    mccsd.conv_tol = 1e-12
    mccsd.conv_tol_normt = 1e-10
    #myccsd.diis_start_cycle=10000
    mccsd.max_cycle = 1000
    mccsd.kernel()
    t1_amps = mccsd.t1
    t2_amps = mccsd.t2

    fmat = sq_ham.make_f(range(n_a),range(n_b))
    vmat = sq_ham.make_v()

    t1_amps, t2_amps = ducc.get_t_ext(t1_amps,t2_amps,n_a,n_b,act_max)
    t1_amps, t2_amps = ducc.transform_t_spatial_to_spin(t1_amps, t2_amps, n_a, n_b, n_orb)

    return geometry, charge, spin, basis, fmat, vmat, t1_amps, t2_amps, n_a, n_b, n_orb, n_occ, act_max


def test_ducc_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "ducc" in sys.modules

def test_energy_of_determinant(beryllium_atom):
    geometry, charge, spin, basis, fmat, vmat, t1_amps, t2_amps, n_a, n_b, n_orb, n_occ, act_max = beryllium_atom
    [n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = ducc.init(geometry,charge,spin,basis,reference='rhf')
    sq_ham = ducc.SQ_Hamiltonian()
    sq_ham.init(h, g, C, S)
    ehf = E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))
    assert(abs(ehf - -14.57233763) < 1e-7)


def test_a3_ham(beryllium_atom):
    geometry, charge, spin, basis, fmat, vmat, t1_amps, t2_amps, n_a, n_b, n_orb, n_occ, act_max = beryllium_atom

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

    a3_expected = -0.06793561

    # assert pytest.approx(a3_expected, abs=1e-7) == eigenspectrum(a3_ham)[0].real


def test_a3_3_ham(beryllium_atom):
    geometry, charge, spin, basis, fmat, vmat, t1_amps, t2_amps, n_a, n_b, n_orb, n_occ, act_max = beryllium_atom

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

    wt2_mat_1, wt2_mat_2, wt2_mat_3, wt2 = ducc.compute_wt2(v_blocks,t_blocks,n_a,n_b,n_orb,compute_three_body=True)
    three_body += 4*wt2_mat_3
    two_body += 4*wt2_mat_2
    one_body += 4*wt2_mat_1
    constant += 4*wt2

    constant_op = of.FermionOperator('', float(constant))
    
    one_body += fmat 
    one_body_op = normal_ordered(ducc.one_body_to_op(one_body,act_max,n_occ))

    two_body_anti = ducc.antisymmetrize_residual(two_body,n_occ,n_orb)
    two_body_anti += 0.25*vmat
    two_body_op = normal_ordered(ducc.two_body_to_op(two_body_anti,act_max,n_occ))

    three_body_op = normal_ordered(ducc.three_body_to_op(three_body,act_max,n_occ))    

    a3_3_ham = constant_op + one_body_op + two_body_op + three_body_op

    a3_3_expected = -0.06742379

    # assert pytest.approx(a3_3_expected, abs=1e-7) == eigenspectrum(a3_3_ham)[0].real


def test_a4_ham(beryllium_atom):
    geometry, charge, spin, basis, fmat, vmat, t1_amps, t2_amps, n_a, n_b, n_orb, n_occ, act_max = beryllium_atom

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

    ft1t1_mat_1, ft1t1 = ducc.compute_ft1t1(f_blocks,t_blocks,n_a,n_b,n_orb)
    one_body += 0.5*ft1t1_mat_1
    constant += 0.5*ft1t1

    ft2t1_mat_1, ft2t1_mat_2, ft2t1 = ducc.compute_ft2t1(f_blocks,t_blocks,n_a,n_b,n_orb)
    two_body += 0.5*ft2t1_mat_2
    one_body += 0.5*ft2t1_mat_1
    constant += 0.5*ft2t1

    ft1t2_mat_1, ft1t2_mat_2 = ducc.compute_ft1t2(f_blocks,t_blocks,n_a,n_b,n_orb)
    two_body += 0.5*ft1t2_mat_2
    one_body += 0.5*ft1t2_mat_1

    ft2t2_mat_1, ft2t2_mat_2, ft2t2_mat_3, ft2t2 = ducc.compute_ft2t2(f_blocks,t_blocks,n_a,n_b,n_orb,compute_three_body=False)
    two_body += 0.5*ft2t2_mat_2
    one_body += 0.5*ft2t2_mat_1
    constant += 0.5*ft2t2

    constant_op = of.FermionOperator('', float(constant))
    
    one_body += fmat 
    one_body_op = normal_ordered(ducc.one_body_to_op(one_body,act_max,n_occ))

    two_body_anti = ducc.antisymmetrize_residual(two_body,n_occ,n_orb)
    two_body_anti += 0.25*vmat
    two_body_op = normal_ordered(ducc.two_body_to_op(two_body_anti,act_max,n_occ))

    a4_ham = constant_op + one_body_op + two_body_op

    a4_expected = -0.03156022

    # assert pytest.approx(a4_expected, abs=1e-7) == eigenspectrum(a4_ham)[0].real



def test_a4_3_ham(beryllium_atom):
    geometry, charge, spin, basis, fmat, vmat, t1_amps, t2_amps, n_a, n_b, n_orb, n_occ, act_max = beryllium_atom

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

    wt2_mat_1, wt2_mat_2, wt2_mat_3, wt2 = ducc.compute_wt2(v_blocks,t_blocks,n_a,n_b,n_orb,compute_three_body=True)
    three_body += 4*wt2_mat_3
    two_body += 4*wt2_mat_2
    one_body += 4*wt2_mat_1
    constant += 4*wt2

    ft1t1_mat_1, ft1t1 = ducc.compute_ft1t1(f_blocks,t_blocks,n_a,n_b,n_orb)
    one_body += 0.5*ft1t1_mat_1
    constant += 0.5*ft1t1

    ft2t1_mat_1, ft2t1_mat_2, ft2t1 = ducc.compute_ft2t1(f_blocks,t_blocks,n_a,n_b,n_orb)
    two_body += 0.5*ft2t1_mat_2
    one_body += 0.5*ft2t1_mat_1
    constant += 0.5*ft2t1

    ft1t2_mat_1, ft1t2_mat_2 = ducc.compute_ft1t2(f_blocks,t_blocks,n_a,n_b,n_orb)
    two_body += 0.5*ft1t2_mat_2
    one_body += 0.5*ft1t2_mat_1

    ft2t2_mat_1, ft2t2_mat_2, ft2t2_mat_3, ft2t2 = ducc.compute_ft2t2(f_blocks,t_blocks,n_a,n_b,n_orb,compute_three_body=True)
    three_body += 0.5*ft2t2_mat_3
    two_body += 0.5*ft2t2_mat_2
    one_body += 0.5*ft2t2_mat_1
    constant += 0.5*ft2t2

    constant_op = of.FermionOperator('', float(constant))
    
    one_body += fmat 
    one_body_op = normal_ordered(ducc.one_body_to_op(one_body,act_max,n_occ))

    two_body_anti = ducc.antisymmetrize_residual(two_body,n_occ,n_orb)
    two_body_anti += 0.25*vmat
    two_body_op = normal_ordered(ducc.two_body_to_op(two_body_anti,act_max,n_occ))

    three_body_op = normal_ordered(ducc.three_body_to_op(three_body,act_max,n_occ))

    a4_3_ham = constant_op + one_body_op + two_body_op + three_body_op

    a4_3_expected = -0.03162960

    # assert pytest.approx(a4_3_expected, abs=1e-7) == eigenspectrum(a4_3_ham)[0].real