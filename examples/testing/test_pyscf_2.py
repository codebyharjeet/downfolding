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


#geometry = [('Be', (0,0,0))]
bond_length = 1
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length)), ('H', (0., 0., 2*bond_length)), ('H', (0., 0., 3*bond_length))]
charge = 0
spin = 0
#basis = 'cc-pvdz'
basis = '6-31g'
[n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = ducc.init(geometry,charge,spin,basis,reference='rhf')

sq_ham = ducc.SQ_Hamiltonian()
sq_ham.init(h, g, C, S)
print(" HF Energy: %21.15f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))

n_occ = n_a+n_b
n_act = 4
n_frz = 0
act_max = n_frz+n_act
shift = 2*n_frz

mf = scf.RHF(mol)
mf.conv_tol_grad = 1e-14
mf.max_cycle = 1000
mf.verbose = 0
mf.init_guess = 'atom'
mf.kernel()
print("MO energies:")
print(mf.mo_energy)

mccsd = cc.UCCSD(mf)
mccsd.conv_tol = 1e-12
mccsd.conv_tol_normt = 1e-10
#myccsd.diis_start_cycle=10000
mccsd.max_cycle = 1000
mccsd.kernel()


t1_amps = mccsd.t1
t2_amps = mccsd.t2
#t1_amps, t2_amps = ducc.get_t_ext(t1_amps,t2_amps,n_a,n_b,n_act)


fmat = sq_ham.make_f(range(n_a),range(n_b))
fdic = ducc.one_body_mat2dic(fmat,n_occ,n_act,n_orb)
fop = ducc.one_body_to_op(fmat,n_occ,n_orb)

vten = 0.25*sq_ham.make_v()
vdic = ducc.two_body_ten2dic(vten,n_occ,n_act,n_orb)
vop = ducc.two_body_to_op(vten,n_occ,n_orb)


t1dic = ducc.t1_mat2dic(t1_amps,n_a,n_act,n_orb)
s1_op = ducc.t1_to_op(ducc.t1_to_ext(t1_amps,n_act))

t2dic = ducc.t2_ten2dic(t2_amps,n_a,n_act,n_orb)
s2_op = ducc.t2_to_op(ducc.t2_to_ext(t2_amps,n_act))


#"""
# Test [Fn,S1]
print("Test [Fn,S1]")
fn_s1_dic = ducc.fn_s1(fdic,t1dic)
fn_s1_op = normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s1_op += of.FermionOperator('',fn_s1_dic['c'])

fn_s1_op_test = ducc.as_proj(normal_ordered(commutator(fop,s1_op)),2*n_act)

print("diff")
print(fn_s1_op - fn_s1_op_test)
#"""

#"""
# Test [Fn,S2]
print("Test [Fn,S2]")
fn_s2_dic = ducc.fn_s2(fdic,t2dic)
fn_s2_op = ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s2_dic,n_occ,n_act,n_act),n_occ,n_act)
fn_s2_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(fn_s2_dic,n_occ,n_act,n_act),n_occ,n_act))

fn_s2_op_test = ducc.as_proj(normal_ordered(commutator(fop,s2_op)),2*n_act)
print("diff")
print(normal_ordered(fn_s2_op - fn_s2_op_test))
#"""

#"""
# Test [Wn,S1]
print("Test [Wn,S1]")
wn_s1_dic = ducc.wn_s1(vdic,t1dic)
wn_s1_op = normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(wn_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
wn_s1_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(wn_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
wn_s1_op_test = ducc.as_proj(normal_ordered(commutator(vop,s1_op)),2*n_act)
print("diff")
print(normal_ordered(wn_s1_op - wn_s1_op_test))
#"""

#"""
# Test [Wn,S2]
print("Test [Wn,S2]")
wn_s2_dic = ducc.wn_s2(vdic,t2dic)
wn_s2_op = of.FermionOperator('',wn_s2_dic["c"])
wn_s2_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(wn_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
wn_s2_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(wn_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
wn_s2_op += normal_ordered(ducc.three_body_to_op(ducc.three_body_dic2ten(wn_s2_dic,n_occ,n_act),n_occ,n_act))
wn_s2_op_test = ducc.as_proj(normal_ordered(commutator(vop,s2_op)),2*n_act)
print("diff")
print(normal_ordered(wn_s2_op - wn_s2_op_test))
#"""


#"""
# Test [[Fn,S1],S1]
print("Test [[Fn,S1],S1]")
fn_s1_s1_dic = ducc.fn_s1_s1(fdic,t1dic)
fn_s1_s1_op = of.FermionOperator('',fn_s1_s1_dic['c'])
fn_s1_s1_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s1_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s1_s1_op_test = ducc.as_proj(normal_ordered(commutator(commutator(fop,s1_op),s1_op)),2*n_act)
print("diff")
print(fn_s1_s1_op - fn_s1_s1_op_test)
#"""

#"""
# Test [[Fn,S1],S2]
print("Test [[Fn,S1],S2]")
fn_s1_s2_dic = ducc.fn_s1_s2(fdic,t1dic,t2dic)
fn_s1_s2_op = normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s1_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s1_s2_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(fn_s1_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s1_s2_op_test = ducc.as_proj(normal_ordered(commutator(commutator(fop,s1_op),s2_op)),2*n_act)
print("diff")
print(fn_s1_s2_op - fn_s1_s2_op_test)
#"""

#"""
# Test [[Fn,S2],S1]
print("Test [[Fn,S2],S1]")
fn_s2_s1_dic = ducc.fn_s2_s1(fdic,t1dic,t2dic)
fn_s2_s1_op = of.FermionOperator('',fn_s2_s1_dic['c'])
fn_s2_s1_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s2_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s2_s1_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(fn_s2_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s2_s1_op_test = ducc.as_proj(normal_ordered(commutator(commutator(fop,s2_op),s1_op)),2*n_act)
print("diff")
print(fn_s2_s1_op - fn_s2_s1_op_test)
#"""

#"""
# Test [[Fn,S2],S2]
print("Test [[Fn,S2],S2]")
fn_s2_s2_dic = ducc.fn_s2_s2(fdic,t2dic)
fn_s2_s2_op = of.FermionOperator('',fn_s2_s2_dic['c'])
fn_s2_s2_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s2_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s2_s2_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(fn_s2_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s2_s2_op += normal_ordered(ducc.three_body_to_op(ducc.three_body_dic2ten(fn_s2_s2_dic,n_occ,n_act),n_occ,n_act))
fn_s2_s2_op_test = ducc.as_proj(normal_ordered(commutator(commutator(fop,s2_op),s2_op)),2*n_act)
print("diff")
print(fn_s2_s2_op - fn_s2_s2_op_test)
#"""
