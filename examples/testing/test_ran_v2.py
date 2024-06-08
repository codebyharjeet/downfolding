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



n_a = 2
n_occ = 4
n_act = 4
n_orb = 5

fmat = ducc.rand_1_body(n_orb)
fdic = ducc.one_body_mat2dic(fmat,n_occ,n_act,n_orb)
fop = ducc.one_body_to_op(fmat,n_occ,n_orb)

vten = ducc.rand_2_body(n_orb)
vdic = ducc.two_body_ten2dic(vten,n_occ,n_act,n_orb)
vop = ducc.two_body_to_op(vten,n_occ,n_orb)

t1_amps = ducc.rand_t1_ext(n_a,n_act,n_orb)
t1dic = ducc.t1_mat2dic(t1_amps,n_a,n_act,n_orb)
s1_op = ducc.t1_to_op(t1_amps)

t2_amps = ducc.rand_t2_ext(n_a,n_act,n_orb)
t2dic = ducc.t2_ten2dic(t2_amps,n_a,n_act,n_orb)
s2_op = ducc.t2_to_op(t2_amps)


"""
# Test [Fn,S1]
print("Test [Fn,S1]")
fn_s1_dic = ducc.fn_s1(fdic,t1dic)
fn_s1_op = normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s1_op += of.FermionOperator('',fn_s1_dic['c'])

fn_s1_op_test = ducc.as_proj(normal_ordered(commutator(fop,s1_op)),2*n_act)

print("diff")
print(fn_s1_op - fn_s1_op_test)
"""

"""
# Test [Fn,S2]
print("Test [Fn,S2]")
fn_s2_dic = ducc.fn_s2(fdic,t2dic)
fn_s2_op = ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s2_dic,n_occ,n_act,n_act),n_occ,n_act)
fn_s2_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(fn_s2_dic,n_occ,n_act,n_act),n_occ,n_act))

fn_s2_op_test = ducc.as_proj(normal_ordered(commutator(fop,s2_op)),2*n_act)
print("diff")
print(normal_ordered(fn_s2_op - fn_s2_op_test))
"""

"""
# Test [Wn,S1]
print("Test [Wn,S1]")
wn_s1_dic = ducc.wn_s1(vdic,t1dic)
wn_s1_op = normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(wn_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
wn_s1_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(wn_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
wn_s1_op_test = ducc.as_proj(normal_ordered(commutator(vop,s1_op)),2*n_act)
print("diff")
print(normal_ordered(wn_s1_op - wn_s1_op_test))
"""

"""
three_body = ducc.rand_3_body(n_act)
three_body_op = normal_ordered(ducc.three_body_to_op(three_body,n_occ,n_act))
# print(three_body_op)
print("Hermitian Test")
print(is_hermitian(three_body_op))
print(normal_ordered(three_body_op - hermitian_conjugated(three_body_op)))
"""

"""
# Test [Wn,S2]
print("Test [Wn,S2]")
wn_s2_dic = ducc.wn_s2(vdic,t2dic)
wn_s2_op = of.FermionOperator('',wn_s2_dic["c"])
wn_s2_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(wn_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
wn_s2_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(wn_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
wn_s2_op += normal_ordered(ducc.three_body_to_op(ducc.three_body_dic2ten(wn_s2_dic,n_occ,n_act),n_occ,n_act))
print("Computed einsum")
wn_s2_op_test = ducc.as_proj(normal_ordered(commutator(vop,s2_op)),2*n_act)
print("Computed openfermion")
print("diff")
print(normal_ordered(wn_s2_op - wn_s2_op_test))
"""

"""
# Test [[Fn,S1],S1]
print("Test [[Fn,S1],S1]")
fn_s1_s1_dic = ducc.fn_s1_s1(fdic,t1dic)
fn_s1_s1_op = of.FermionOperator('',fn_s1_s1_dic['c'])
fn_s1_s1_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s1_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s1_s1_op_test = ducc.as_proj(normal_ordered(commutator(commutator(fop,s1_op),s1_op)),2*n_act)
print("diff")
print(fn_s1_s1_op - fn_s1_s1_op_test)
"""

"""
# Test [[Fn,S1],S2]
print("Test [[Fn,S1],S2]")
fn_s1_s2_dic = ducc.fn_s1_s2(fdic,t1dic,t2dic)
fn_s1_s2_op = normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s1_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s1_s2_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(fn_s1_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s1_s2_op_test = ducc.as_proj(normal_ordered(commutator(commutator(fop,s1_op),s2_op)),2*n_act)
print("diff")
print(fn_s1_s2_op - fn_s1_s2_op_test)
"""

"""
# Test [[Fn,S2],S1]
print("Test [[Fn,S2],S1]")
fn_s2_s1_dic = ducc.fn_s2_s1(fdic,t1dic,t2dic)
fn_s2_s1_op = of.FermionOperator('',fn_s2_s1_dic['c'])
fn_s2_s1_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s2_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s2_s1_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(fn_s2_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
print("Computed einsum")
fn_s2_s1_op_test = ducc.as_proj(normal_ordered(commutator(commutator(fop,s2_op),s1_op)),2*n_act)
print("diff")
print(fn_s2_s1_op - fn_s2_s1_op_test)
"""


"""
# Test [[Fn,S2],S2]
print("Test [[Fn,S2],S2]")
fn_s2_s2_dic = ducc.fn_s2_s2(fdic,t2dic)
fn_s2_s2_op = of.FermionOperator('',fn_s2_s2_dic['c'])
fn_s2_s2_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s2_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s2_s2_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(fn_s2_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s2_s2_op += normal_ordered(ducc.three_body_to_op(ducc.three_body_dic2ten(fn_s2_s2_dic,n_occ,n_act),n_occ,n_act))
print("Computed einsum")
fn_s2_s2_op_test = ducc.as_proj(normal_ordered(commutator(commutator(fop,s2_op),s2_op)),2*n_act)
print("diff")
print(fn_s2_s2_op - fn_s2_s2_op_test)
"""

"""
# Test [[Wn,S1],S1]
print("Test [[Wn,S1],S1]")
wn_s1_s1_dic = ducc.wn_s1_s1(vdic,t1dic)
wn_s1_s1_op = of.FermionOperator('',wn_s1_s1_dic['c'])
wn_s1_s1_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(wn_s1_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
#print("Constant")
#print(wn_s1_s1_op)
print("One-body terms")
print(wn_s1_s1_op)

wn_s1_s1_op_test = ducc.as_proj(normal_ordered(commutator(commutator(vop,s1_op),s1_op)),2*n_act)
print("Test")
terms = ducc.get_many_body_terms(wn_s1_s1_op_test)
#print(terms[0])
print(terms[1])
#print(terms[2])
"""

"""
# Not done yet
# Test [[Wn,S1],S2]
print("Test [[Wn,S1],S2]")
wn_s1_s2_dic = ducc.wn_s1_s2(vdic,t1dic,t2dic)
wn_s1_s2_op = of.FermionOperator('',wn_s1_s2_dic['c'])
wn_s1_s2_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(wn_s1_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
#print("Constant")
#print(wn_s1_s1_op)
print("One-body terms")
print(wn_s1_s2_op)

wn_s1_s2_op_test = ducc.as_proj(normal_ordered(commutator(commutator(vop,s1_op),s2_op)),2*n_act)
print("Test")
terms = ducc.get_many_body_terms(wn_s1_s2_op_test)
print(terms[0])
print(terms[1])
print(terms[2])
"""

"""
# Done
# Test [[[Fn,S1],S1],S1]
print("Test [[[Fn,S1],S1],S1]")
fn_s1_s1_s1_dic = ducc.fn_s1_s1_s1(fdic,t1dic)
fn_s1_s1_s1_op = of.FermionOperator('',fn_s1_s1_s1_dic['c'])
fn_s1_s1_s1_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s1_s1_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s1_s1_s1_op_test = ducc.as_proj(normal_ordered(commutator(commutator(commutator(fop,s1_op),s1_op),s1_op)),2*n_act)
print("diff")
print(fn_s1_s1_s1_op - fn_s1_s1_s1_op_test)
"""

"""
# not able to catch the pattern so - antisymmetrized, not normalized
# Test [[[Fn,S1],S1],S2]
print("Test [[[Fn,S1],S1],S2]")
fn_s1_s1_s2_dic = ducc.fn_s1_s1_s2(fdic,t1dic,t2dic)
fn_s1_s1_s2_op = normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s1_s1_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s1_s1_s2_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(fn_s1_s1_s2_dic,n_occ,n_act,n_act),n_occ,n_act))
print(fn_s1_s1_s2_op)
fn_s1_s1_s2_op_test = ducc.as_proj(normal_ordered(commutator(commutator(commutator(fop,s1_op),s1_op),s2_op)),2*n_act)
print("Test")
#terms = ducc.get_many_body_terms(fn_s1_s1_s2_op_test)
print(fn_s1_s1_s2_op_test)
print("diff")
print(fn_s1_s1_s2_op_test-fn_s1_s1_s2_op)
"""

"""
# Test [[[Fn,S1],S2],S1]
print("Test [[[Fn,S1],S2],S1]")
fn_s1_s2_s1_dic = ducc.fn_s1_s2_s1(fdic,t1dic,t2dic)
fn_s1_s2_s1_op = normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s1_s2_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s1_s2_s1_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(fn_s1_s2_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
print(fn_s1_s2_s1_op)
fn_s1_s2_s1_op_test = ducc.as_proj(normal_ordered(commutator(commutator(commutator(fop,s1_op),s2_op),s1_op)),2*n_act)
print("Test")
terms = ducc.get_many_body_terms(fn_s1_s2_s1_op_test)
print(terms[0])
print(terms[1])
print(terms[2])
print("diff")
print(fn_s1_s2_s1_op_test-fn_s1_s2_s1_op)
"""
"""
# Test [[[Fn,S2],S1],S1]
print("Test [[[Fn,S2],S1],S1]")
fn_s2_s1_s1_dic = ducc.fn_s2_s1_s1(fdic,t1dic,t2dic)
fn_s2_s1_s1_op = of.FermionOperator('',fn_s2_s1_s1_dic['c'])
fn_s2_s1_s1_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s2_s1_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s2_s1_s1_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(fn_s2_s1_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
terms_1 = ducc.get_many_body_terms(fn_s2_s1_s1_op)
print(terms_1[0])
print(terms_1[1])
print(terms_1[2])
fn_s2_s1_s1_op_test = ducc.as_proj(normal_ordered(commutator(commutator(commutator(fop,s2_op),s1_op),s1_op)),2*n_act)
print("Test")
terms = ducc.get_many_body_terms(fn_s2_s1_s1_op_test)
print(terms[0])
print(terms[1])
print(terms[2])
print("diff")
#print(fn_s2_s1_s1_op_test-fn_s2_s1_s1_op)
"""
"""
# Test [[[Fn,S1],S2],S2]
print("Test [[[Fn,S1],S2],S2]")
fn_s2_s1_s1_dic = ducc.fn_s2_s1_s1(fdic,t1dic,t2dic)
fn_s2_s1_s1_op = of.FermionOperator('',fn_s2_s1_s1_dic['c'])
fn_s2_s1_s1_op += normal_ordered(ducc.one_body_to_op(ducc.one_body_dic2mat(fn_s2_s1_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
fn_s2_s1_s1_op += normal_ordered(ducc.two_body_to_op(ducc.two_body_dic2ten(fn_s2_s1_s1_dic,n_occ,n_act,n_act),n_occ,n_act))
terms_1 = ducc.get_many_body_terms(fn_s2_s1_s1_op)
print(terms_1[0])
print(terms_1[1])
print(terms_1[2])
fn_s2_s1_s1_op_test = ducc.as_proj(normal_ordered(commutator(commutator(commutator(fop,s2_op),s1_op),s1_op)),2*n_act)
print("Test")
terms = ducc.get_many_body_terms(fn_s2_s1_s1_op_test)
print(terms[0])
print(terms[1])
print(terms[2])
print("diff")
#print(fn_s2_s1_s1_op_test-fn_s2_s1_s1_op)
"""