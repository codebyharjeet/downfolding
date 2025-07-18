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


geometry = [('Be', (0,0,0))]
charge = 0
spin = 0
basis = 'cc-pvdz'
[n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = ducc.init(geometry,charge,spin,basis,reference='rhf')

sq_ham = ducc.SQ_Hamiltonian()
sq_ham.init(h, g, C, S)
print(" HF Energy: %21.15f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))

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
t1_amps, t2_amps = ducc.get_t_ext(t1_amps,t2_amps,n_a,n_b,n_act)

fmat = sq_ham.make_f(range(n_a),range(n_b))
fdic = ducc.one_body_mat2dic(fmat,n_occ,n_act,n_orb)
fop = ducc.one_body_to_op(fmat,n_occ,n_orb)

vten = 0.25*sq_ham.make_v()
vdic = ducc.two_body_ten2dic(vten,n_occ,n_act,n_orb)
vop = ducc.two_body_to_op(vten,n_occ,n_orb)


t1dic = ducc.t1_mat2dic(t1_amps,n_a,n_act,n_orb)
s1_op = ducc.t1_to_op(t1_amps)

t2dic = ducc.t2_ten2dic(t2_amps,n_a,n_act,n_orb)
s2_op = ducc.t2_to_op(t2_amps)

#bare_ham = fop + vop
#print("ham constructed")
#ham_sparse = get_sparse_operator(bare_ham)
#energy_2, state = get_ground_state(ham_sparse)
#print("fci energy sparse = ",energy_2)

constant = 0
one_body = np.zeros((2*n_act,2*n_act))
two_body = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act))
three_body = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))

one_body += fmat[0:2*n_act,0:2*n_act]
two_body += vten[0:2*n_act,0:2*n_act,0:2*n_act,0:2*n_act]  

fn_s1_dic = ducc.fn_s1(fdic,t1dic)
constant += fn_s1_dic['c']
one_body += ducc.one_body_dic2mat(fn_s1_dic,n_occ,n_act,n_act)

fn_s2_dic = ducc.fn_s2(fdic,t2dic)
one_body += ducc.one_body_dic2mat(fn_s2_dic,n_occ,n_act,n_act)
two_body += ducc.two_body_dic2ten(fn_s2_dic,n_occ,n_act,n_act)

wn_s1_dic = ducc.wn_s1(vdic,t1dic)
one_body += ducc.one_body_dic2mat(wn_s1_dic,n_occ,n_act,n_act)
two_body += ducc.two_body_dic2ten(wn_s1_dic,n_occ,n_act,n_act)

wn_s2_dic = ducc.wn_s2(vdic,t2dic)
constant += wn_s2_dic["c"]
one_body += ducc.one_body_dic2mat(wn_s2_dic,n_occ,n_act,n_act)
two_body += ducc.two_body_dic2ten(wn_s2_dic,n_occ,n_act,n_act)
three_body += ducc.three_body_dic2ten(wn_s2_dic,n_occ,n_act)

fn_s1_s1_dic = ducc.fn_s1_s1(fdic,t1dic)
constant += 0.500*fn_s1_s1_dic['c']
one_body += 0.500*ducc.one_body_dic2mat(fn_s1_s1_dic,n_occ,n_act,n_act)

fn_s1_s2_dic = ducc.fn_s1_s2(fdic,t1dic,t2dic)
one_body += 0.500*ducc.one_body_dic2mat(fn_s1_s2_dic,n_occ,n_act,n_act)
two_body += 0.500*ducc.two_body_dic2ten(fn_s1_s2_dic,n_occ,n_act,n_act)

fn_s2_s1_dic = ducc.fn_s2_s1(fdic,t1dic,t2dic)
constant += 0.500*fn_s2_s1_dic['c']
one_body += 0.500*ducc.one_body_dic2mat(fn_s2_s1_dic,n_occ,n_act,n_act)
two_body += 0.500*ducc.two_body_dic2ten(fn_s2_s1_dic,n_occ,n_act,n_act)

fn_s2_s2_dic = ducc.fn_s2_s2(fdic,t2dic)
constant += 0.500*fn_s2_s2_dic['c']
one_body += 0.500*ducc.one_body_dic2mat(fn_s2_s2_dic,n_occ,n_act,n_act)
two_body += 0.500*ducc.two_body_dic2ten(fn_s2_s2_dic,n_occ,n_act,n_act)
three_body += 0.500*ducc.three_body_dic2ten(fn_s2_s2_dic,n_occ,n_act)

constant_op = of.FermionOperator("", float(constant))
one_body_op = normal_ordered(ducc.one_body_to_op(one_body, n_occ, n_act))
two_body_op = normal_ordered(ducc.two_body_to_op(two_body, n_occ, n_act))
three_body_op = normal_ordered(ducc.three_body_to_op(three_body, n_occ, n_act))



a4_3_ham = constant_op + one_body_op + two_body_op 
print("DUCC Ham constructed")
ducc_ham_sparse = get_sparse_operator(a4_3_ham)
ducc_energy, ducc_state = get_ground_state(ducc_ham_sparse)
print("DUCC Energy = ",ducc_energy)

print("DUCC openfermion = ",eigenspectrum(a4_3_ham)[0].real)







