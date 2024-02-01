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




def one_body_to_matrix(operator, n_orb):
	"""
	Converts normal-ordered one-body fermionic operator to dense matrix
	F = f_{pq} p^ q
	"""
	one_body_mat = np.zeros((2*n_orb,2*n_orb))
	terms = operator.terms 
	for term in terms:
		one_body_mat[term[0][0],term[1][0]] = terms.get(term)
	return one_body_mat

def one_body_to_matrix_ph(operator, n_orb, n_occ):
	"""
	Converts particle-hole normal-ordered one-body fermionic operator to dense matrix
	"""
	one_body_mat = np.zeros((2*n_orb,2*n_orb))
	terms = operator.terms 
	for term in terms: 
		# OO
		if((term[0][0] < n_occ) and (term[1][0] < n_occ)):
			one_body_mat[term[1][0],term[0][0]] += -terms.get(term)
		# OV
		elif(not(term[0][1])):
			one_body_mat[term[1][0],term[0][0]] += -terms.get(term)
		# VO
		elif(term[1][1]):
			one_body_mat[term[0][0],term[1][0]] +=  terms.get(term)
		# VV
		elif((term[0][0] >= n_occ) and (term[1][0] >= n_occ)):
			one_body_mat[term[0][0],term[1][0]] +=  terms.get(term)
	return one_body_mat

def two_body_to_tensor(operator, n_orb):
	"""
	Converts normal-ordered two-body fermionic operator to dense tensor
	V = v_pqrs p^ q^ s r
	"""
	# if (operator.many_body_order != 4):
	# 	print("Error: not a two-body operator")
	# 	exit()
	two_body_tensor = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
	terms = operator.terms 
	for term in terms:
		two_body_tensor[term[0][0],term[1][0],term[3][0],term[2][0]] = 0.25*terms.get(term)
		two_body_tensor[term[1][0],term[0][0],term[3][0],term[2][0]] = -0.25*terms.get(term)
		two_body_tensor[term[0][0],term[1][0],term[2][0],term[3][0]] = -0.25*terms.get(term)
		two_body_tensor[term[1][0],term[0][0],term[2][0],term[3][0]] = 0.25*terms.get(term)
	return two_body_tensor

def two_body_to_tensor_ph(operator, n_orb, n_occ):
	"""
	Converts particle-hole normal-ordered two-body fermionic operator to dense tensor
	"""
	two_body_tens = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
	terms = operator.terms 
	for term in terms:
		p = term[0][0]
		q = term[1][0]
		s = term[2][0]
		r = term[3][0]
		# OOOO
		if((p < n_occ) and (q < n_occ) and (s < n_occ) and (r < n_occ)):
			two_body_tens[s,r,q,p] +=  0.25*terms.get(term)
			two_body_tens[s,r,p,q] += -0.25*terms.get(term)
			two_body_tens[r,s,q,p] += -0.25*terms.get(term)
			two_body_tens[r,s,p,q] +=  0.25*terms.get(term)
		# OOOV and OOVO
		elif((p < n_occ) and (q >= n_occ) and (s < n_occ) and (r < n_occ)):
			two_body_tens[s,r,p,q] += -0.25*terms.get(term)
			two_body_tens[s,r,q,p] +=  0.25*terms.get(term)
			two_body_tens[r,s,p,q] +=  0.25*terms.get(term)
			two_body_tens[r,s,q,p] += -0.25*terms.get(term)
		# OVOO and VOOO
		elif((p >= n_occ) and (q < n_occ) and (s < n_occ) and (r < n_occ)):
			two_body_tens[r,p,s,q] += -0.25*terms.get(term)
			two_body_tens[r,p,q,s] +=  0.25*terms.get(term)
			two_body_tens[p,r,s,q] +=  0.25*terms.get(term)
			two_body_tens[p,r,q,s] += -0.25*terms.get(term)
		# OOVV and VVOO
		elif((p >= n_occ) and (q >= n_occ) and (s < n_occ) and (r < n_occ)):
			two_body_tens[s,r,q,p] +=  0.125*terms.get(term) 
			two_body_tens[s,r,p,q] += -0.125*terms.get(term)
			two_body_tens[r,s,q,p] += -0.125*terms.get(term)
			two_body_tens[r,s,p,q] +=  0.125*terms.get(term)

			two_body_tens[p,q,r,s] +=  0.125*terms.get(term)
			two_body_tens[p,q,s,r] += -0.125*terms.get(term)
			two_body_tens[q,p,r,s] += -0.125*terms.get(term)
			two_body_tens[q,p,s,r] +=  0.125*terms.get(term)
		# OVOV and VOOV and OVVO and VOVO
		elif((p >= n_occ) and (q < n_occ) and (s >= n_occ) and (r < n_occ)):
			two_body_tens[r,p,q,s] +=  0.25*terms.get(term)
			two_body_tens[p,r,q,s] += -0.25*terms.get(term)
			two_body_tens[r,p,s,q] += -0.25*terms.get(term)
			two_body_tens[p,r,s,q] +=  0.25*terms.get(term)
		# VVVO and VVOV
		elif((p >= n_occ) and (q >= n_occ) and (s < n_occ) and (r >= n_occ)):
			two_body_tens[p,q,r,s] +=  0.25*terms.get(term)
			two_body_tens[q,p,r,s] += -0.25*terms.get(term)
			two_body_tens[p,q,s,r] += -0.25*terms.get(term)
			two_body_tens[q,p,s,r] +=  0.25*terms.get(term)
		# VOVV and OVVV
		elif((p >= n_occ) and (q >= n_occ) and (s >= n_occ) and (r < n_occ)):
			two_body_tens[p,r,s,q] +=  0.25*terms.get(term)
			two_body_tens[p,r,q,s] += -0.25*terms.get(term)
			two_body_tens[r,p,s,q] += -0.25*terms.get(term)
			two_body_tens[r,p,q,s] +=  0.25*terms.get(term)
		# VVVV
		elif((p >= n_occ) and (q >= n_occ) and (s >= n_occ) and (r >= n_occ)):
			two_body_tens[p,q,r,s] +=  0.25*terms.get(term)
			two_body_tens[p,q,s,r] += -0.25*terms.get(term)
			two_body_tens[q,p,r,s] += -0.25*terms.get(term)
			two_body_tens[q,p,s,r] +=  0.25*terms.get(term)
	return two_body_tens



def three_body_to_tensor(operator, n_orb):
	"""
	Converts normal_ordered three-body fermionic operator to dense tensor
	W = w_pqrstu p^ q^ r^ u t s
	"""
	# if (operator.many_body_order != 6):
	# 	print("Error: not a three-body operator")
	# 	exit()
	three_body_tensor = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb,
								  2*n_orb))
	terms = operator.terms 
	for term in terms: 
		coeff = terms.get(term)
		p = term[0][0]
		q = term[1][0]
		r = term[2][0]
		s = term[5][0]
		t = term[4][0]
		u = term[3][0]

		three_body_tensor[p,q,r,s,t,u] =  (1.0/36.0)*coeff
		three_body_tensor[p,q,r,s,u,t] = -(1.0/36.0)*coeff
		three_body_tensor[p,q,r,t,s,u] = -(1.0/36.0)*coeff
		three_body_tensor[p,q,r,t,u,s] =  (1.0/36.0)*coeff
		three_body_tensor[p,q,r,u,s,t] =  (1.0/36.0)*coeff
		three_body_tensor[p,q,r,u,t,s] = -(1.0/36.0)*coeff

		three_body_tensor[p,r,q,s,t,u] = -(1.0/36.0)*coeff
		three_body_tensor[p,r,q,s,u,t] =  (1.0/36.0)*coeff
		three_body_tensor[p,r,q,t,s,u] =  (1.0/36.0)*coeff
		three_body_tensor[p,r,q,t,u,s] = -(1.0/36.0)*coeff
		three_body_tensor[p,r,q,u,s,t] = -(1.0/36.0)*coeff
		three_body_tensor[p,r,q,u,t,s] =  (1.0/36.0)*coeff

		three_body_tensor[q,p,r,s,t,u] = -(1.0/36.0)*coeff
		three_body_tensor[q,p,r,s,u,t] =  (1.0/36.0)*coeff
		three_body_tensor[q,p,r,t,s,u] =  (1.0/36.0)*coeff
		three_body_tensor[q,p,r,t,u,s] = -(1.0/36.0)*coeff
		three_body_tensor[q,p,r,u,s,t] = -(1.0/36.0)*coeff
		three_body_tensor[q,p,r,u,t,s] =  (1.0/36.0)*coeff

		three_body_tensor[q,r,p,s,t,u] =  (1.0/36.0)*coeff
		three_body_tensor[q,r,p,s,u,t] = -(1.0/36.0)*coeff
		three_body_tensor[q,r,p,t,s,u] = -(1.0/36.0)*coeff
		three_body_tensor[q,r,p,t,u,s] =  (1.0/36.0)*coeff
		three_body_tensor[q,r,p,u,s,t] =  (1.0/36.0)*coeff
		three_body_tensor[q,r,p,u,t,s] = -(1.0/36.0)*coeff

		three_body_tensor[r,p,q,s,t,u] =  (1.0/36.0)*coeff
		three_body_tensor[r,p,q,s,u,t] = -(1.0/36.0)*coeff
		three_body_tensor[r,p,q,t,s,u] = -(1.0/36.0)*coeff
		three_body_tensor[r,p,q,t,u,s] =  (1.0/36.0)*coeff
		three_body_tensor[r,p,q,u,s,t] =  (1.0/36.0)*coeff
		three_body_tensor[r,p,q,u,t,s] = -(1.0/36.0)*coeff

		three_body_tensor[r,q,p,s,t,u] = -(1.0/36.0)*coeff
		three_body_tensor[r,q,p,s,u,t] =  (1.0/36.0)*coeff
		three_body_tensor[r,q,p,t,s,u] =  (1.0/36.0)*coeff
		three_body_tensor[r,q,p,t,u,s] = -(1.0/36.0)*coeff
		three_body_tensor[r,q,p,u,s,t] = -(1.0/36.0)*coeff
		three_body_tensor[r,q,p,u,t,s] =  (1.0/36.0)*coeff 

	return three_body_tensor



def one_body_to_op(one_body_mat,n_orb,n_occ):
    print(n_orb)
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

def two_body_to_op(two_body_tens,n_orb,n_occ):
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

def three_body_to_op(three_body_tens,n_orb,n_occ):
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
	for p in range(0,n_occ):
		for q in range(0,n_occ):
			for r in range(0,n_occ):
				for s in range(n_occ,2*n_orb):
					for t in range(n_occ,2*n_orb):
						for u in range(n_occ,2*n_orb):
							# OOO|VVV
							three_body_op += of.FermionOperator(((u,0),(t,0),(s,0),(p,1),(q,1),(r,1)), -three_body_tens[p,q,r,s,t,u])
							# OOV|OVV
							three_body_op += of.FermionOperator(((s,1),(r,0),(u,0),(t,0),(p,1),(q,1)),  three_body_tens[p,q,s,r,t,u])
							# OVO|OVV
							three_body_op += of.FermionOperator(((s,1),(q,0),(u,0),(t,0),(p,1),(r,1)), -three_body_tens[p,s,r,q,t,u])
							# VOO|OVV
							three_body_op += of.FermionOperator(((s,1),(p,0),(u,0),(t,0),(q,1),(r,1)),  three_body_tens[s,q,r,p,t,u])
							# OOV|VOV
							three_body_op += of.FermionOperator(((t,1),(r,0),(u,0),(s,0),(p,1),(q,1)), -three_body_tens[p,q,t,s,r,u])
							# OVO|VOV
							three_body_op += of.FermionOperator(((t,1),(q,0),(u,0),(s,0),(p,1),(r,1)),  three_body_tens[p,t,r,s,q,u])
							# VOO|VOV
							three_body_op += of.FermionOperator(((t,1),(p,0),(u,0),(s,0),(q,1),(r,1)), -three_body_tens[t,q,r,s,p,u])
							# OOV|VVO
							three_body_op += of.FermionOperator(((u,1),(r,0),(t,0),(s,0),(p,1),(q,1)),  three_body_tens[p,q,u,s,t,r])
							# OVO|VVO
							three_body_op += of.FermionOperator(((u,1),(q,0),(t,0),(s,0),(p,1),(q,1)), -three_body_tens[p,u,r,s,t,q])
							# VOO|VVO
							three_body_op += of.FermionOperator(((u,1),(p,0),(t,0),(s,0),(q,1),(r,1)),  three_body_tens[u,q,r,s,t,p])
							# OVV|OOV
							three_body_op += of.FermionOperator(((s,1),(t,1),(r,0),(q,0),(u,0),(p,1)), -three_body_tens[p,s,t,q,r,u])
							# VOV|OOV
							three_body_op += of.FermionOperator(((s,1),(t,1),(r,0),(p,0),(u,0),(q,1)),  three_body_tens[s,q,t,p,r,u])
							# VVO|OOV
							three_body_op += of.FermionOperator(((s,1),(t,1),(q,0),(p,0),(u,0),(r,1)), -three_body_tens[s,t,r,p,q,u])
							# OVV|OVO
							three_body_op += of.FermionOperator(((s,1),(u,1),(r,0),(q,0),(t,0),(p,1)),  three_body_tens[p,s,u,q,t,r])
							# VOV|OVO
							three_body_op += of.FermionOperator(((s,1),(u,1),(r,0),(p,0),(t,0),(q,1)), -three_body_tens[s,q,u,p,t,r])
							# VVO|OVO
							three_body_op += of.FermionOperator(((s,1),(u,1),(q,0),(p,0),(t,0),(r,1)),  three_body_tens[s,u,r,p,t,q])
							# OVV|VOO
							three_body_op += of.FermionOperator(((t,1),(u,1),(r,0),(q,0),(s,0),(p,1)), -three_body_tens[p,t,u,s,q,r])
							# VOV|VOO
							three_body_op += of.FermionOperator(((t,1),(u,1),(r,0),(p,0),(s,0),(q,1)),  three_body_tens[t,q,u,s,p,r])
							# VVO|VOO
							three_body_op += of.FermionOperator(((t,1),(u,1),(q,0),(p,0),(s,0),(r,1)), -three_body_tens[t,u,r,s,p,q])
							# VVV|OOO
							three_body_op += of.FermionOperator(((s,1),(t,1),(u,1),(r,0),(q,0),(p,0)),  three_body_tens[s,t,u,p,q,r])
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


def make_full_one(big,small,n_a,n_b,n_orb):
    if n_a+n_b==small.shape[0]:
        ind_0 = slice(0,n_a+n_b)
    else:
        ind_0 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[1]:
        ind_1 = slice(0,n_a+n_b)
    else:
        ind_1 = slice(n_a+n_b,2*n_orb)
    big[ind_0,ind_1] = small
    return big


# function to create two body matrix      
def make_full_two(big,small,n_a,n_b,n_orb):
    if n_a+n_b==small.shape[0]:
        ind_0 = slice(0,n_a+n_b)
    else:
        ind_0 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[1]:
        ind_1 = slice(0,n_a+n_b)
    else:
        ind_1 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[2]:
        ind_2 = slice(0,n_a+n_b)
    else:
        ind_2 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[3]:
        ind_3 = slice(0,n_a+n_b)
    else:
        ind_3 = slice(n_a+n_b,2*n_orb)
    big[ind_0,ind_1,ind_2,ind_3] = small
    return big


# function to create three body matrix      
def make_full_three(big,small,n_a,n_b,n_orb):
    if n_a+n_b==small.shape[0]:
        ind_0 = slice(0,n_a+n_b)
    else:
        ind_0 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[1]:
        ind_1 = slice(0,n_a+n_b)
    else:
        ind_1 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[2]:
        ind_2 = slice(0,n_a+n_b)
    else:
        ind_2 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[3]:
        ind_3 = slice(0,n_a+n_b)
    else:
        ind_3 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[4]:
        ind_4 = slice(0,n_a+n_b)
    else:
        ind_4 = slice(n_a+n_b,2*n_orb)
    if n_a+n_b==small.shape[5]:
        ind_5= slice(0,n_a+n_b)
    else:
        ind_5 = slice(n_a+n_b,2*n_orb)
    big[ind_0,ind_1,ind_2,ind_3,ind_4,ind_5] = small
    return big



def transform_t_spatial_to_spin(t1_amps, t2_amps, n_a, n_b, n_orb):
    # expanding the t1 amplitude into alpha and beta space
    t1 = np.zeros((n_a+n_b,2*n_orb-n_a-n_b))
    for i in range(0,n_a):
        for a in range(n_a,n_orb):
            ia = 2*i
            aa = 2*a
            t1[ia,aa-(n_a+n_b)] = t1_amps[0][i,a-n_a]
    for i in range(0,n_b):
        for a in range(n_b,n_orb):
            ib = 2*i+1
            ab = 2*a+1
            t1[ib,ab-(n_a+n_b)] = t1_amps[1][i,a-n_b]

    print(t1.shape)
    #print(t1)

    # expanding the t2 amplitude_aa
    t2 = np.zeros((n_a+n_b,n_a+n_b,2*n_orb-n_a-n_b,2*n_orb-n_a-n_b))
    for i in range(0,n_a):
        ia = 2*i
        for j in range(0,n_a):
            ja = 2*j
            for a in range(n_a,n_orb):
                aa = 2*a
                for b in range(n_a,n_orb):
                    ba = 2*b
                    t2[ia,ja,aa-(n_a+n_b),ba-(n_a+n_b)]=t2_amps[0][i,j,a-n_a,b-n_a]
          
    # expanding the t2 amplitude_ab
    for i in range(0,n_a):
        ia = 2*i
        for j in range(0,n_b):
            jb = 2*j+1
            for a in range(n_a,n_orb):
                aa = 2*a
                for b in range(n_b,n_orb):
                    bb = 2*b+1 
                    t2[ia,jb,aa-(n_a+n_b),bb-(n_a+n_b)]=t2_amps[1][i,j,a-n_a,b-n_b]
                    t2[jb,ia,bb-(n_a+n_b),aa-(n_a+n_b)]=t2_amps[1][i,j,a-n_a,b-n_b]
                    t2[ia,jb,bb-(n_a+n_b),aa-(n_a+n_b)]=-t2_amps[1][i,j,a-n_a,b-n_b]
                    t2[jb,ia,aa-(n_a+n_b),bb-(n_a+n_b)]=-t2_amps[1][i,j,a-n_a,b-n_b]
                    
    # expanding the t2 amplitude_bb
    for i in range(0,n_b):
        ib = 2*i+1
        for j in range(0,n_b):
            jb = 2*j+1
            for a in range(n_b,n_orb):
                ab = 2*a+1
                for b in range(n_b,n_orb):
                    bb = 2*b+1 
                    t2[ib,jb,ab-(n_a+n_b),bb-(n_a+n_b)]=t2_amps[2][i,j,a-n_b,b-n_b]

    print(t2.shape)
    return t1, t2


def get_spin_to_spatial(integral,n_orb):
    if(np.ndim(integral) == 0):
        print("It is a constant term.")
        exit()
    elif(np.ndim(integral) == 2):
        integral_spatial = np.zeros((n_orb,n_orb))
        for p in range(0,n_orb):
            pa = 2*p
            for q in range(0,n_orb):
                qa = 2*q
                qb = 2*q + 1
                integral_spatial[p,q] = integral[pa,qa]
    elif(np.ndim(integral) == 4): 
        integral_spatial = np.zeros((n_orb,n_orb,n_orb,n_orb))
        for p in range(0,n_orb):
            pa = 2*p
            pb = 2*p+1
            for q in range(0,n_orb):
                qa = 2*q
                qb = 2*q+1
                for r in range(0,n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    for s in range(0,n_orb):
                        sa = 2*s
                        sb = 2*s+1
                        # vmat_spatial[p,q,r,s] = vmat[pa,qb,ra,sb]
                        integral_spatial[p,r,q,s] = integral[pa,qb,ra,sb]
                        #integral_spatial[p,r,q,s] = integral[pb,qa,rb,sa]
                        #vmat_spatial[p,q,r,s] = vmat[pa, ra,qb,sb]
        
    return integral_spatial


def get_spatial_to_spin(integral,n_orb):
    A = np.array([[1,0],[0,0]])
    B = np.array([[0,0],[0,1]])
    AA = np.einsum("pq,rs->pqrs",A,A)
    AB = np.einsum("pq,rs->pqrs",A,B)
    BA = np.einsum("pq,rs->pqrs",B,A)
    BB = np.einsum("pq,rs->pqrs",B,B)

    if(np.ndim(integral) == 0):
        print("It is a constant term.")
        exit()

    elif(np.ndim(integral) == 2):
        integral_spin = np.kron(integral,A) + np.kron(integral,B)

    elif(np.ndim(integral) == 4): 
        integral_spin = np.kron(integral,AA) + np.kron(integral,AB) + np.kron(integral,BA) + np.kron(integral,BB)
    return integral_spin

