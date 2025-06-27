import scipy
import numpy as np 
import copy as cp 
import pyscf
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, cc, lib
from pyscf.cc import ccsd
from downfolding.hamiltonian import HamFormat, Hamiltonian
from downfolding.ccsd import calc_ccsd
from downfolding.helper import asym_term, one_body_mat2dic, one_body_dic2mat, two_body_ten2dic, two_body_dic2ten, three_body_ten2dic, three_body_dic2ten, four_body_ten2dic, four_body_dic2ten, t1_mat2dic, t2_ten2dic, get_many_body_terms, as_proj, t1_to_op, t2_to_op, t1_to_ext, t2_to_ext
from downfolding.printing import ccsd_summary
import time 
from opt_einsum import contract

# DUCC functions
def fn_s1(f,t1):
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize dictionary
	fs1 = {
	"c":  0.0,
	"oo": np.zeros((n_occ,n_occ)),
	"ov": np.zeros((n_occ,n_virt_int)),
	"vo": np.zeros((n_virt_int,n_occ)), 
	}
	# Populate [Fn,S_1ext]
	fs1["c"]  += 1.000 * contract("Ai,iA->",f["Vo"],t1["oV"]) # o*V
	fs1["c"]  += 1.000 * contract("iA,Ai->",f["oV"],t1["Vo"]) # o*V

	fs1["oo"] += 1.000 * contract("Ai,jA->ji",f["Vo"],t1["oV"]) # o*o*V
	fs1["oo"] += 1.000 * contract("iA,Aj->ij",f["oV"],t1["Vo"]) # o*o*V

	fs1["ov"] += 1.000 * contract("Aa,iA->ia",f["Vv"],t1["oV"]) # o*v*V

	fs1["vo"] += 1.000 * contract("aA,Ai->ai",f["vV"],t1["Vo"]) # o*v*V

	return fs1  

def fn_s2(f,t2):
	# [Fn,S_2ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize dictionary
	fs2 = {
		"ov":   np.zeros((n_occ,n_virt_int)),
		"vo":   np.zeros((n_virt_int,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	}
	# Populate [Fn,S_2ext]
	fs2["ov"] +=  1.000 * contract("Aj,ijaA->ia",f["Vo"],t2["oovV"]) # o*o*v*V

	fs2["vo"] +=  1.000 * contract("jA,aAij->ai",f["oV"],t2["vVoo"]) # o*o*v*V

	# ooov += np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2["ooov"] += -0.250 * contract("Ai,jkaA->jkia",f["Vo"],t2["oovV"])
	fs2["ooov"] = asym_term(fs2["ooov"],"ooov")

	# ovoo += np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2["ovoo"] += -0.250 * contract("iA,aAjk->iajk",f["oV"],t2["vVoo"])
	fs2["ovoo"] = asym_term(fs2["ovoo"],"ovoo")

	# oovv += np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	fs2["oovv"] += -0.500 * contract("Aa,ijbA->ijab",f["Vv"],t2["oovV"])
	fs2["oovv"] = asym_term(fs2["oovv"],"oovv")

	# vvoo += np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	fs2["vvoo"] += -0.500 * contract("aA,bAij->abij",f["vV"],t2["vVoo"])
	fs2["vvoo"] = asym_term(fs2["vvoo"],"vvoo")

	return fs2 

def wn_s1(v,t1):
	# [Wn,S_1ext]
	# for sizing arrays
	n_occ = v["oooo"].shape[0]
	n_virt_int = v["vvvv"].shape[0]
	# initialize
	ws1 = {
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"vv": np.zeros((n_virt_int,n_virt_int)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"ovov": np.zeros((n_occ,n_virt_int,n_occ,n_virt_int)),
		"ovvv": np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ)),
		"vvov": np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	}
	# Populate [Wn,S_1ext]
	ws1["oo"] += 4.000 * contract("Ak,jkiA->ji",t1["Vo"],v["oooV"])
	ws1["oo"] += 4.000 * contract("kA,jAik->ji",t1["oV"],v["oVoo"])

	ws1["ov"] +=  4.000 * contract("Aj,ijaA->ia",t1["Vo"],v["oovV"])
	ws1["ov"] += -4.000 * contract("jA,iAja->ia",t1["oV"],v["oVov"])

	ws1["vo"] += -4.000 * contract("Aj,jaiA->ai",t1["Vo"],v["ovoV"])
	ws1["vo"] +=  4.000 * contract("jA,aAij->ai",t1["oV"],v["vVoo"])

	ws1["vv"] += -4.000 * contract("Ai,ibaA->ba",t1["Vo"],v["ovvV"])
	ws1["vv"] += -4.000 * contract("iA,bAia->ba",t1["oV"],v["vVov"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	ws1["oooo"] += -2.000 * contract("Ai,kljA->klij",t1["Vo"],v["oooV"])
	ws1["oooo"] += -2.000 * contract("iA,lAjk->iljk",t1["oV"],v["oVoo"])
	ws1["oooo"] = asym_term(ws1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	ws1["ooov"] += -2.000 * contract("iA,kAja->ikja",t1["oV"],v["oVov"])
	ws1["ooov"] += -1.000 * contract("Ai,jkaA->jkia",t1["Vo"],v["oovV"])
	ws1["ooov"] = asym_term(ws1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	ws1["ovoo"] += -2.000 * contract("Ai,kajA->kaij",t1["Vo"],v["ovoV"])
	ws1["ovoo"] += -1.000 * contract("iA,aAjk->iajk",t1["oV"],v["vVoo"])
	ws1["ovoo"] = asym_term(ws1["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	ws1["oovv"] += -2.000 * contract("iA,jAab->ijab",t1["oV"],v["oVvv"])
	ws1["oovv"] = asym_term(ws1["oovv"],"oovv")

	ws1["ovov"] += -1.000 * contract("Ai,jbaA->jbia",t1["Vo"],v["ovvV"])
	ws1["ovov"] += -1.000 * contract("iA,bAja->ibja",t1["oV"],v["vVov"])

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	ws1["vvoo"] += -2.000 * contract("Ai,abjA->abij",t1["Vo"],v["vvoV"])
	ws1["vvoo"] = asym_term(ws1["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	ws1["ovvv"] += -1.000 * contract("iA,cAab->icab",t1["oV"],v["vVvv"])
	ws1["ovvv"] = asym_term(ws1["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	ws1["vvov"] += -1.000 * contract("Ai,bcaA->bcia",t1["Vo"],v["vvvV"])
	ws1["vvov"] = asym_term(ws1["vvov"],"vvov")

	return ws1 

def wn_s2(v,t2,inc_3_body=True):
	# [Wn,S_2ext]
	# for sizing arrays
	n_occ = v["oooo"].shape[0]
	n_virt_int = v["vvvv"].shape[0]
	# initialize
	ws2 = {
		"c": 0.0, 
		"oo":   np.zeros((n_occ,n_occ)),
		"ov":   np.zeros((n_occ,n_virt_int)),
		"vo":   np.zeros((n_virt_int,n_occ)),
		"vv":   np.zeros((n_virt_int,n_virt_int)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"ovov": np.zeros((n_occ,n_virt_int,n_occ,n_virt_int)),
		"ovvv": np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ)),
		"vvov": np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
		}
	# Populate [Wn,S_2ext]
	ws2["c"] += 2.000 * contract("aAij,ijaA->",t2["vVoo"],v["oovV"])
	ws2["c"] += 1.000 * contract("ABij,ijAB->",t2["VVoo"],v["ooVV"])
	ws2["c"] += 2.000 * contract("ijaA,aAij->",t2["oovV"],v["vVoo"])
	ws2["c"] += 1.000 * contract("ijAB,ABij->",t2["ooVV"],v["VVoo"])

	ws2["oo"] += 4.000 * contract("aAik,jkaA->ji",t2["vVoo"],v["oovV"])
	ws2["oo"] += 2.000 * contract("ABik,jkAB->ji",t2["VVoo"],v["ooVV"])
	ws2["oo"] += 4.000 * contract("ikaA,aAjk->ij",t2["oovV"],v["vVoo"])
	ws2["oo"] += 2.000 * contract("ikAB,ABjk->ij",t2["ooVV"],v["VVoo"])

	ws2["ov"] += -2.000 * contract("jkaA,iAjk->ia",t2["oovV"],v["oVoo"])
	ws2["ov"] += -4.000 * contract("ijbA,bAja->ia",t2["oovV"],v["vVov"])
	ws2["ov"] += -2.000 * contract("ijAB,ABja->ia",t2["ooVV"],v["VVov"])

	ws2["vo"] += -4.000 * contract("bAij,jabA->ai",t2["vVoo"],v["ovvV"])
	ws2["vo"] += -2.000 * contract("ABij,jaAB->ai",t2["VVoo"],v["ovVV"])
	ws2["vo"] += -2.000 * contract("aAjk,jkiA->ai",t2["vVoo"],v["oooV"])

	ws2["vv"] += -2.000 * contract("aAij,ijbA->ab",t2["vVoo"],v["oovV"])
	ws2["vv"] += -2.000 * contract("ijaA,bAij->ba",t2["oovV"],v["vVoo"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	ws2["oooo"] +=  1.000 * contract("aAij,klaA->klij",t2["vVoo"],v["oovV"])
	ws2["oooo"] +=  1.000 * contract("ijaA,aAkl->ijkl",t2["oovV"],v["vVoo"])
	ws2["oooo"] +=  0.500 * contract("ABij,klAB->klij",t2["VVoo"],v["ooVV"])
	ws2["oooo"] +=  0.500 * contract("ijAB,ABkl->ijkl",t2["ooVV"],v["VVoo"])
	ws2["oooo"] = asym_term(ws2["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	ws2["ooov"] += -2.000 * contract("ilaA,kAjl->ikja",t2["oovV"],v["oVoo"])
	ws2["ooov"] +=  1.000 * contract("ijbA,bAka->ijka",t2["oovV"],v["vVov"])
	ws2["ooov"] +=  0.500 * contract("ijAB,ABka->ijka",t2["ooVV"],v["VVov"])
	ws2["ooov"] = asym_term(ws2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	ws2["ovoo"] += -2.000 * contract("aAil,kljA->kaij",t2["vVoo"],v["oooV"])
	ws2["ovoo"] +=  1.000 * contract("bAij,kabA->kaij",t2["vVoo"],v["ovvV"])
	ws2["ovoo"] +=  0.500 * contract("ABij,kaAB->kaij",t2["VVoo"],v["ovVV"])
	ws2["ovoo"] = asym_term(ws2["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	ws2["oovv"] += -4.000 * contract("ikaA,jAkb->ijab",t2["oovV"],v["oVov"])
	ws2["oovv"] +=  1.000 * contract("ijcA,cAab->ijab",t2["oovV"],v["vVvv"])
	ws2["oovv"] +=  0.500 * contract("ijAB,ABab->ijab",t2["ooVV"],v["VVvv"])
	ws2["oovv"] = asym_term(ws2["oovv"],"oovv")

	ws2["ovov"] += -1.000 * contract("aAik,jkbA->jaib",t2["vVoo"],v["oovV"])
	ws2["ovov"] += -1.000 * contract("ikaA,bAjk->ibja",t2["oovV"],v["vVoo"])

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	ws2["vvoo"] += -4.000 * contract("aAik,kbjA->abij",t2["vVoo"],v["ovoV"])
	ws2["vvoo"] +=  1.000 * contract("cAij,abcA->abij",t2["vVoo"],v["vvvV"])
	ws2["vvoo"] +=  0.500 * contract("ABij,abAB->abij",t2["VVoo"],v["vvVV"])
	ws2["vvoo"] = asym_term(ws2["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	ws2["ovvv"] += -2.000 * contract("ijaA,cAjb->icab",t2["oovV"],v["vVov"])
	ws2["ovvv"] = asym_term(ws2["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	ws2["vvov"] += -2.000 * contract("aAij,jcbA->acib",t2["vVoo"],v["ovvV"])
	ws2["vvov"] = asym_term(ws2["vvov"],"vvov")

	if(inc_3_body):
		ws2["ooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		ws2["oooovv"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2["ooovvv"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2["oovooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		ws2["oovoov"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2["oovovv"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2["oovvvv"] = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws2["ovvooo"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2["ovvoov"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2["vvvooo"] = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2["vvvoov"] = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		ws2["ooooov"] += -(1./3.) * contract("ijaA,mAkl->ijmkla",t2["oovV"],v["oVoo"])
		ws2["ooooov"] = asym_term(ws2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		ws2["oovooo"] += -(1./3.) * contract("aAij,lmkA->lmaijk",t2["vVoo"],v["oooV"])
		ws2["oovooo"] = asym_term(ws2["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2["oooovv"] +=  (2./3.) * contract("ijaA,lAkb->ijlkab",t2["oovV"],v["oVov"])
		ws2["oooovv"] = asym_term(ws2["oooovv"],"oooovv")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2["oovoov"] += -(1./9.) * contract("aAij,klbA->klaijb",t2["vVoo"],v["oovV"])
		ws2["oovoov"] += -(1./9.) * contract("ijaA,bAkl->ijbkla",t2["oovV"],v["vVoo"])
		ws2["oovoov"] = asym_term(ws2["oovoov"],"oovoov")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2["ovvooo"] +=  (2./3.) * contract("aAij,lbkA->labijk",t2["vVoo"],v["ovoV"])
		ws2["ovvooo"] = asym_term(ws2["ovvooo"],"ovvooo")

		# ooovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2["ooovvv"] += -1.000 * contract("ijaA,kAbc->ijkabc",t2["oovV"],v["oVvv"])
		ws2["ooovvv"] = asym_term(ws2["ooovvv"],"ooovvv")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2["oovovv"] +=  (2./9.) * contract("ijaA,cAkb->ijckab",t2["oovV"],v["vVov"])
		ws2["oovovv"] = asym_term(ws2["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2["ovvoov"] +=  (2./9.) * contract("aAij,kcbA->kacijb",t2["vVoo"],v["ovvV"])
		ws2["ovvoov"] = asym_term(ws2["ovvoov"],"ovvoov")

		# vvvooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2["vvvooo"] += -1.000 * contract("aAij,bckA->abcijk",t2["vVoo"],v["vvoV"])
		ws2["vvvooo"] = asym_term(ws2["vvvooo"],"vvvooo")

		# oovvvv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws2["oovvvv"] += -(1./3.) * contract("ijaA,dAbc->ijdabc",t2["oovV"],v["vVvv"])
		ws2["oovvvv"] = asym_term(ws2["oovvvv"],"oovvvv")

		# vvvoov = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2["vvvoov"] += -(1./3.) * contract("aAij,cdbA->acdijb",t2["vVoo"],v["vvvV"])
		ws2["vvvoov"] = asym_term(ws2["vvvoov"],"vvvoov")

	return ws2

def fn_s1_s1(f,t1):
	# [[Fn,S_1ext],S_1ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	fs1s1 = {
		"c": 0.0, 
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ))
	}
	# Populate [[Fn,S_1ext],S_1ext]
	fs1s1["c"] += -2.000 * contract("ji,iA,Aj->",f["oo"],t1["oV"],t1["Vo"]) # o * o * Vext
	fs1s1["c"] +=  2.000 * contract("BA,iB,Ai->",f["VV"],t1["oV"],t1["Vo"]) # o * Vext * Vext

	fs1s1["oo"] +=  2.000 * contract("BA,iB,Aj->ij",f["VV"],t1["oV"],t1["Vo"])
	fs1s1["oo"] += -1.000 * contract("ki,jA,Ak->ji",f["oo"],t1["oV"],t1["Vo"])
	fs1s1["oo"] += -1.000 * contract("ik,kA,Aj->ij",f["oo"],t1["oV"],t1["Vo"])

	fs1s1["ov"] += -1.000 * contract("ja,iA,Aj->ia",f["ov"],t1["oV"],t1["Vo"])

	fs1s1["vo"] += -1.000 * contract("aj,jA,Ai->ai",f["vo"],t1["oV"],t1["Vo"])

	return fs1s1

def fn_s1_s2(f,t1,t2):
	# [[Fn,S_1ext],S_2ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize
	fs1s2 = {
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	}
	# Populate [[Fn,S_1ext],S_2ext]
	fs1s2["ov"] += -1.000 * contract("kj,Ak,ijaA->ia",f["oo"],t1["Vo"],t2["oovV"])
	fs1s2["ov"] +=  1.000 * contract("BA,Aj,ijaB->ia",f["VV"],t1["Vo"],t2["oovV"])

	fs1s2["vo"] += -1.000 * contract("kj,jA,aAik->ai",f["oo"],t1["oV"],t2["vVoo"])
	fs1s2["vo"] +=  1.000 * contract("BA,jB,aAij->ai",f["VV"],t1["oV"],t2["vVoo"])

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs1s2["ooov"] +=  0.250 * contract("li,Al,jkaA->jkia",f["oo"],t1["Vo"],t2["oovV"])
	fs1s2["ooov"] += -0.250 * contract("BA,Ai,jkaB->jkia",f["VV"],t1["Vo"],t2["oovV"])
	fs1s2["ooov"] = asym_term(fs1s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs1s2["ovoo"] +=  0.250 * contract("il,lA,aAjk->iajk",f["oo"],t1["oV"],t2["vVoo"])
	fs1s2["ovoo"] += -0.250 * contract("BA,iB,aAjk->iajk",f["VV"],t1["oV"],t2["vVoo"])
	fs1s2["ovoo"] = asym_term(fs1s2["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	fs1s2["oovv"] +=  0.500 * contract("ka,Ak,ijbA->ijab",f["ov"],t1["Vo"],t2["oovV"])
	fs1s2["oovv"] = asym_term(fs1s2["oovv"],"oovv")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	fs1s2["vvoo"] +=  0.500 * contract("ak,kA,bAij->abij",f["vo"],t1["oV"],t2["vVoo"])
	fs1s2["vvoo"] = asym_term(fs1s2["vvoo"],"vvoo")

	return fs1s2

def fn_s2_s1(f,t1,t2):
	# [[Fn,S_2ext],S_1ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize
	fs2s1 = {
		"c": 0.0,
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"vv": np.zeros((n_virt_int,n_virt_int)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"ovov": np.zeros((n_occ,n_virt_int,n_occ,n_virt_int))
	}
	# Populate [[Fn,S_1ext],S_2ext]
	fs2s1["c"] += 1.000 * contract("ai,Aj,ijaA->",f["vo"],t1["Vo"],t2["oovV"])
	fs2s1["c"] += 1.000 * contract("Ai,Bj,ijAB->",f["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1["c"] += 1.000 * contract("ia,jA,aAij->",f["ov"],t1["oV"],t2["vVoo"])
	fs2s1["c"] += 1.000 * contract("iA,jB,ABij->",f["oV"],t1["oV"],t2["VVoo"])

	fs2s1["oo"] +=  1.000 * contract("ai,Ak,jkaA->ji",f["vo"],t1["Vo"],t2["oovV"])
	fs2s1["oo"] +=  1.000 * contract("Ai,Bk,jkAB->ji",f["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1["oo"] += -1.000 * contract("ak,Ai,jkaA->ji",f["vo"],t1["Vo"],t2["oovV"])
	fs2s1["oo"] += -1.000 * contract("Ak,Bi,jkAB->ji",f["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1["oo"] +=  1.000 * contract("ia,kA,aAjk->ij",f["ov"],t1["oV"],t2["vVoo"])
	fs2s1["oo"] += -1.000 * contract("ka,iA,aAjk->ij",f["ov"],t1["oV"],t2["vVoo"])
	fs2s1["oo"] +=  1.000 * contract("iA,kB,ABjk->ij",f["oV"],t1["oV"],t2["VVoo"])
	fs2s1["oo"] += -1.000 * contract("kA,iB,ABjk->ij",f["oV"],t1["oV"],t2["VVoo"])

	fs2s1["ov"] += -1.000 * contract("ij,Ak,jkaA->ia",f["oo"],t1["Vo"],t2["oovV"])
	fs2s1["ov"] += -1.000 * contract("kj,Ak,ijaA->ia",f["oo"],t1["Vo"],t2["oovV"])
	fs2s1["ov"] +=  1.000 * contract("ba,Aj,ijbA->ia",f["vv"],t1["Vo"],t2["oovV"])
	fs2s1["ov"] +=  1.000 * contract("Aa,Bj,ijAB->ia",f["Vv"],t1["Vo"],t2["ooVV"])
	fs2s1["ov"] +=  1.000 * contract("BA,Aj,ijaB->ia",f["VV"],t1["Vo"],t2["oovV"])

	fs2s1["vo"] += -1.000 * contract("ji,kA,aAjk->ai",f["oo"],t1["oV"],t2["vVoo"])
	fs2s1["vo"] += -1.000 * contract("kj,jA,aAik->ai",f["oo"],t1["oV"],t2["vVoo"])
	fs2s1["vo"] +=  1.000 * contract("ab,jA,bAij->ai",f["vv"],t1["oV"],t2["vVoo"])
	fs2s1["vo"] +=  1.000 * contract("aA,jB,ABij->ai",f["vV"],t1["oV"],t2["VVoo"])
	fs2s1["vo"] +=  1.000 * contract("BA,jB,aAij->ai",f["VV"],t1["oV"],t2["vVoo"])

	fs2s1["vv"] += -1.000 * contract("ai,Aj,ijbA->ab",f["vo"],t1["Vo"],t2["oovV"])
	fs2s1["vv"] += -1.000 * contract("ia,jA,bAij->ba",f["ov"],t1["oV"],t2["vVoo"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs2s1["oooo"] +=  0.500 * contract("ai,Aj,klaA->klij",f["vo"],t1["Vo"],t2["oovV"])
	fs2s1["oooo"] +=  0.500 * contract("Ai,Bj,klAB->klij",f["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1["oooo"] +=  0.500 * contract("ia,jA,aAkl->ijkl",f["ov"],t1["oV"],t2["vVoo"])
	fs2s1["oooo"] +=  0.500 * contract("iA,jB,ABkl->ijkl",f["oV"],t1["oV"],t2["VVoo"])
	fs2s1["oooo"] = asym_term(fs2s1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2s1["ooov"] += -0.500 * contract("il,Aj,klaA->ikja",f["oo"],t1["Vo"],t2["oovV"])
	fs2s1["ooov"] += -0.250 * contract("ba,Ai,jkbA->jkia",f["vv"],t1["Vo"],t2["oovV"])
	fs2s1["ooov"] += -0.250 * contract("Aa,Bi,jkAB->jkia",f["Vv"],t1["Vo"],t2["ooVV"])
	fs2s1["ooov"] += -0.250 * contract("BA,Ai,jkaB->jkia",f["VV"],t1["Vo"],t2["oovV"])
	fs2s1["ooov"] = asym_term(fs2s1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2s1["ovoo"] += -0.500 * contract("li,jA,aAkl->jaik",f["oo"],t1["oV"],t2["vVoo"])
	fs2s1["ovoo"] += -0.250 * contract("ab,iA,bAjk->iajk",f["vv"],t1["oV"],t2["vVoo"])
	fs2s1["ovoo"] += -0.250 * contract("aA,iB,ABjk->iajk",f["vV"],t1["oV"],t2["VVoo"])
	fs2s1["ovoo"] += -0.250 * contract("BA,iB,aAjk->iajk",f["VV"],t1["oV"],t2["vVoo"])
	fs2s1["ovoo"] = asym_term(fs2s1["ovoo"],"ovoo")

	fs2s1["ovov"] +=  0.250 * contract("ak,Ai,jkbA->jaib",f["vo"],t1["Vo"],t2["oovV"])
	fs2s1["ovov"] +=  0.250 * contract("ka,iA,bAjk->ibja",f["ov"],t1["oV"],t2["vVoo"])

	return fs2s1 

def fn_s2_s2(f,t2,inc_3_body=True):
	# [[Fn,S_2ext],S_2ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize
	fs2s2 = {
		"c": 0.0,
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"vv": np.zeros((n_virt_int,n_virt_int)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"ovov": np.zeros((n_occ,n_virt_int,n_occ,n_virt_int)),
		"ovvv": np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int)),
		"vvov": np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	}
	# Populate [[Fn,S_2ext],S_2ext]
	fs2s2["c"] += -2.000 * contract("ji,ikaA,aAjk->",f["oo"],t2["oovV"],t2["vVoo"])
	fs2s2["c"] += -1.000 * contract("ji,ikAB,ABjk->",f["oo"],t2["ooVV"],t2["VVoo"])
	fs2s2["c"] +=  1.000 * contract("ba,ijbA,aAij->",f["vv"],t2["oovV"],t2["vVoo"])
	fs2s2["c"] +=  1.000 * contract("Aa,ijAB,aBij->",f["Vv"],t2["ooVV"],t2["vVoo"])
	fs2s2["c"] +=  1.000 * contract("aA,ijaB,ABij->",f["vV"],t2["oovV"],t2["VVoo"])
	fs2s2["c"] +=  1.000 * contract("BA,ijaB,aAij->",f["VV"],t2["oovV"],t2["vVoo"])
	fs2s2["c"] +=  1.000 * contract("BA,ijBC,ACij->",f["VV"],t2["ooVV"],t2["VVoo"]) # V^3 o^2

	fs2s2["oo"] +=  2.000 * contract("ba,ikbA,aAjk->ij",f["vv"],t2["oovV"],t2["vVoo"])
	fs2s2["oo"] +=  2.000 * contract("Aa,ikAB,aBjk->ij",f["Vv"],t2["ooVV"],t2["vVoo"])
	fs2s2["oo"] +=  2.000 * contract("aA,ikaB,ABjk->ij",f["vV"],t2["oovV"],t2["VVoo"])
	fs2s2["oo"] +=  2.000 * contract("BA,ikaB,aAjk->ij",f["VV"],t2["oovV"],t2["vVoo"])
	fs2s2["oo"] +=  2.000 * contract("BA,ikBC,ACjk->ij",f["VV"],t2["ooVV"],t2["VVoo"]) # V^3 o^4
	fs2s2["oo"] += -2.000 * contract("lk,ikaA,aAjl->ij",f["oo"],t2["oovV"],t2["vVoo"])
	fs2s2["oo"] += -1.000 * contract("ki,jlaA,aAkl->ji",f["oo"],t2["oovV"],t2["vVoo"])
	fs2s2["oo"] += -1.000 * contract("ik,klaA,aAjl->ij",f["oo"],t2["oovV"],t2["vVoo"])
	fs2s2["oo"] += -1.000 * contract("lk,ikAB,ABjl->ij",f["oo"],t2["ooVV"],t2["VVoo"])
	fs2s2["oo"] += -0.500 * contract("ki,jlAB,ABkl->ji",f["oo"],t2["ooVV"],t2["VVoo"])
	fs2s2["oo"] += -0.500 * contract("ik,klAB,ABjl->ij",f["oo"],t2["ooVV"],t2["VVoo"])

	fs2s2["ov"] +=  1.000 * contract("jb,ikaA,bAjk->ia",f["ov"],t2["oovV"],t2["vVoo"])
	fs2s2["ov"] +=  1.000 * contract("jA,ikaB,ABjk->ia",f["oV"],t2["oovV"],t2["VVoo"])
	fs2s2["ov"] += -1.000 * contract("ja,ikbA,bAjk->ia",f["ov"],t2["oovV"],t2["vVoo"])
	fs2s2["ov"] += -0.500 * contract("ja,ikAB,ABjk->ia",f["ov"],t2["ooVV"],t2["VVoo"])
	fs2s2["ov"] += -0.500 * contract("ib,jkaA,bAjk->ia",f["ov"],t2["oovV"],t2["vVoo"])
	fs2s2["ov"] += -0.500 * contract("iA,jkaB,ABjk->ia",f["oV"],t2["oovV"],t2["VVoo"])

	fs2s2["vo"] +=  1.000 * contract("bj,jkbA,aAik->ai",f["vo"],t2["oovV"],t2["vVoo"])
	fs2s2["vo"] +=  1.000 * contract("Aj,jkAB,aBik->ai",f["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2["vo"] += -1.000 * contract("aj,jkbA,bAik->ai",f["vo"],t2["oovV"],t2["vVoo"])
	fs2s2["vo"] += -0.500 * contract("bi,jkbA,aAjk->ai",f["vo"],t2["oovV"],t2["vVoo"])
	fs2s2["vo"] += -0.500 * contract("Ai,jkAB,aBjk->ai",f["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2["vo"] += -0.500 * contract("aj,jkAB,ABik->ai",f["vo"],t2["ooVV"],t2["VVoo"])

	fs2s2["vv"] +=  2.000 * contract("ji,ikaA,bAjk->ba",f["oo"],t2["oovV"],t2["vVoo"])
	fs2s2["vv"] += -1.000 * contract("BA,ijaB,bAij->ba",f["VV"],t2["oovV"],t2["vVoo"]) # v^2 V^2 o^2
	fs2s2["vv"] += -0.500 * contract("ca,ijcA,bAij->ba",f["vv"],t2["oovV"],t2["vVoo"])
	fs2s2["vv"] += -0.500 * contract("Aa,ijAB,bBij->ba",f["Vv"],t2["ooVV"],t2["vVoo"])
	fs2s2["vv"] += -0.500 * contract("ac,ijbA,cAij->ab",f["vv"],t2["oovV"],t2["vVoo"])
	fs2s2["vv"] += -0.500 * contract("aA,ijbB,ABij->ab",f["vV"],t2["oovV"],t2["VVoo"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs2s2["oooo"] +=  0.500 * contract("mi,jkaA,aAlm->jkil",f["oo"],t2["oovV"],t2["vVoo"])
	fs2s2["oooo"] +=  0.500 * contract("im,jmaA,aAkl->ijkl",f["oo"],t2["oovV"],t2["vVoo"])
	fs2s2["oooo"] +=  0.500 * contract("ba,ijbA,aAkl->ijkl",f["vv"],t2["oovV"],t2["vVoo"])
	fs2s2["oooo"] +=  0.500 * contract("Aa,ijAB,aBkl->ijkl",f["Vv"],t2["ooVV"],t2["vVoo"])
	fs2s2["oooo"] +=  0.500 * contract("aA,ijaB,ABkl->ijkl",f["vV"],t2["oovV"],t2["VVoo"])
	fs2s2["oooo"] +=  0.500 * contract("BA,ijaB,aAkl->ijkl",f["VV"],t2["oovV"],t2["vVoo"])
	fs2s2["oooo"] +=  0.500 * contract("BA,ijBC,ACkl->ijkl",f["VV"],t2["ooVV"],t2["VVoo"])
	fs2s2["oooo"] +=  0.250 * contract("mi,jkAB,ABlm->jkil",f["oo"],t2["ooVV"],t2["VVoo"])
	fs2s2["oooo"] +=  0.250 * contract("im,jmAB,ABkl->ijkl",f["oo"],t2["ooVV"],t2["VVoo"])
	fs2s2["oooo"] = asym_term(fs2s2["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2s2["ooov"] +=  0.500 * contract("ib,jlaA,bAkl->ijka",f["ov"],t2["oovV"],t2["vVoo"])
	fs2s2["ooov"] +=  0.500 * contract("iA,jlaB,ABkl->ijka",f["oV"],t2["oovV"],t2["VVoo"])
	fs2s2["ooov"] += -0.250 * contract("la,ijbA,bAkl->ijka",f["ov"],t2["oovV"],t2["vVoo"])
	fs2s2["ooov"] +=  0.250 * contract("lb,ijaA,bAkl->ijka",f["ov"],t2["oovV"],t2["vVoo"])
	fs2s2["ooov"] +=  0.250 * contract("lA,ijaB,ABkl->ijka",f["oV"],t2["oovV"],t2["VVoo"])
	fs2s2["ooov"] += -0.125 * contract("la,ijAB,ABkl->ijka",f["ov"],t2["ooVV"],t2["VVoo"])
	fs2s2["ooov"] = asym_term(fs2s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2s2["ovoo"] +=  0.500 * contract("bi,jlbA,aAkl->jaik",f["vo"],t2["oovV"],t2["vVoo"])
	fs2s2["ovoo"] +=  0.500 * contract("Ai,jlAB,aBkl->jaik",f["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2["ovoo"] += -0.250 * contract("al,ilbA,bAjk->iajk",f["vo"],t2["oovV"],t2["vVoo"])
	fs2s2["ovoo"] +=  0.250 * contract("bl,ilbA,aAjk->iajk",f["vo"],t2["oovV"],t2["vVoo"])
	fs2s2["ovoo"] +=  0.250 * contract("Al,ilAB,aBjk->iajk",f["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2["ovoo"] += -0.125 * contract("al,ilAB,ABjk->iajk",f["vo"],t2["ooVV"],t2["VVoo"])
	fs2s2["ovoo"] = asym_term(fs2s2["ovoo"],"ovoo")

	fs2s2["ovov"] +=  0.500 * contract("lk,ikaA,bAjl->ibja",f["oo"],t2["oovV"],t2["vVoo"])
	fs2s2["ovov"] += -0.500 * contract("BA,ikaB,bAjk->ibja",f["VV"],t2["oovV"],t2["vVoo"])
	fs2s2["ovov"] +=  0.250 * contract("ki,jlaA,bAkl->jbia",f["oo"],t2["oovV"],t2["vVoo"])
	fs2s2["ovov"] +=  0.250 * contract("ik,klaA,bAjl->ibja",f["oo"],t2["oovV"],t2["vVoo"])
	fs2s2["ovov"] += -0.250 * contract("ca,ikcA,bAjk->ibja",f["vv"],t2["oovV"],t2["vVoo"])
	fs2s2["ovov"] += -0.250 * contract("Aa,ikAB,bBjk->ibja",f["Vv"],t2["ooVV"],t2["vVoo"])
	fs2s2["ovov"] += -0.250 * contract("ac,ikbA,cAjk->iajb",f["vv"],t2["oovV"],t2["vVoo"])
	fs2s2["ovov"] += -0.250 * contract("aA,ikbB,ABjk->iajb",f["vV"],t2["oovV"],t2["VVoo"])

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	fs2s2["ovvv"] +=  0.500 * contract("ja,ikbA,cAjk->icab",f["ov"],t2["oovV"],t2["vVoo"])
	fs2s2["ovvv"] = asym_term(fs2s2["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	fs2s2["vvov"] +=  0.500 * contract("aj,jkbA,cAik->acib",f["vo"],t2["oovV"],t2["vVoo"])
	fs2s2["vvov"] = asym_term(fs2s2["vvov"],"vvov")

	if(inc_3_body):
		fs2s2["ooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2["oovooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2["oovoov"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2["oovovv"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs2s2["ovvoov"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2["ooooov"] += -(1./12.) * contract("ib,jkaA,bAlm->ijklma",f["ov"],t2["oovV"],t2["vVoo"])
		fs2s2["ooooov"] += -(1./12.) * contract("iA,jkaB,ABlm->ijklma",f["oV"],t2["oovV"],t2["VVoo"])
		fs2s2["ooooov"] = asym_term(fs2s2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2["oovooo"] += -(1./12.) * contract("bi,jkbA,aAlm->jkailm",f["vo"],t2["oovV"],t2["vVoo"])
		fs2s2["oovooo"] += -(1./12.) * contract("Ai,jkAB,aBlm->jkailm",f["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s2["oovooo"] = asym_term(fs2s2["oovooo"],"oovooo")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2["oovoov"] += -(1./18.) * contract("BA,ijaB,bAkl->ijbkla",f["VV"],t2["oovV"],t2["vVoo"])
		fs2s2["oovoov"] += -(1./18.) * contract("mi,jkaA,bAlm->jkbila",f["oo"],t2["oovV"],t2["vVoo"])
		fs2s2["oovoov"] += -(1./18.) * contract("im,jmaA,bAkl->ijbkla",f["oo"],t2["oovV"],t2["vVoo"])
		fs2s2["oovoov"] += -(1./36.) * contract("ca,ijcA,bAkl->ijbkla",f["vv"],t2["oovV"],t2["vVoo"])
		fs2s2["oovoov"] += -(1./36.) * contract("Aa,ijAB,bBkl->ijbkla",f["Vv"],t2["ooVV"],t2["vVoo"])
		fs2s2["oovoov"] += -(1./36.) * contract("ac,ijbA,cAkl->ijaklb",f["vv"],t2["oovV"],t2["vVoo"])
		fs2s2["oovoov"] += -(1./36.) * contract("aA,ijbB,ABkl->ijaklb",f["vV"],t2["oovV"],t2["VVoo"])
		fs2s2["oovoov"] = asym_term(fs2s2["oovoov"],"oovoov")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs2s2["oovovv"] +=  (1./18.) * contract("la,ijbA,cAkl->ijckab",f["ov"],t2["oovV"],t2["vVoo"])
		fs2s2["oovovv"] = asym_term(fs2s2["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2["ovvoov"] +=  (1./18.) * contract("al,ilbA,cAjk->iacjkb",f["vo"],t2["oovV"],t2["vVoo"])
		fs2s2["ovvoov"] = asym_term(fs2s2["ovvoov"],"ovvoov")	

	return fs2s2 

def wn_s1_s1(v,t1):
	# [[Wn,S_1ext],S_1ext]
	# for sizing arrays
	n_occ = v["oooo"].shape[0]
	n_virt_int = v["vvvv"].shape[0]
	# initialize
	ws1s1 = {
		"c": 0.0,
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"vv": np.zeros((n_virt_int,n_virt_int)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"ovov": np.zeros((n_occ,n_virt_int,n_occ,n_virt_int)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ)),
		"ovvv": np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int)),
		"vvov": np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	}
	# Populate [[Wn,S_1ext],S_1ext]
	ws1s1["c"] +=  4.000 * contract("Ai,Bj,ijAB->",t1["Vo"],t1["Vo"],v["ooVV"])
	ws1s1["c"] += -8.000 * contract("iA,Bj,jAiB->",t1["oV"],t1["Vo"],v["oVoV"])
	ws1s1["c"] +=  4.000 * contract("iA,jB,ABij->",t1["oV"],t1["oV"],v["VVoo"])

	ws1s1["oo"] +=  8.000 * contract("Ai,Bk,jkAB->ji",t1["Vo"],t1["Vo"],v["ooVV"])
	ws1s1["oo"] += -8.000 * contract("iA,Bk,kAjB->ij",t1["oV"],t1["Vo"],v["oVoV"])
	ws1s1["oo"] +=  8.000 * contract("iA,kB,ABjk->ij",t1["oV"],t1["oV"],v["VVoo"])
	ws1s1["oo"] += -8.000 * contract("kA,Bi,jAkB->ji",t1["oV"],t1["Vo"],v["oVoV"])
	ws1s1["oo"] +=  8.000 * contract("kA,Bk,jAiB->ji",t1["oV"],t1["Vo"],v["oVoV"])
	ws1s1["oo"] += -8.000 * contract("kA,Al,jlik->ji",t1["oV"],t1["Vo"],v["oooo"])

	ws1s1["ov"] += -8.000 * contract("iA,Bj,jAaB->ia",t1["oV"],t1["Vo"],v["oVvV"])
	ws1s1["ov"] += -8.000 * contract("iA,jB,ABja->ia",t1["oV"],t1["oV"],v["VVov"])
	ws1s1["ov"] +=  8.000 * contract("jA,Bj,iAaB->ia",t1["oV"],t1["Vo"],v["oVvV"])
	ws1s1["ov"] +=  8.000 * contract("jA,Ak,ikja->ia",t1["oV"],t1["Vo"],v["ooov"])

	ws1s1["vo"] += -8.000 * contract("Ai,Bj,jaAB->ai",t1["Vo"],t1["Vo"],v["ovVV"])
	ws1s1["vo"] += -8.000 * contract("jA,Bi,aAjB->ai",t1["oV"],t1["Vo"],v["vVoV"])
	ws1s1["vo"] +=  8.000 * contract("jA,Bj,aAiB->ai",t1["oV"],t1["Vo"],v["vVoV"])
	ws1s1["vo"] +=  8.000 * contract("jA,Ak,kaij->ai",t1["oV"],t1["Vo"],v["ovoo"])

	ws1s1["vv"] +=  8.000 * contract("iA,Bi,bAaB->ba",t1["oV"],t1["Vo"],v["vVvV"])
	ws1s1["vv"] += -8.000 * contract("iA,Aj,jbia->ba",t1["oV"],t1["Vo"],v["ovov"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	ws1s1["oooo"] +=  8.000 * contract("iA,Bj,lAkB->iljk",t1["oV"],t1["Vo"],v["oVoV"])
	ws1s1["oooo"] +=  2.000 * contract("Ai,Bj,klAB->klij",t1["Vo"],t1["Vo"],v["ooVV"])
	ws1s1["oooo"] +=  2.000 * contract("iA,Am,lmjk->iljk",t1["oV"],t1["Vo"],v["oooo"])
	ws1s1["oooo"] +=  2.000 * contract("iA,jB,ABkl->ijkl",t1["oV"],t1["oV"],v["VVoo"])
	ws1s1["oooo"] +=  2.000 * contract("mA,Ai,kljm->klij",t1["oV"],t1["Vo"],v["oooo"])
	ws1s1["oooo"] = asym_term(ws1s1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	ws1s1["ooov"] +=  4.000 * contract("iA,Bj,kAaB->ikja",t1["oV"],t1["Vo"],v["oVvV"])
	ws1s1["ooov"] +=  2.000 * contract("iA,Al,klja->ikja",t1["oV"],t1["Vo"],v["ooov"])
	ws1s1["ooov"] +=  2.000 * contract("iA,jB,ABka->ijka",t1["oV"],t1["oV"],v["VVov"])
	ws1s1["ooov"] += -1.000 * contract("lA,Ai,jkla->jkia",t1["oV"],t1["Vo"],v["ooov"])
	ws1s1["ooov"] = asym_term(ws1s1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	ws1s1["ovoo"] +=  4.000 * contract("iA,Bj,aAkB->iajk",t1["oV"],t1["Vo"],v["vVoV"])
	ws1s1["ovoo"] +=  2.000 * contract("lA,Ai,kajl->kaij",t1["oV"],t1["Vo"],v["ovoo"])
	ws1s1["ovoo"] +=  2.000 * contract("Ai,Bj,kaAB->kaij",t1["Vo"],t1["Vo"],v["ovVV"])
	ws1s1["ovoo"] += -1.000 * contract("iA,Al,lajk->iajk",t1["oV"],t1["Vo"],v["ovoo"])
	ws1s1["ovoo"] = asym_term(ws1s1["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	ws1s1["oovv"] +=  2.000 * contract("iA,Ak,jkab->ijab",t1["oV"],t1["Vo"],v["oovv"])
	ws1s1["oovv"] +=  2.000 * contract("iA,jB,ABab->ijab",t1["oV"],t1["oV"],v["VVvv"])
	ws1s1["oovv"] = asym_term(ws1s1["oovv"],"oovv")

	ws1s1["ovov"] +=  2.000 * contract("iA,Bj,bAaB->ibja",t1["oV"],t1["Vo"],v["vVvV"])
	ws1s1["ovov"] += -1.000 * contract("iA,Ak,kbja->ibja",t1["oV"],t1["Vo"],v["ovov"])
	ws1s1["ovov"] += -1.000 * contract("kA,Ai,jbka->jbia",t1["oV"],t1["Vo"],v["ovov"])

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	ws1s1["vvoo"] += 2.000 * contract("Ai,Bj,abAB->abij",t1["Vo"],t1["Vo"],v["vvVV"])
	ws1s1["vvoo"] += 2.000 * contract("kA,Ai,abjk->abij",t1["oV"],t1["Vo"],v["vvoo"])
	ws1s1["vvoo"] = asym_term(ws1s1["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	ws1s1["ovvv"] += -1.000 * contract("iA,Aj,jcab->icab",t1["oV"],t1["Vo"],v["ovvv"])
	ws1s1["ovvv"] = asym_term(ws1s1["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	ws1s1["vvov"] += -1.000 * contract("jA,Ai,bcja->bcia",t1["oV"],t1["Vo"],v["vvov"])
	ws1s1["vvov"] = asym_term(ws1s1["vvov"],"vvov")

	return ws1s1 

def wn_s1_s2(v,t1,t2,inc_3_body=True):
	# [[Wn,S_1ext],S_2ext]
	# for sizing arrays
	n_occ = v["oooo"].shape[0]
	n_virt_int = v["vvvv"].shape[0]
	# initialize
	ws1s2 = {
		"c": 0.0,
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"vv": np.zeros((n_virt_int,n_virt_int)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"ovov": np.zeros((n_occ,n_virt_int,n_occ,n_virt_int)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ)),
		"ovvv": np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int)),
		"vvov": np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	}
	# Populate [[Wn,S_1ext],S_2ext]
	ws1s2["c"] +=  2.000 * contract("Ai,jkaA,iajk->",t1["Vo"],t2["oovV"],v["ovoo"])
	ws1s2["c"] += -4.000 * contract("Ai,ijaB,aBjA->",t1["Vo"],t2["oovV"],v["vVoV"])
	ws1s2["c"] += -2.000 * contract("Ai,jkAB,iBjk->",t1["Vo"],t2["ooVV"],v["oVoo"])
	ws1s2["c"] += -2.000 * contract("Ai,ijBC,BCjA->",t1["Vo"],t2["ooVV"],v["VVoV"])
	ws1s2["c"] += -4.000 * contract("iA,aBij,jAaB->",t1["oV"],t2["vVoo"],v["oVvV"])
	ws1s2["c"] += -2.000 * contract("iA,BCij,jABC->",t1["oV"],t2["VVoo"],v["oVVV"])
	ws1s2["c"] +=  2.000 * contract("iA,aAjk,jkia->",t1["oV"],t2["vVoo"],v["ooov"])
	ws1s2["c"] += -2.000 * contract("iA,ABjk,jkiB->",t1["oV"],t2["VVoo"],v["oooV"])

	ws1s2["oo"] += -4.000 * contract("Ai,jkaB,aBkA->ji",t1["Vo"],t2["oovV"],v["vVoV"])
	ws1s2["oo"] += -2.000 * contract("Ai,jkBC,BCkA->ji",t1["Vo"],t2["ooVV"],v["VVoV"])
	ws1s2["oo"] +=  4.000 * contract("Ak,ilaA,kajl->ij",t1["Vo"],t2["oovV"],v["ovoo"])
	ws1s2["oo"] +=  4.000 * contract("Ak,ikaB,aBjA->ij",t1["Vo"],t2["oovV"],v["vVoV"])
	ws1s2["oo"] += -4.000 * contract("Ak,ilAB,kBjl->ij",t1["Vo"],t2["ooVV"],v["oVoo"])
	ws1s2["oo"] +=  2.000 * contract("Ak,ikBC,BCjA->ij",t1["Vo"],t2["ooVV"],v["VVoV"])
	ws1s2["oo"] += -4.000 * contract("iA,aBjk,kAaB->ij",t1["oV"],t2["vVoo"],v["oVvV"])
	ws1s2["oo"] += -2.000 * contract("iA,BCjk,kABC->ij",t1["oV"],t2["VVoo"],v["oVVV"])
	ws1s2["oo"] +=  4.000 * contract("kA,aBik,jAaB->ji",t1["oV"],t2["vVoo"],v["oVvV"])
	ws1s2["oo"] +=  2.000 * contract("kA,BCik,jABC->ji",t1["oV"],t2["VVoo"],v["oVVV"])
	ws1s2["oo"] +=  4.000 * contract("kA,aAil,jlka->ji",t1["oV"],t2["vVoo"],v["ooov"])
	ws1s2["oo"] += -4.000 * contract("kA,ABil,jlkB->ji",t1["oV"],t2["VVoo"],v["oooV"])

	ws1s2["ov"] +=  2.000 * contract("Aj,klaA,ijkl->ia",t1["Vo"],t2["oovV"],v["oooo"])
	ws1s2["ov"] += -4.000 * contract("Aj,ikaB,jBkA->ia",t1["Vo"],t2["oovV"],v["oVoV"])
	ws1s2["ov"] +=  4.000 * contract("Aj,jkaB,iBkA->ia",t1["Vo"],t2["oovV"],v["oVoV"])
	ws1s2["ov"] += -4.000 * contract("Aj,ikbA,jbka->ia",t1["Vo"],t2["oovV"],v["ovov"])
	ws1s2["ov"] +=  4.000 * contract("Aj,ijbB,bBaA->ia",t1["Vo"],t2["oovV"],v["vVvV"])
	ws1s2["ov"] +=  4.000 * contract("Aj,ikAB,jBka->ia",t1["Vo"],t2["ooVV"],v["oVov"])
	ws1s2["ov"] +=  2.000 * contract("Aj,ijBC,BCaA->ia",t1["Vo"],t2["ooVV"],v["VVvV"])
	ws1s2["ov"] += -2.000 * contract("iA,jkaB,ABjk->ia",t1["oV"],t2["oovV"],v["VVoo"])
	ws1s2["ov"] +=  4.000 * contract("jA,ikaB,ABjk->ia",t1["oV"],t2["oovV"],v["VVoo"])

	ws1s2["vo"] += -2.000 * contract("Ai,aBjk,jkAB->ai",t1["Vo"],t2["vVoo"],v["ooVV"])
	ws1s2["vo"] +=  4.000 * contract("Aj,aBik,jkAB->ai",t1["Vo"],t2["vVoo"],v["ooVV"])
	ws1s2["vo"] +=  4.000 * contract("jA,bBij,aAbB->ai",t1["oV"],t2["vVoo"],v["vVvV"])
	ws1s2["vo"] +=  2.000 * contract("jA,BCij,aABC->ai",t1["oV"],t2["VVoo"],v["vVVV"])
	ws1s2["vo"] += -4.000 * contract("jA,aBik,kAjB->ai",t1["oV"],t2["vVoo"],v["oVoV"])
	ws1s2["vo"] += -4.000 * contract("jA,bAik,kajb->ai",t1["oV"],t2["vVoo"],v["ovov"])
	ws1s2["vo"] +=  4.000 * contract("jA,ABik,kajB->ai",t1["oV"],t2["VVoo"],v["ovoV"])
	ws1s2["vo"] +=  4.000 * contract("jA,aBjk,kAiB->ai",t1["oV"],t2["vVoo"],v["oVoV"])
	ws1s2["vo"] +=  2.000 * contract("jA,aAkl,klij->ai",t1["oV"],t2["vVoo"],v["oooo"])

	ws1s2["vv"] += -2.000 * contract("Ai,jkaA,ibjk->ba",t1["Vo"],t2["oovV"],v["ovoo"])
	ws1s2["vv"] +=  4.000 * contract("Ai,ijaB,bBjA->ba",t1["Vo"],t2["oovV"],v["vVoV"])
	ws1s2["vv"] +=  4.000 * contract("iA,aBij,jAbB->ab",t1["oV"],t2["vVoo"],v["oVvV"])
	ws1s2["vv"] += -2.000 * contract("iA,aAjk,jkib->ab",t1["oV"],t2["vVoo"],v["ooov"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	ws1s2["oooo"] += -2.000 * contract("Ai,jkaB,aBlA->jkil",t1["Vo"],t2["oovV"],v["vVoV"])
	ws1s2["oooo"] += -2.000 * contract("iA,aBjk,lAaB->iljk",t1["oV"],t2["vVoo"],v["oVvV"])
	ws1s2["oooo"] += -1.000 * contract("Ai,jkBC,BClA->jkil",t1["Vo"],t2["ooVV"],v["VVoV"])
	ws1s2["oooo"] +=  1.000 * contract("Am,ijaA,makl->ijkl",t1["Vo"],t2["oovV"],v["ovoo"])
	ws1s2["oooo"] += -1.000 * contract("Am,ijAB,mBkl->ijkl",t1["Vo"],t2["ooVV"],v["oVoo"])
	ws1s2["oooo"] += -1.000 * contract("iA,BCjk,lABC->iljk",t1["oV"],t2["VVoo"],v["oVVV"])
	ws1s2["oooo"] +=  1.000 * contract("mA,aAij,klma->klij",t1["oV"],t2["vVoo"],v["ooov"])
	ws1s2["oooo"] += -1.000 * contract("mA,ABij,klmB->klij",t1["oV"],t2["VVoo"],v["oooV"])
	ws1s2["oooo"] = asym_term(ws1s2["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	ws1s2["ooov"] +=  2.000 * contract("Ai,jlaB,kBlA->jkia",t1["Vo"],t2["oovV"],v["oVoV"])
	ws1s2["ooov"] +=  2.000 * contract("Al,imaA,kljm->ikja",t1["Vo"],t2["oovV"],v["oooo"])
	ws1s2["ooov"] += -2.000 * contract("Al,ilaB,kBjA->ikja",t1["Vo"],t2["oovV"],v["oVoV"])
	ws1s2["ooov"] +=  2.000 * contract("iA,jlaB,ABkl->ijka",t1["oV"],t2["oovV"],v["VVoo"])
	ws1s2["ooov"] += -1.000 * contract("Ai,jkbB,bBaA->jkia",t1["Vo"],t2["oovV"],v["vVvV"])
	ws1s2["ooov"] +=  1.000 * contract("Al,ijaB,lBkA->ijka",t1["Vo"],t2["oovV"],v["oVoV"])
	ws1s2["ooov"] +=  1.000 * contract("Al,ijbA,lbka->ijka",t1["Vo"],t2["oovV"],v["ovov"])
	ws1s2["ooov"] += -1.000 * contract("Al,ijAB,lBka->ijka",t1["Vo"],t2["ooVV"],v["oVov"])
	ws1s2["ooov"] +=  1.000 * contract("lA,ijaB,ABkl->ijka",t1["oV"],t2["oovV"],v["VVoo"])
	ws1s2["ooov"] += -0.500 * contract("Ai,jkBC,BCaA->jkia",t1["Vo"],t2["ooVV"],v["VVvV"])
	ws1s2["ooov"] = asym_term(ws1s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	ws1s2["ovoo"] +=  2.000 * contract("Ai,aBjl,klAB->kaij",t1["Vo"],t2["vVoo"],v["ooVV"])
	ws1s2["ovoo"] +=  2.000 * contract("iA,aBjl,lAkB->iajk",t1["oV"],t2["vVoo"],v["oVoV"])
	ws1s2["ovoo"] += -2.000 * contract("lA,aBil,kAjB->kaij",t1["oV"],t2["vVoo"],v["oVoV"])
	ws1s2["ovoo"] +=  2.000 * contract("lA,aAim,kmjl->kaij",t1["oV"],t2["vVoo"],v["oooo"])
	ws1s2["ovoo"] +=  1.000 * contract("Al,aBij,klAB->kaij",t1["Vo"],t2["vVoo"],v["ooVV"])
	ws1s2["ovoo"] += -1.000 * contract("iA,bBjk,aAbB->iajk",t1["oV"],t2["vVoo"],v["vVvV"])
	ws1s2["ovoo"] +=  1.000 * contract("lA,aBij,kAlB->kaij",t1["oV"],t2["vVoo"],v["oVoV"])
	ws1s2["ovoo"] +=  1.000 * contract("lA,bAij,kalb->kaij",t1["oV"],t2["vVoo"],v["ovov"])
	ws1s2["ovoo"] += -1.000 * contract("lA,ABij,kalB->kaij",t1["oV"],t2["VVoo"],v["ovoV"])
	ws1s2["ovoo"] += -0.500 * contract("iA,BCjk,aABC->iajk",t1["oV"],t2["VVoo"],v["vVVV"])
	ws1s2["ovoo"] = asym_term(ws1s2["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	ws1s2["oovv"] +=  4.000 * contract("Ak,ilaA,jklb->ijab",t1["Vo"],t2["oovV"],v["ooov"])
	ws1s2["oovv"] +=  4.000 * contract("Ak,ikaB,jBbA->ijab",t1["Vo"],t2["oovV"],v["oVvV"])
	ws1s2["oovv"] +=  4.000 * contract("iA,jkaB,ABkb->ijab",t1["oV"],t2["oovV"],v["VVov"])
	ws1s2["oovv"] +=  2.000 * contract("kA,ijaB,ABkb->ijab",t1["oV"],t2["oovV"],v["VVov"])
	ws1s2["oovv"] += -2.000 * contract("Ak,ijaB,kBbA->ijab",t1["Vo"],t2["oovV"],v["oVvV"])
	ws1s2["oovv"] +=  1.000 * contract("Ak,ijcA,kcab->ijab",t1["Vo"],t2["oovV"],v["ovvv"])
	ws1s2["oovv"] += -1.000 * contract("Ak,ijAB,kBab->ijab",t1["Vo"],t2["ooVV"],v["oVvv"])
	ws1s2["oovv"] = asym_term(ws1s2["oovv"],"oovv")

	ws1s2["ovov"] +=  1.000 * contract("Ai,jkaB,bBkA->jbia",t1["Vo"],t2["oovV"],v["vVoV"])
	ws1s2["ovov"] += -1.000 * contract("Ak,ilaA,kbjl->ibja",t1["Vo"],t2["oovV"],v["ovoo"])
	ws1s2["ovov"] += -1.000 * contract("Ak,ikaB,bBjA->ibja",t1["Vo"],t2["oovV"],v["vVoV"])
	ws1s2["ovov"] +=  1.000 * contract("iA,aBjk,kAbB->iajb",t1["oV"],t2["vVoo"],v["oVvV"])
	ws1s2["ovov"] += -1.000 * contract("kA,aBik,jAbB->jaib",t1["oV"],t2["vVoo"],v["oVvV"])
	ws1s2["ovov"] += -1.000 * contract("kA,aAil,jlkb->jaib",t1["oV"],t2["vVoo"],v["ooov"])

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	ws1s2["vvoo"] +=  4.000 * contract("Ai,aBjk,kbAB->abij",t1["Vo"],t2["vVoo"],v["ovVV"])
	ws1s2["vvoo"] +=  4.000 * contract("kA,aBik,bAjB->abij",t1["oV"],t2["vVoo"],v["vVoV"])
	ws1s2["vvoo"] +=  4.000 * contract("kA,aAil,lbjk->abij",t1["oV"],t2["vVoo"],v["ovoo"])
	ws1s2["vvoo"] +=  2.000 * contract("Ak,aBij,kbAB->abij",t1["Vo"],t2["vVoo"],v["ovVV"])
	ws1s2["vvoo"] += -2.000 * contract("kA,aBij,bAkB->abij",t1["oV"],t2["vVoo"],v["vVoV"])
	ws1s2["vvoo"] +=  1.000 * contract("kA,cAij,abkc->abij",t1["oV"],t2["vVoo"],v["vvov"])
	ws1s2["vvoo"] += -1.000 * contract("kA,ABij,abkB->abij",t1["oV"],t2["VVoo"],v["vvoV"])
	ws1s2["vvoo"] = asym_term(ws1s2["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	ws1s2["ovvv"] += -2.000 * contract("Aj,ikaA,jckb->icab",t1["Vo"],t2["oovV"],v["ovov"])
	ws1s2["ovvv"] +=  2.000 * contract("Aj,ijaB,cBbA->icab",t1["Vo"],t2["oovV"],v["vVvV"])
	ws1s2["ovvv"] = asym_term(ws1s2["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	ws1s2["vvov"] +=  2.000 * contract("jA,aBij,cAbB->acib",t1["oV"],t2["vVoo"],v["vVvV"])
	ws1s2["vvov"] += -2.000 * contract("jA,aAik,kcjb->acib",t1["oV"],t2["vVoo"],v["ovov"])
	ws1s2["vvov"] = asym_term(ws1s2["vvov"],"vvov")

	if(inc_3_body):
		ws1s2["ooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		ws1s2["oovooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		ws1s2["oooovv"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws1s2["oovoov"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		ws1s2["ovvooo"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws1s2["ooovvv"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws1s2["oovovv"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws1s2["ovvoov"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws1s2["vvvooo"] = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws1s2["oovvvv"] = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws1s2["vvvoov"] = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		ws1s2["ooooov"] +=  (2./3.) * contract("Ai,jkaB,mBlA->jkmila",t1["Vo"],t2["oovV"],v["oVoV"])
		ws1s2["ooooov"] +=  (1./3.) * contract("An,ijaA,mnkl->ijmkla",t1["Vo"],t2["oovV"],v["oooo"])
		ws1s2["ooooov"] += -(1./3.) * contract("iA,jkaB,ABlm->ijklma",t1["oV"],t2["oovV"],v["VVoo"])
		ws1s2["ooooov"] = asym_term(ws1s2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		ws1s2["oovooo"] +=  (2./3.) * contract("iA,aBjk,mAlB->imajkl",t1["oV"],t2["vVoo"],v["oVoV"])
		ws1s2["oovooo"] += -(1./3.) * contract("Ai,aBjk,lmAB->lmaijk",t1["Vo"],t2["vVoo"],v["ooVV"])
		ws1s2["oovooo"] +=  (1./3.) * contract("nA,aAij,lmkn->lmaijk",t1["oV"],t2["vVoo"],v["oooo"])
		ws1s2["oovooo"] = asym_term(ws1s2["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws1s2["oooovv"] += -(2./3.) * contract("Ai,jkaB,lBbA->jkliab",t1["Vo"],t2["oovV"],v["oVvV"])
		ws1s2["oooovv"] += -(2./3.) * contract("Am,ijaA,lmkb->ijlkab",t1["Vo"],t2["oovV"],v["ooov"])
		ws1s2["oooovv"] +=  (2./3.) * contract("iA,jkaB,ABlb->ijklab",t1["oV"],t2["oovV"],v["VVov"])
		ws1s2["oooovv"] = asym_term(ws1s2["oooovv"],"oooovv")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		ws1s2["oovoov"] +=  (2./9.) * contract("Ai,jkaB,bBlA->jkbila",t1["Vo"],t2["oovV"],v["vVoV"])
		ws1s2["oovoov"] +=  (2./9.) * contract("iA,aBjk,lAbB->ilajkb",t1["oV"],t2["vVoo"],v["oVvV"])
		ws1s2["oovoov"] += -(1./9.) * contract("Am,ijaA,mbkl->ijbkla",t1["Vo"],t2["oovV"],v["ovoo"])
		ws1s2["oovoov"] += -(1./9.) * contract("mA,aAij,klmb->klaijb",t1["oV"],t2["vVoo"],v["ooov"])
		ws1s2["oovoov"] = asym_term(ws1s2["oovoov"],"oovoov")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws1s2["ovvooo"] +=  (2./3.) * contract("Ai,aBjk,lbAB->labijk",t1["Vo"],t2["vVoo"],v["ovVV"])
		ws1s2["ovvooo"] += -(2./3.) * contract("iA,aBjk,bAlB->iabjkl",t1["oV"],t2["vVoo"],v["vVoV"])
		ws1s2["ovvooo"] += -(2./3.) * contract("mA,aAij,lbkm->labijk",t1["oV"],t2["vVoo"],v["ovoo"])
		ws1s2["ovvooo"] = asym_term(ws1s2["ovvooo"],"ovvooo")

		# ooovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws1s2["ooovvv"] +=  1.000 * contract("Al,ijaA,klbc->ijkabc",t1["Vo"],t2["oovV"],v["oovv"])
		ws1s2["ooovvv"] += -1.000 * contract("iA,jkaB,ABbc->ijkabc",t1["oV"],t2["oovV"],v["VVvv"])
		ws1s2["ooovvv"] = asym_term(ws1s2["ooovvv"],"ooovvv")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws1s2["oovovv"] += -(2./9.) * contract("Ai,jkaB,cBbA->jkciab",t1["Vo"],t2["oovV"],v["vVvV"])
		ws1s2["oovovv"] +=  (2./9.) * contract("Al,ijaA,lckb->ijckab",t1["Vo"],t2["oovV"],v["ovov"])
		ws1s2["oovovv"] = asym_term(ws1s2["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws1s2["ovvoov"] += -(2./9.) * contract("iA,aBjk,cAbB->iacjkb",t1["oV"],t2["vVoo"],v["vVvV"])
		ws1s2["ovvoov"] +=  (2./9.) * contract("lA,aAij,kclb->kacijb",t1["oV"],t2["vVoo"],v["ovov"])
		ws1s2["ovvoov"] = asym_term(ws1s2["ovvoov"],"ovvoov")

		# vvvooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws1s2["vvvooo"] += -1.000 * contract("Ai,aBjk,bcAB->abcijk",t1["Vo"],t2["vVoo"],v["vvVV"])
		ws1s2["vvvooo"] +=  1.000 * contract("lA,aAij,bckl->abcijk",t1["oV"],t2["vVoo"],v["vvoo"])
		ws1s2["vvvooo"] = asym_term(ws1s2["vvvooo"],"vvvooo")

		# oovvvv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws1s2["oovvvv"] += -(1./3.) * contract("Ak,ijaA,kdbc->ijdabc",t1["Vo"],t2["oovV"],v["ovvv"])
		ws1s2["oovvvv"] = asym_term(ws1s2["oovvvv"],"oovvvv")

		# vvvoov = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws1s2["vvvoov"] += -(1./3.) * contract("kA,aAij,cdkb->acdijb",t1["oV"],t2["vVoo"],v["vvov"])
		ws1s2["vvvoov"] = asym_term(ws1s2["vvvoov"],"vvvoov")

	return ws1s2 

def wn_s2_s1(v,t1,t2,inc_3_body=True):
	# [[Wn,S_2ext],S_1ext]
	# for sizing arrays
	n_occ = v["oooo"].shape[0]
	n_virt_int = v["vvvv"].shape[0]
	# initialize
	ws2s1 = {
		"c": 0.0,
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"vv": np.zeros((n_virt_int,n_virt_int)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"ovov": np.zeros((n_occ,n_virt_int,n_occ,n_virt_int)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ)),
		"ovvv": np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int)),
		"vvov": np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int)),
		"vvvv": np.zeros((n_virt_int,n_virt_int,n_virt_int,n_virt_int))
	}
	# Populate [[Wn,S_2ext],S_1ext]
	ws2s1["c"] +=  2.000 * contract("Ai,jkaA,iajk->",t1["Vo"],t2["oovV"],v["ovoo"])
	ws2s1["c"] += -4.000 * contract("Ai,ijaB,aBjA->",t1["Vo"],t2["oovV"],v["vVoV"])
	ws2s1["c"] += -2.000 * contract("Ai,jkAB,iBjk->",t1["Vo"],t2["ooVV"],v["oVoo"])
	ws2s1["c"] += -2.000 * contract("Ai,ijBC,BCjA->",t1["Vo"],t2["ooVV"],v["VVoV"])
	ws2s1["c"] += -4.000 * contract("iA,aBij,jAaB->",t1["oV"],t2["vVoo"],v["oVvV"])
	ws2s1["c"] += -2.000 * contract("iA,BCij,jABC->",t1["oV"],t2["VVoo"],v["oVVV"])
	ws2s1["c"] +=  2.000 * contract("iA,aAjk,jkia->",t1["oV"],t2["vVoo"],v["ooov"])
	ws2s1["c"] += -2.000 * contract("iA,ABjk,jkiB->",t1["oV"],t2["VVoo"],v["oooV"])

	ws2s1["oo"] +=  2.000 * contract("Ai,klaA,jakl->ji",t1["Vo"],t2["oovV"],v["ovoo"])
	ws2s1["oo"] += -4.000 * contract("Ai,jkaB,aBkA->ji",t1["Vo"],t2["oovV"],v["vVoV"])
	ws2s1["oo"] += -2.000 * contract("Ai,klAB,jBkl->ji",t1["Vo"],t2["ooVV"],v["oVoo"])
	ws2s1["oo"] += -2.000 * contract("Ai,jkBC,BCkA->ji",t1["Vo"],t2["ooVV"],v["VVoV"])
	ws2s1["oo"] +=  4.000 * contract("Ak,ilaA,kajl->ij",t1["Vo"],t2["oovV"],v["ovoo"])
	ws2s1["oo"] += -4.000 * contract("Ak,klaA,jail->ji",t1["Vo"],t2["oovV"],v["ovoo"])
	ws2s1["oo"] +=  4.000 * contract("Ak,ikaB,aBjA->ij",t1["Vo"],t2["oovV"],v["vVoV"])
	ws2s1["oo"] += -4.000 * contract("Ak,ilAB,kBjl->ij",t1["Vo"],t2["ooVV"],v["oVoo"])
	ws2s1["oo"] +=  4.000 * contract("Ak,klAB,jBil->ji",t1["Vo"],t2["ooVV"],v["oVoo"])
	ws2s1["oo"] +=  2.000 * contract("Ak,ikBC,BCjA->ij",t1["Vo"],t2["ooVV"],v["VVoV"])
	ws2s1["oo"] += -4.000 * contract("iA,aBjk,kAaB->ij",t1["oV"],t2["vVoo"],v["oVvV"])
	ws2s1["oo"] += -2.000 * contract("iA,BCjk,kABC->ij",t1["oV"],t2["VVoo"],v["oVVV"])
	ws2s1["oo"] +=  2.000 * contract("iA,aAkl,klja->ij",t1["oV"],t2["vVoo"],v["ooov"])
	ws2s1["oo"] += -2.000 * contract("iA,ABkl,kljB->ij",t1["oV"],t2["VVoo"],v["oooV"])
	ws2s1["oo"] +=  4.000 * contract("kA,aBik,jAaB->ji",t1["oV"],t2["vVoo"],v["oVvV"])
	ws2s1["oo"] +=  2.000 * contract("kA,BCik,jABC->ji",t1["oV"],t2["VVoo"],v["oVVV"])
	ws2s1["oo"] +=  4.000 * contract("kA,aAil,jlka->ji",t1["oV"],t2["vVoo"],v["ooov"])
	ws2s1["oo"] += -4.000 * contract("kA,ABil,jlkB->ji",t1["oV"],t2["VVoo"],v["oooV"])
	ws2s1["oo"] += -4.000 * contract("kA,aAkl,jlia->ji",t1["oV"],t2["vVoo"],v["ooov"])
	ws2s1["oo"] +=  4.000 * contract("kA,ABkl,jliB->ji",t1["oV"],t2["VVoo"],v["oooV"])

	ws2s1["ov"] +=  2.000 * contract("Aj,klaA,ijkl->ia",t1["Vo"],t2["oovV"],v["oooo"])
	ws2s1["ov"] += -4.000 * contract("Aj,ikaB,jBkA->ia",t1["Vo"],t2["oovV"],v["oVoV"])
	ws2s1["ov"] +=  4.000 * contract("Aj,jkaB,iBkA->ia",t1["Vo"],t2["oovV"],v["oVoV"])
	ws2s1["ov"] += -4.000 * contract("Aj,ikbA,jbka->ia",t1["Vo"],t2["oovV"],v["ovov"])
	ws2s1["ov"] +=  4.000 * contract("Aj,jkbA,ibka->ia",t1["Vo"],t2["oovV"],v["ovov"])
	ws2s1["ov"] +=  4.000 * contract("Aj,ijbB,bBaA->ia",t1["Vo"],t2["oovV"],v["vVvV"])
	ws2s1["ov"] +=  4.000 * contract("Aj,ikAB,jBka->ia",t1["Vo"],t2["ooVV"],v["oVov"])
	ws2s1["ov"] += -4.000 * contract("Aj,jkAB,iBka->ia",t1["Vo"],t2["ooVV"],v["oVov"])
	ws2s1["ov"] +=  2.000 * contract("Aj,ijBC,BCaA->ia",t1["Vo"],t2["ooVV"],v["VVvV"])
	ws2s1["ov"] +=  2.000 * contract("iA,bAjk,jkab->ia",t1["oV"],t2["vVoo"],v["oovv"])
	ws2s1["ov"] += -2.000 * contract("iA,ABjk,jkaB->ia",t1["oV"],t2["VVoo"],v["oovV"])
	ws2s1["ov"] += -2.000 * contract("iA,jkaB,ABjk->ia",t1["oV"],t2["oovV"],v["VVoo"])
	ws2s1["ov"] += -4.000 * contract("jA,bAjk,ikab->ia",t1["oV"],t2["vVoo"],v["oovv"])
	ws2s1["ov"] +=  4.000 * contract("jA,ABjk,ikaB->ia",t1["oV"],t2["VVoo"],v["oovV"])
	ws2s1["ov"] +=  4.000 * contract("jA,ikaB,ABjk->ia",t1["oV"],t2["oovV"],v["VVoo"])

	ws2s1["vo"] += -2.000 * contract("Ai,aBjk,jkAB->ai",t1["Vo"],t2["vVoo"],v["ooVV"])
	ws2s1["vo"] +=  2.000 * contract("Ai,jkbA,abjk->ai",t1["Vo"],t2["oovV"],v["vvoo"])
	ws2s1["vo"] += -2.000 * contract("Ai,jkAB,aBjk->ai",t1["Vo"],t2["ooVV"],v["vVoo"])
	ws2s1["vo"] +=  4.000 * contract("Aj,aBik,jkAB->ai",t1["Vo"],t2["vVoo"],v["ooVV"])
	ws2s1["vo"] += -4.000 * contract("Aj,jkbA,abik->ai",t1["Vo"],t2["oovV"],v["vvoo"])
	ws2s1["vo"] +=  4.000 * contract("Aj,jkAB,aBik->ai",t1["Vo"],t2["ooVV"],v["vVoo"])
	ws2s1["vo"] +=  4.000 * contract("jA,bBij,aAbB->ai",t1["oV"],t2["vVoo"],v["vVvV"])
	ws2s1["vo"] +=  2.000 * contract("jA,BCij,aABC->ai",t1["oV"],t2["VVoo"],v["vVVV"])
	ws2s1["vo"] += -4.000 * contract("jA,aBik,kAjB->ai",t1["oV"],t2["vVoo"],v["oVoV"])
	ws2s1["vo"] += -4.000 * contract("jA,bAik,kajb->ai",t1["oV"],t2["vVoo"],v["ovov"])
	ws2s1["vo"] +=  4.000 * contract("jA,ABik,kajB->ai",t1["oV"],t2["VVoo"],v["ovoV"])
	ws2s1["vo"] +=  4.000 * contract("jA,aBjk,kAiB->ai",t1["oV"],t2["vVoo"],v["oVoV"])
	ws2s1["vo"] +=  4.000 * contract("jA,bAjk,kaib->ai",t1["oV"],t2["vVoo"],v["ovov"])
	ws2s1["vo"] += -4.000 * contract("jA,ABjk,kaiB->ai",t1["oV"],t2["VVoo"],v["ovoV"])
	ws2s1["vo"] +=  2.000 * contract("jA,aAkl,klij->ai",t1["oV"],t2["vVoo"],v["oooo"])

	ws2s1["vv"] += -2.000 * contract("Ai,jkaA,ibjk->ba",t1["Vo"],t2["oovV"],v["ovoo"])
	ws2s1["vv"] +=  4.000 * contract("Ai,ijaB,bBjA->ba",t1["Vo"],t2["oovV"],v["vVoV"])
	ws2s1["vv"] +=  4.000 * contract("Ai,ijcA,bcja->ba",t1["Vo"],t2["oovV"],v["vvov"])
	ws2s1["vv"] += -4.000 * contract("Ai,ijAB,bBja->ba",t1["Vo"],t2["ooVV"],v["vVov"])
	ws2s1["vv"] +=  4.000 * contract("iA,aBij,jAbB->ab",t1["oV"],t2["vVoo"],v["oVvV"])
	ws2s1["vv"] +=  4.000 * contract("iA,cAij,jbac->ba",t1["oV"],t2["vVoo"],v["ovvv"])
	ws2s1["vv"] += -4.000 * contract("iA,ABij,jbaB->ba",t1["oV"],t2["VVoo"],v["ovvV"])
	ws2s1["vv"] += -2.000 * contract("iA,aAjk,jkib->ab",t1["oV"],t2["vVoo"],v["ooov"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	ws2s1["oooo"] += -4.000 * contract("Ai,jmaA,lakm->jlik",t1["Vo"],t2["oovV"],v["ovoo"])
	ws2s1["oooo"] +=  4.000 * contract("Ai,jmAB,lBkm->jlik",t1["Vo"],t2["ooVV"],v["oVoo"])
	ws2s1["oooo"] += -4.000 * contract("iA,aAjm,lmka->iljk",t1["oV"],t2["vVoo"],v["ooov"])
	ws2s1["oooo"] +=  4.000 * contract("iA,ABjm,lmkB->iljk",t1["oV"],t2["VVoo"],v["oooV"])
	ws2s1["oooo"] += -2.000 * contract("Am,imaA,lajk->iljk",t1["Vo"],t2["oovV"],v["ovoo"])
	ws2s1["oooo"] +=  2.000 * contract("Am,imAB,lBjk->iljk",t1["Vo"],t2["ooVV"],v["oVoo"])
	ws2s1["oooo"] += -2.000 * contract("iA,aBjk,lAaB->iljk",t1["oV"],t2["vVoo"],v["oVvV"])
	ws2s1["oooo"] += -2.000 * contract("mA,aAim,klja->klij",t1["oV"],t2["vVoo"],v["ooov"])
	ws2s1["oooo"] +=  2.000 * contract("mA,ABim,kljB->klij",t1["oV"],t2["VVoo"],v["oooV"])
	ws2s1["oooo"] += -2.000 * contract("Ai,jkaB,aBlA->jkil",t1["Vo"],t2["oovV"],v["vVoV"])
	ws2s1["oooo"] += -1.000 * contract("Ai,jkBC,BClA->jkil",t1["Vo"],t2["ooVV"],v["VVoV"])
	ws2s1["oooo"] +=  1.000 * contract("Am,ijaA,makl->ijkl",t1["Vo"],t2["oovV"],v["ovoo"])
	ws2s1["oooo"] += -1.000 * contract("Am,ijAB,mBkl->ijkl",t1["Vo"],t2["ooVV"],v["oVoo"])
	ws2s1["oooo"] += -1.000 * contract("iA,BCjk,lABC->iljk",t1["oV"],t2["VVoo"],v["oVVV"])
	ws2s1["oooo"] +=  1.000 * contract("mA,aAij,klma->klij",t1["oV"],t2["vVoo"],v["ooov"])
	ws2s1["oooo"] += -1.000 * contract("mA,ABij,klmB->klij",t1["oV"],t2["VVoo"],v["oooV"])
	ws2s1["oooo"] = asym_term(ws2s1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	ws2s1["ooov"] +=  2.000 * contract("Ai,jlaB,kBlA->jkia",t1["Vo"],t2["oovV"],v["oVoV"])
	ws2s1["ooov"] +=  2.000 * contract("Ai,jlbA,kbla->jkia",t1["Vo"],t2["oovV"],v["ovov"])
	ws2s1["ooov"] += -2.000 * contract("Ai,jlAB,kBla->jkia",t1["Vo"],t2["ooVV"],v["oVov"])
	ws2s1["ooov"] +=  2.000 * contract("Al,imaA,kljm->ikja",t1["Vo"],t2["oovV"],v["oooo"])
	ws2s1["ooov"] += -2.000 * contract("Al,ilaB,kBjA->ikja",t1["Vo"],t2["oovV"],v["oVoV"])
	ws2s1["ooov"] += -2.000 * contract("Al,ilbA,kbja->ikja",t1["Vo"],t2["oovV"],v["ovov"])
	ws2s1["ooov"] +=  2.000 * contract("Al,ilAB,kBja->ikja",t1["Vo"],t2["ooVV"],v["oVov"])
	ws2s1["ooov"] += -2.000 * contract("iA,bAjl,klab->ikja",t1["oV"],t2["vVoo"],v["oovv"])
	ws2s1["ooov"] +=  2.000 * contract("iA,ABjl,klaB->ikja",t1["oV"],t2["VVoo"],v["oovV"])
	ws2s1["ooov"] +=  2.000 * contract("iA,jlaB,ABkl->ijka",t1["oV"],t2["oovV"],v["VVoo"])
	ws2s1["ooov"] += -1.000 * contract("Ai,jkbB,bBaA->jkia",t1["Vo"],t2["oovV"],v["vVvV"])
	ws2s1["ooov"] +=  1.000 * contract("Al,lmaA,jkim->jkia",t1["Vo"],t2["oovV"],v["oooo"])
	ws2s1["ooov"] +=  1.000 * contract("Al,ijaB,lBkA->ijka",t1["Vo"],t2["oovV"],v["oVoV"])
	ws2s1["ooov"] +=  1.000 * contract("Al,ijbA,lbka->ijka",t1["Vo"],t2["oovV"],v["ovov"])
	ws2s1["ooov"] += -1.000 * contract("Al,ijAB,lBka->ijka",t1["Vo"],t2["ooVV"],v["oVov"])
	ws2s1["ooov"] += -1.000 * contract("lA,bAil,jkab->jkia",t1["oV"],t2["vVoo"],v["oovv"])
	ws2s1["ooov"] +=  1.000 * contract("lA,ABil,jkaB->jkia",t1["oV"],t2["VVoo"],v["oovV"])
	ws2s1["ooov"] +=  1.000 * contract("lA,ijaB,ABkl->ijka",t1["oV"],t2["oovV"],v["VVoo"])
	ws2s1["ooov"] += -0.500 * contract("Ai,lmaA,jklm->jkia",t1["Vo"],t2["oovV"],v["oooo"])
	ws2s1["ooov"] += -0.500 * contract("Ai,jkBC,BCaA->jkia",t1["Vo"],t2["ooVV"],v["VVvV"])
	ws2s1["ooov"] = asym_term(ws2s1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	ws2s1["ovoo"] +=  2.000 * contract("Ai,aBjl,klAB->kaij",t1["Vo"],t2["vVoo"],v["ooVV"])
	ws2s1["ovoo"] += -2.000 * contract("Ai,jlbA,abkl->jaik",t1["Vo"],t2["oovV"],v["vvoo"])
	ws2s1["ovoo"] +=  2.000 * contract("Ai,jlAB,aBkl->jaik",t1["Vo"],t2["ooVV"],v["vVoo"])
	ws2s1["ovoo"] +=  2.000 * contract("iA,aBjl,lAkB->iajk",t1["oV"],t2["vVoo"],v["oVoV"])
	ws2s1["ovoo"] +=  2.000 * contract("iA,bAjl,lakb->iajk",t1["oV"],t2["vVoo"],v["ovov"])
	ws2s1["ovoo"] += -2.000 * contract("iA,ABjl,lakB->iajk",t1["oV"],t2["VVoo"],v["ovoV"])
	ws2s1["ovoo"] += -2.000 * contract("lA,aBil,kAjB->kaij",t1["oV"],t2["vVoo"],v["oVoV"])
	ws2s1["ovoo"] += -2.000 * contract("lA,bAil,kajb->kaij",t1["oV"],t2["vVoo"],v["ovov"])
	ws2s1["ovoo"] +=  2.000 * contract("lA,ABil,kajB->kaij",t1["oV"],t2["VVoo"],v["ovoV"])
	ws2s1["ovoo"] +=  2.000 * contract("lA,aAim,kmjl->kaij",t1["oV"],t2["vVoo"],v["oooo"])
	ws2s1["ovoo"] +=  1.000 * contract("Al,aBij,klAB->kaij",t1["Vo"],t2["vVoo"],v["ooVV"])
	ws2s1["ovoo"] += -1.000 * contract("Al,ilbA,abjk->iajk",t1["Vo"],t2["oovV"],v["vvoo"])
	ws2s1["ovoo"] +=  1.000 * contract("Al,ilAB,aBjk->iajk",t1["Vo"],t2["ooVV"],v["vVoo"])
	ws2s1["ovoo"] += -1.000 * contract("iA,bBjk,aAbB->iajk",t1["oV"],t2["vVoo"],v["vVvV"])
	ws2s1["ovoo"] +=  1.000 * contract("lA,aBij,kAlB->kaij",t1["oV"],t2["vVoo"],v["oVoV"])
	ws2s1["ovoo"] +=  1.000 * contract("lA,bAij,kalb->kaij",t1["oV"],t2["vVoo"],v["ovov"])
	ws2s1["ovoo"] += -1.000 * contract("lA,ABij,kalB->kaij",t1["oV"],t2["VVoo"],v["ovoV"])
	ws2s1["ovoo"] +=  1.000 * contract("lA,aAlm,kmij->kaij",t1["oV"],t2["vVoo"],v["oooo"])
	ws2s1["ovoo"] += -0.500 * contract("iA,BCjk,aABC->iajk",t1["oV"],t2["VVoo"],v["vVVV"])
	ws2s1["ovoo"] += -0.500 * contract("iA,aAlm,lmjk->iajk",t1["oV"],t2["vVoo"],v["oooo"])
	ws2s1["ovoo"] = asym_term(ws2s1["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	ws2s1["oovv"] +=  4.000 * contract("Ak,ilaA,jklb->ijab",t1["Vo"],t2["oovV"],v["ooov"])
	ws2s1["oovv"] +=  4.000 * contract("Ak,ikaB,jBbA->ijab",t1["Vo"],t2["oovV"],v["oVvV"])
	ws2s1["oovv"] +=  4.000 * contract("iA,jkaB,ABkb->ijab",t1["oV"],t2["oovV"],v["VVov"])
	ws2s1["oovv"] +=  2.000 * contract("Ak,klaA,ijlb->ijab",t1["Vo"],t2["oovV"],v["ooov"])
	ws2s1["oovv"] += -2.000 * contract("Ak,ijaB,kBbA->ijab",t1["Vo"],t2["oovV"],v["oVvV"])
	ws2s1["oovv"] += -2.000 * contract("Ak,ikcA,jcab->ijab",t1["Vo"],t2["oovV"],v["ovvv"])
	ws2s1["oovv"] +=  2.000 * contract("Ak,ikAB,jBab->ijab",t1["Vo"],t2["ooVV"],v["oVvv"])
	ws2s1["oovv"] +=  2.000 * contract("kA,ijaB,ABkb->ijab",t1["oV"],t2["oovV"],v["VVov"])
	ws2s1["oovv"] +=  1.000 * contract("Ak,ijcA,kcab->ijab",t1["Vo"],t2["oovV"],v["ovvv"])
	ws2s1["oovv"] += -1.000 * contract("Ak,ijAB,kBab->ijab",t1["Vo"],t2["ooVV"],v["oVvv"])
	ws2s1["oovv"] = asym_term(ws2s1["oovv"],"oovv")

	ws2s1["ovov"] +=  1.000 * contract("Ai,jkaB,bBkA->jbia",t1["Vo"],t2["oovV"],v["vVoV"])
	ws2s1["ovov"] +=  1.000 * contract("Ai,jkcA,bcka->jbia",t1["Vo"],t2["oovV"],v["vvov"])
	ws2s1["ovov"] += -1.000 * contract("Ai,jkAB,bBka->jbia",t1["Vo"],t2["ooVV"],v["vVov"])
	ws2s1["ovov"] += -1.000 * contract("Ak,ilaA,kbjl->ibja",t1["Vo"],t2["oovV"],v["ovoo"])
	ws2s1["ovov"] +=  1.000 * contract("Ak,klaA,jbil->jbia",t1["Vo"],t2["oovV"],v["ovoo"])
	ws2s1["ovov"] += -1.000 * contract("Ak,ikaB,bBjA->ibja",t1["Vo"],t2["oovV"],v["vVoV"])
	ws2s1["ovov"] += -1.000 * contract("Ak,ikcA,bcja->ibja",t1["Vo"],t2["oovV"],v["vvov"])
	ws2s1["ovov"] +=  1.000 * contract("Ak,ikAB,bBja->ibja",t1["Vo"],t2["ooVV"],v["vVov"])
	ws2s1["ovov"] +=  1.000 * contract("iA,aBjk,kAbB->iajb",t1["oV"],t2["vVoo"],v["oVvV"])
	ws2s1["ovov"] +=  1.000 * contract("iA,cAjk,kbac->ibja",t1["oV"],t2["vVoo"],v["ovvv"])
	ws2s1["ovov"] += -1.000 * contract("iA,ABjk,kbaB->ibja",t1["oV"],t2["VVoo"],v["ovvV"])
	ws2s1["ovov"] += -1.000 * contract("kA,aBik,jAbB->jaib",t1["oV"],t2["vVoo"],v["oVvV"])
	ws2s1["ovov"] += -1.000 * contract("kA,cAik,jbac->jbia",t1["oV"],t2["vVoo"],v["ovvv"])
	ws2s1["ovov"] +=  1.000 * contract("kA,ABik,jbaB->jbia",t1["oV"],t2["VVoo"],v["ovvV"])
	ws2s1["ovov"] += -1.000 * contract("kA,aAil,jlkb->jaib",t1["oV"],t2["vVoo"],v["ooov"])
	ws2s1["ovov"] +=  1.000 * contract("kA,aAkl,jlib->jaib",t1["oV"],t2["vVoo"],v["ooov"])
	ws2s1["ovov"] += -0.500 * contract("Ai,klaA,jbkl->jbia",t1["Vo"],t2["oovV"],v["ovoo"])
	ws2s1["ovov"] += -0.500 * contract("iA,aAkl,kljb->iajb",t1["oV"],t2["vVoo"],v["ooov"])

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	ws2s1["vvoo"] +=  4.000 * contract("Ai,aBjk,kbAB->abij",t1["Vo"],t2["vVoo"],v["ovVV"])
	ws2s1["vvoo"] +=  4.000 * contract("kA,aBik,bAjB->abij",t1["oV"],t2["vVoo"],v["vVoV"])
	ws2s1["vvoo"] +=  4.000 * contract("kA,aAil,lbjk->abij",t1["oV"],t2["vVoo"],v["ovoo"])
	ws2s1["vvoo"] += -2.000 * contract("kA,cAik,abjc->abij",t1["oV"],t2["vVoo"],v["vvov"])
	ws2s1["vvoo"] +=  2.000 * contract("kA,ABik,abjB->abij",t1["oV"],t2["VVoo"],v["vvoV"])
	ws2s1["vvoo"] +=  2.000 * contract("kA,aAkl,lbij->abij",t1["oV"],t2["vVoo"],v["ovoo"])
	ws2s1["vvoo"] +=  2.000 * contract("Ak,aBij,kbAB->abij",t1["Vo"],t2["vVoo"],v["ovVV"])
	ws2s1["vvoo"] += -2.000 * contract("kA,aBij,bAkB->abij",t1["oV"],t2["vVoo"],v["vVoV"])
	ws2s1["vvoo"] +=  1.000 * contract("kA,cAij,abkc->abij",t1["oV"],t2["vVoo"],v["vvov"])
	ws2s1["vvoo"] += -1.000 * contract("kA,ABij,abkB->abij",t1["oV"],t2["VVoo"],v["vvoV"])
	ws2s1["vvoo"] = asym_term(ws2s1["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	ws2s1["ovvv"] += -2.000 * contract("Aj,ikaA,jckb->icab",t1["Vo"],t2["oovV"],v["ovov"])
	ws2s1["ovvv"] +=  2.000 * contract("Aj,jkaA,ickb->icab",t1["Vo"],t2["oovV"],v["ovov"])
	ws2s1["ovvv"] +=  2.000 * contract("Aj,ijaB,cBbA->icab",t1["Vo"],t2["oovV"],v["vVvV"])
	ws2s1["ovvv"] += -1.000 * contract("Aj,ijdA,cdab->icab",t1["Vo"],t2["oovV"],v["vvvv"])
	ws2s1["ovvv"] +=  1.000 * contract("Aj,ijAB,cBab->icab",t1["Vo"],t2["ooVV"],v["vVvv"])
	ws2s1["ovvv"] +=  1.000 * contract("jA,aAjk,ikbc->iabc",t1["oV"],t2["vVoo"],v["oovv"])
	ws2s1["ovvv"] += -0.500 * contract("iA,aAjk,jkbc->iabc",t1["oV"],t2["vVoo"],v["oovv"])
	ws2s1["ovvv"] = asym_term(ws2s1["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	ws2s1["vvov"] +=  2.000 * contract("jA,aBij,cAbB->acib",t1["oV"],t2["vVoo"],v["vVvV"])
	ws2s1["vvov"] += -2.000 * contract("jA,aAik,kcjb->acib",t1["oV"],t2["vVoo"],v["ovov"])
	ws2s1["vvov"] +=  2.000 * contract("jA,aAjk,kcib->acib",t1["oV"],t2["vVoo"],v["ovov"])
	ws2s1["vvov"] += -1.000 * contract("jA,dAij,bcad->bcia",t1["oV"],t2["vVoo"],v["vvvv"])
	ws2s1["vvov"] +=  1.000 * contract("jA,ABij,bcaB->bcia",t1["oV"],t2["VVoo"],v["vvvV"])
	ws2s1["vvov"] +=  1.000 * contract("Aj,jkaA,bcik->bcia",t1["Vo"],t2["oovV"],v["vvoo"])
	ws2s1["vvov"] += -0.500 * contract("Ai,jkaA,bcjk->bcia",t1["Vo"],t2["oovV"],v["vvoo"])
	ws2s1["vvov"] = asym_term(ws2s1["vvov"],"vvov")

	# vvvv = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_virt_int))
	ws2s1["vvvv"] +=  2.000 * contract("Ai,ijaA,cdjb->cdab",t1["Vo"],t2["oovV"],v["vvov"])
	ws2s1["vvvv"] +=  2.000 * contract("iA,aAij,jdbc->adbc",t1["oV"],t2["vVoo"],v["ovvv"])
	ws2s1["vvvv"] = asym_term(ws2s1["vvvv"],"vvvv")

	if(inc_3_body):
		ws2s1["oooooo"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ))
		ws2s1["ooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		ws2s1["oovooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		ws2s1["oooovv"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s1["oovoov"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s1["ovvooo"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2s1["ooovvv"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s1["oovovv"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s1["ovvoov"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s1["vvvooo"] = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2s1["ovvovv"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_virt_int,n_virt_int))

		# oooooo = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ))
		ws2s1["oooooo"] +=  1.000 * contract("Ai,jkaA,nalm->jknilm",t1["Vo"],t2["oovV"],v["ovoo"])
		ws2s1["oooooo"] += -1.000 * contract("Ai,jkAB,nBlm->jknilm",t1["Vo"],t2["ooVV"],v["oVoo"])
		ws2s1["oooooo"] +=  1.000 * contract("iA,aAjk,mnla->imnjkl",t1["oV"],t2["vVoo"],v["ooov"])
		ws2s1["oooooo"] += -1.000 * contract("iA,ABjk,mnlB->imnjkl",t1["oV"],t2["VVoo"],v["oooV"])
		ws2s1["oooooo"] = asym_term(ws2s1["oooooo"],"oooooo")

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		ws2s1["ooooov"] +=  (2./3.) * contract("Ai,jnaA,lmkn->jlmika",t1["Vo"],t2["oovV"],v["oooo"])
		ws2s1["ooooov"] +=  (2./3.) * contract("Ai,jkaB,mBlA->jkmila",t1["Vo"],t2["oovV"],v["oVoV"])
		ws2s1["ooooov"] +=  (2./3.) * contract("Ai,jkbA,mbla->jkmila",t1["Vo"],t2["oovV"],v["ovov"])
		ws2s1["ooooov"] += -(2./3.) * contract("Ai,jkAB,mBla->jkmila",t1["Vo"],t2["ooVV"],v["oVov"])
		ws2s1["ooooov"] +=  (1./3.) * contract("iA,bAjk,lmab->ilmjka",t1["oV"],t2["vVoo"],v["oovv"])
		ws2s1["ooooov"] += -(1./3.) * contract("iA,ABjk,lmaB->ilmjka",t1["oV"],t2["VVoo"],v["oovV"])
		ws2s1["ooooov"] += -(1./3.) * contract("iA,jkaB,ABlm->ijklma",t1["oV"],t2["oovV"],v["VVoo"])
		ws2s1["ooooov"] = asym_term(ws2s1["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		ws2s1["oovooo"] +=  (2./3.) * contract("iA,aBjk,mAlB->imajkl",t1["oV"],t2["vVoo"],v["oVoV"])
		ws2s1["oovooo"] +=  (2./3.) * contract("iA,bAjk,malb->imajkl",t1["oV"],t2["vVoo"],v["ovov"])
		ws2s1["oovooo"] += -(2./3.) * contract("iA,ABjk,malB->imajkl",t1["oV"],t2["VVoo"],v["ovoV"])
		ws2s1["oovooo"] +=  (2./3.) * contract("iA,aAjn,mnkl->imajkl",t1["oV"],t2["vVoo"],v["oooo"])
		ws2s1["oovooo"] += -(1./3.) * contract("Ai,aBjk,lmAB->lmaijk",t1["Vo"],t2["vVoo"],v["ooVV"])
		ws2s1["oovooo"] +=  (1./3.) * contract("Ai,jkbA,ablm->jkailm",t1["Vo"],t2["oovV"],v["vvoo"])
		ws2s1["oovooo"] += -(1./3.) * contract("Ai,jkAB,aBlm->jkailm",t1["Vo"],t2["ooVV"],v["vVoo"])
		ws2s1["oovooo"] = asym_term(ws2s1["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s1["oooovv"] +=  (2./3.) * contract("Ai,jmaA,klmb->jkliab",t1["Vo"],t2["oovV"],v["ooov"])
		ws2s1["oooovv"] += -(2./3.) * contract("Ai,jkaB,lBbA->jkliab",t1["Vo"],t2["oovV"],v["oVvV"])
		ws2s1["oooovv"] +=  (2./3.) * contract("iA,jkaB,ABlb->ijklab",t1["oV"],t2["oovV"],v["VVov"])
		ws2s1["oooovv"] +=  (1./3.) * contract("Ai,jkcA,lcab->jkliab",t1["Vo"],t2["oovV"],v["ovvv"])
		ws2s1["oooovv"] += -(1./3.) * contract("Ai,jkAB,lBab->jkliab",t1["Vo"],t2["ooVV"],v["oVvv"])
		ws2s1["oooovv"] = asym_term(ws2s1["oooovv"],"oooovv")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s1["oovoov"] +=  (4./9.) * contract("Ai,jmaA,lbkm->jlbika",t1["Vo"],t2["oovV"],v["ovoo"])
		ws2s1["oovoov"] +=  (4./9.) * contract("iA,aAjm,lmkb->ilajkb",t1["oV"],t2["vVoo"],v["ooov"])
		ws2s1["oovoov"] +=  (2./9.) * contract("Ai,jkaB,bBlA->jkbila",t1["Vo"],t2["oovV"],v["vVoV"])
		ws2s1["oovoov"] +=  (2./9.) * contract("Ai,jkcA,bcla->jkbila",t1["Vo"],t2["oovV"],v["vvov"])
		ws2s1["oovoov"] += -(2./9.) * contract("Ai,jkAB,bBla->jkbila",t1["Vo"],t2["ooVV"],v["vVov"])
		ws2s1["oovoov"] +=  (2./9.) * contract("iA,aBjk,lAbB->ilajkb",t1["oV"],t2["vVoo"],v["oVvV"])
		ws2s1["oovoov"] +=  (2./9.) * contract("iA,cAjk,lbac->ilbjka",t1["oV"],t2["vVoo"],v["ovvv"])
		ws2s1["oovoov"] += -(2./9.) * contract("iA,ABjk,lbaB->ilbjka",t1["oV"],t2["VVoo"],v["ovvV"])
		ws2s1["oovoov"] = asym_term(ws2s1["oovoov"],"oovoov")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2s1["ovvooo"] +=  (2./3.) * contract("Ai,aBjk,lbAB->labijk",t1["Vo"],t2["vVoo"],v["ovVV"])
		ws2s1["ovvooo"] += -(2./3.) * contract("iA,aBjk,bAlB->iabjkl",t1["oV"],t2["vVoo"],v["vVoV"])
		ws2s1["ovvooo"] +=  (2./3.) * contract("iA,aAjm,mbkl->iabjkl",t1["oV"],t2["vVoo"],v["ovoo"])
		ws2s1["ovvooo"] +=  (1./3.) * contract("iA,cAjk,ablc->iabjkl",t1["oV"],t2["vVoo"],v["vvov"])
		ws2s1["ovvooo"] += -(1./3.) * contract("iA,ABjk,ablB->iabjkl",t1["oV"],t2["VVoo"],v["vvoV"])
		ws2s1["ovvooo"] = asym_term(ws2s1["ovvooo"],"ovvooo")

		# ooovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s1["ooovvv"] += -1.000 * contract("iA,jkaB,ABbc->ijkabc",t1["oV"],t2["oovV"],v["VVvv"])
		ws2s1["ooovvv"] = asym_term(ws2s1["ooovvv"],"ooovvv")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s1["oovovv"] +=  (4./9.) * contract("Ai,jlaA,kclb->jkciab",t1["Vo"],t2["oovV"],v["ovov"])
		ws2s1["oovovv"] += -(2./9.) * contract("Ai,jkaB,cBbA->jkciab",t1["Vo"],t2["oovV"],v["vVvV"])
		ws2s1["oovovv"] +=  (2./9.) * contract("iA,aAjl,klbc->ikajbc",t1["oV"],t2["vVoo"],v["oovv"])
		ws2s1["oovovv"] +=  (1./9.) * contract("Ai,jkdA,cdab->jkciab",t1["Vo"],t2["oovV"],v["vvvv"])
		ws2s1["oovovv"] += -(1./9.) * contract("Ai,jkAB,cBab->jkciab",t1["Vo"],t2["ooVV"],v["vVvv"])
		ws2s1["oovovv"] = asym_term(ws2s1["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s1["ovvoov"] +=  (4./9.) * contract("iA,aAjl,lckb->iacjkb",t1["oV"],t2["vVoo"],v["ovov"])
		ws2s1["ovvoov"] +=  (2./9.) * contract("Ai,jlaA,bckl->jbcika",t1["Vo"],t2["oovV"],v["vvoo"])
		ws2s1["ovvoov"] += -(2./9.) * contract("iA,aBjk,cAbB->iacjkb",t1["oV"],t2["vVoo"],v["vVvV"])
		ws2s1["ovvoov"] +=  (1./9.) * contract("iA,dAjk,bcad->ibcjka",t1["oV"],t2["vVoo"],v["vvvv"])
		ws2s1["ovvoov"] += -(1./9.) * contract("iA,ABjk,bcaB->ibcjka",t1["oV"],t2["VVoo"],v["vvvV"])
		ws2s1["ovvoov"] = asym_term(ws2s1["ovvoov"],"ovvoov")

		# vvvooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2s1["vvvooo"] += -1.000 * contract("Ai,aBjk,bcAB->abcijk",t1["Vo"],t2["vVoo"],v["vvVV"])
		ws2s1["vvvooo"] = asym_term(ws2s1["vvvooo"],"vvvooo")

		# ovvovv = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s1["ovvovv"] +=  (2./9.) * contract("Ai,jkaA,cdkb->jcdiab",t1["Vo"],t2["oovV"],v["vvov"])
		ws2s1["ovvovv"] +=  (2./9.) * contract("iA,aAjk,kdbc->iadjbc",t1["oV"],t2["vVoo"],v["ovvv"])
		ws2s1["ovvovv"] = asym_term(ws2s1["ovvovv"],"ovvovv")

	return ws2s1 

def wn_s2_s2(v,t2,inc_3_body=True,inc_4_body=True):
	# [[Wn,S_2ext],S_2ext]
	# for sizing arrays
	n_occ = v["oooo"].shape[0]
	n_virt_int = v["vvvv"].shape[0]
	# initialize
	ws2s2 = {
		"c": 0.0,
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"vv": np.zeros((n_virt_int,n_virt_int)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"ovov": np.zeros((n_occ,n_virt_int,n_occ,n_virt_int)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ)),
		"ovvv": np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int)),
		"vvov": np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int)),
		"vvvv": np.zeros((n_virt_int,n_virt_int,n_virt_int,n_virt_int))
	}
	# Populate [[Wn,S_2ext],S_2ext]
	ws2s2["c"] +=  8.000 * contract("ijAB,aAik,kBja->",t2["ooVV"],t2["vVoo"],v["oVov"])
	ws2s2["c"] += -8.000 * contract("ijAB,ACik,kBjC->",t2["ooVV"],t2["VVoo"],v["oVoV"])
	ws2s2["c"] += -8.000 * contract("ijaA,aBik,kAjB->",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["c"] += -8.000 * contract("ijaA,bAik,kajb->",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["c"] +=  8.000 * contract("ijaA,ABik,kajB->",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["c"] +=  4.000 * contract("ijaA,bBij,aAbB->",t2["oovV"],t2["vVoo"],v["vVvV"])
	ws2s2["c"] +=  2.000 * contract("ijaA,aAkl,klij->",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["c"] +=  2.000 * contract("ijAB,aCij,ABaC->",t2["ooVV"],t2["vVoo"],v["VVvV"])
	ws2s2["c"] +=  2.000 * contract("ijaA,BCij,aABC->",t2["oovV"],t2["VVoo"],v["vVVV"])
	ws2s2["c"] +=  1.000 * contract("ijAB,CDij,ABCD->",t2["ooVV"],t2["VVoo"],v["VVVV"])
	ws2s2["c"] +=  1.000 * contract("ijAB,ABkl,klij->",t2["ooVV"],t2["VVoo"],v["oooo"])

	ws2s2["oo"] += -8.000 * contract("klaA,aAkm,jmil->ji",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["oo"] +=  8.000 * contract("klaA,bAik,jalb->ji",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["oo"] += -8.000 * contract("ikaA,bAjl,lakb->ij",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["oo"] +=  8.000 * contract("ikaA,bAkl,lajb->ij",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["oo"] +=  8.000 * contract("ikaA,bBjk,aAbB->ij",t2["oovV"],t2["vVoo"],v["vVvV"])
	ws2s2["oo"] += -8.000 * contract("ikaA,aBjl,lAkB->ij",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["oo"] +=  8.000 * contract("ikaA,ABjl,lakB->ij",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["oo"] +=  8.000 * contract("ikaA,aBkl,lAjB->ij",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["oo"] += -8.000 * contract("ikaA,ABkl,lajB->ij",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["oo"] +=  8.000 * contract("klaA,aBik,jAlB->ji",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["oo"] += -8.000 * contract("klaA,ABik,jalB->ji",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["oo"] +=  8.000 * contract("ikAB,aAjl,lBka->ij",t2["ooVV"],t2["vVoo"],v["oVov"])
	ws2s2["oo"] += -8.000 * contract("ikAB,ACjl,lBkC->ij",t2["ooVV"],t2["VVoo"],v["oVoV"])
	ws2s2["oo"] += -8.000 * contract("ikAB,aAkl,lBja->ij",t2["ooVV"],t2["vVoo"],v["oVov"])
	ws2s2["oo"] +=  8.000 * contract("ikAB,ACkl,lBjC->ij",t2["ooVV"],t2["VVoo"],v["oVoV"])
	ws2s2["oo"] += -8.000 * contract("klAB,aAik,jBla->ji",t2["ooVV"],t2["vVoo"],v["oVov"])
	ws2s2["oo"] +=  8.000 * contract("klAB,ACik,jBlC->ji",t2["ooVV"],t2["VVoo"],v["oVoV"])
	ws2s2["oo"] +=  4.000 * contract("ikAB,aCjk,ABaC->ij",t2["ooVV"],t2["vVoo"],v["VVvV"])
	ws2s2["oo"] +=  4.000 * contract("klaA,aAim,jmkl->ji",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["oo"] +=  4.000 * contract("klaA,aBkl,jAiB->ji",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["oo"] +=  4.000 * contract("klaA,bAkl,jaib->ji",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["oo"] += -4.000 * contract("klaA,ABkl,jaiB->ji",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["oo"] +=  4.000 * contract("ikaA,aAlm,lmjk->ij",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["oo"] +=  4.000 * contract("ikaA,BCjk,aABC->ij",t2["oovV"],t2["VVoo"],v["vVVV"])
	ws2s2["oo"] += -4.000 * contract("klAB,aAkl,jBia->ji",t2["ooVV"],t2["vVoo"],v["oVov"])
	ws2s2["oo"] +=  4.000 * contract("klAB,ACkl,jBiC->ji",t2["ooVV"],t2["VVoo"],v["oVoV"])
	ws2s2["oo"] += -4.000 * contract("klAB,ABkm,jmil->ji",t2["ooVV"],t2["VVoo"],v["oooo"])
	ws2s2["oo"] +=  2.000 * contract("ikAB,ABlm,lmjk->ij",t2["ooVV"],t2["VVoo"],v["oooo"])
	ws2s2["oo"] +=  2.000 * contract("klAB,ABim,jmkl->ji",t2["ooVV"],t2["VVoo"],v["oooo"])
	ws2s2["oo"] +=  2.000 * contract("ikAB,CDjk,ABCD->ij",t2["ooVV"],t2["VVoo"],v["VVVV"])

	ws2s2["ov"] += -8.000 * contract("ijaA,bBjk,kAbB->ia",t2["oovV"],t2["vVoo"],v["oVvV"])
	ws2s2["ov"] += -4.000 * contract("ijaA,BCjk,kABC->ia",t2["oovV"],t2["VVoo"],v["oVVV"])
	ws2s2["ov"] +=  4.000 * contract("ijaA,bAkl,kljb->ia",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ov"] += -4.000 * contract("ijaA,ABkl,kljB->ia",t2["oovV"],t2["VVoo"],v["oooV"])
	ws2s2["ov"] += -4.000 * contract("jkaA,bBjk,iAbB->ia",t2["oovV"],t2["vVoo"],v["oVvV"])
	ws2s2["ov"] += -2.000 * contract("jkaA,BCjk,iABC->ia",t2["oovV"],t2["VVoo"],v["oVVV"])
	ws2s2["ov"] += -8.000 * contract("jkaA,bAjl,ilkb->ia",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ov"] +=  8.000 * contract("jkaA,ABjl,ilkB->ia",t2["oovV"],t2["VVoo"],v["oooV"])
	ws2s2["ov"] +=  8.000 * contract("ijbA,bBjk,kAaB->ia",t2["oovV"],t2["vVoo"],v["oVvV"])
	ws2s2["ov"] +=  8.000 * contract("ijbA,cAjk,kbac->ia",t2["oovV"],t2["vVoo"],v["ovvv"])
	ws2s2["ov"] += -8.000 * contract("ijbA,ABjk,kbaB->ia",t2["oovV"],t2["VVoo"],v["ovvV"])
	ws2s2["ov"] += -4.000 * contract("ijbA,bAkl,klja->ia",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ov"] +=  4.000 * contract("jkbA,bBjk,iAaB->ia",t2["oovV"],t2["vVoo"],v["oVvV"])
	ws2s2["ov"] +=  4.000 * contract("jkbA,cAjk,ibac->ia",t2["oovV"],t2["vVoo"],v["ovvv"])
	ws2s2["ov"] += -4.000 * contract("jkbA,ABjk,ibaB->ia",t2["oovV"],t2["VVoo"],v["ovvV"])
	ws2s2["ov"] +=  8.000 * contract("jkbA,bAjl,ilka->ia",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ov"] += -8.000 * contract("ijAB,bAjk,kBab->ia",t2["ooVV"],t2["vVoo"],v["oVvv"])
	ws2s2["ov"] +=  8.000 * contract("ijAB,ACjk,kBaC->ia",t2["ooVV"],t2["VVoo"],v["oVvV"])
	ws2s2["ov"] += -2.000 * contract("ijAB,ABkl,klja->ia",t2["ooVV"],t2["VVoo"],v["ooov"])
	ws2s2["ov"] += -4.000 * contract("jkAB,bAjk,iBab->ia",t2["ooVV"],t2["vVoo"],v["oVvv"])
	ws2s2["ov"] +=  4.000 * contract("jkAB,ACjk,iBaC->ia",t2["ooVV"],t2["VVoo"],v["oVvV"])
	ws2s2["ov"] +=  4.000 * contract("jkAB,ABjl,ilka->ia",t2["ooVV"],t2["VVoo"],v["ooov"])

	ws2s2["vo"] += -8.000 * contract("jkbA,aBij,bAkB->ai",t2["oovV"],t2["vVoo"],v["vVoV"])
	ws2s2["vo"] +=  8.000 * contract("jkbA,bBij,aAkB->ai",t2["oovV"],t2["vVoo"],v["vVoV"])
	ws2s2["vo"] +=  8.000 * contract("jkbA,cAij,abkc->ai",t2["oovV"],t2["vVoo"],v["vvov"])
	ws2s2["vo"] += -8.000 * contract("jkbA,ABij,abkB->ai",t2["oovV"],t2["VVoo"],v["vvoV"])
	ws2s2["vo"] +=  4.000 * contract("jkbA,aAil,lbjk->ai",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["vo"] += -4.000 * contract("jkbA,bAil,lajk->ai",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["vo"] += -4.000 * contract("jkbA,aBjk,bAiB->ai",t2["oovV"],t2["vVoo"],v["vVoV"])
	ws2s2["vo"] +=  4.000 * contract("jkbA,bBjk,aAiB->ai",t2["oovV"],t2["vVoo"],v["vVoV"])
	ws2s2["vo"] +=  4.000 * contract("jkbA,cAjk,abic->ai",t2["oovV"],t2["vVoo"],v["vvov"])
	ws2s2["vo"] += -4.000 * contract("jkbA,ABjk,abiB->ai",t2["oovV"],t2["VVoo"],v["vvoV"])
	ws2s2["vo"] += -8.000 * contract("jkbA,aAjl,lbik->ai",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["vo"] +=  8.000 * contract("jkbA,bAjl,laik->ai",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["vo"] += -4.000 * contract("jkAB,aCij,ABkC->ai",t2["ooVV"],t2["vVoo"],v["VVoV"])
	ws2s2["vo"] += -8.000 * contract("jkAB,bAij,aBkb->ai",t2["ooVV"],t2["vVoo"],v["vVov"])
	ws2s2["vo"] +=  8.000 * contract("jkAB,ACij,aBkC->ai",t2["ooVV"],t2["VVoo"],v["vVoV"])
	ws2s2["vo"] += -4.000 * contract("jkAB,aAil,lBjk->ai",t2["ooVV"],t2["vVoo"],v["oVoo"])
	ws2s2["vo"] += -2.000 * contract("jkAB,ABil,lajk->ai",t2["ooVV"],t2["VVoo"],v["ovoo"])
	ws2s2["vo"] += -2.000 * contract("jkAB,aCjk,ABiC->ai",t2["ooVV"],t2["vVoo"],v["VVoV"])
	ws2s2["vo"] += -4.000 * contract("jkAB,bAjk,aBib->ai",t2["ooVV"],t2["vVoo"],v["vVov"])
	ws2s2["vo"] +=  4.000 * contract("jkAB,ACjk,aBiC->ai",t2["ooVV"],t2["VVoo"],v["vVoV"])
	ws2s2["vo"] +=  8.000 * contract("jkAB,aAjl,lBik->ai",t2["ooVV"],t2["vVoo"],v["oVoo"])
	ws2s2["vo"] +=  4.000 * contract("jkAB,ABjl,laik->ai",t2["ooVV"],t2["VVoo"],v["ovoo"])

	ws2s2["vv"] += -4.000 * contract("ijaA,cBij,bAcB->ba",t2["oovV"],t2["vVoo"],v["vVvV"])
	ws2s2["vv"] += -2.000 * contract("ijaA,BCij,bABC->ba",t2["oovV"],t2["VVoo"],v["vVVV"])
	ws2s2["vv"] +=  8.000 * contract("ijaA,bBik,kAjB->ba",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["vv"] +=  8.000 * contract("ijaA,cAik,kbjc->ba",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["vv"] += -8.000 * contract("ijaA,ABik,kbjB->ba",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["vv"] += -2.000 * contract("ijaA,bAkl,klij->ba",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["vv"] += -4.000 * contract("ijcA,aBij,cAbB->ab",t2["oovV"],t2["vVoo"],v["vVvV"])
	ws2s2["vv"] +=  4.000 * contract("ijcA,cBij,bAaB->ba",t2["oovV"],t2["vVoo"],v["vVvV"])
	ws2s2["vv"] +=  4.000 * contract("ijcA,dAij,bcad->ba",t2["oovV"],t2["vVoo"],v["vvvv"])
	ws2s2["vv"] += -4.000 * contract("ijcA,ABij,bcaB->ba",t2["oovV"],t2["VVoo"],v["vvvV"])
	ws2s2["vv"] +=  8.000 * contract("ijcA,aAik,kcjb->ab",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["vv"] += -8.000 * contract("ijcA,cAik,kbja->ba",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["vv"] += -2.000 * contract("ijAB,aCij,ABbC->ab",t2["ooVV"],t2["vVoo"],v["VVvV"])
	ws2s2["vv"] += -4.000 * contract("ijAB,cAij,bBac->ba",t2["ooVV"],t2["vVoo"],v["vVvv"])
	ws2s2["vv"] +=  4.000 * contract("ijAB,ACij,bBaC->ba",t2["ooVV"],t2["VVoo"],v["vVvV"])
	ws2s2["vv"] += -8.000 * contract("ijAB,aAik,kBjb->ab",t2["ooVV"],t2["vVoo"],v["oVov"])
	ws2s2["vv"] += -4.000 * contract("ijAB,ABik,kbja->ba",t2["ooVV"],t2["VVoo"],v["ovov"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	ws2s2["oooo"] +=  8.0000 * contract("imaA,aBjm,lAkB->iljk",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["oooo"] +=  8.0000 * contract("imaA,bAjm,lakb->iljk",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["oooo"] += -8.0000 * contract("imaA,ABjm,lakB->iljk",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["oooo"] += -8.0000 * contract("imaA,aAjn,lnkm->iljk",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["oooo"] += -8.0000 * contract("imAB,aAjm,lBka->iljk",t2["ooVV"],t2["vVoo"],v["oVov"])
	ws2s2["oooo"] +=  8.0000 * contract("imAB,ACjm,lBkC->iljk",t2["ooVV"],t2["VVoo"],v["oVoV"])
	ws2s2["oooo"] += -4.0000 * contract("ijaA,aBkm,mAlB->ijkl",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["oooo"] += -4.0000 * contract("ijaA,bAkm,malb->ijkl",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["oooo"] +=  4.0000 * contract("ijaA,ABkm,malB->ijkl",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["oooo"] += -4.0000 * contract("imaA,aBjk,lAmB->iljk",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["oooo"] += -4.0000 * contract("imaA,bAjk,lamb->iljk",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["oooo"] +=  4.0000 * contract("imaA,ABjk,lamB->iljk",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["oooo"] +=  4.0000 * contract("ijAB,aAkm,mBla->ijkl",t2["ooVV"],t2["vVoo"],v["oVov"])
	ws2s2["oooo"] += -4.0000 * contract("ijAB,ACkm,mBlC->ijkl",t2["ooVV"],t2["VVoo"],v["oVoV"])
	ws2s2["oooo"] +=  4.0000 * contract("imAB,aAjk,lBma->iljk",t2["ooVV"],t2["vVoo"],v["oVov"])
	ws2s2["oooo"] += -4.0000 * contract("imAB,ACjk,lBmC->iljk",t2["ooVV"],t2["VVoo"],v["oVoV"])
	ws2s2["oooo"] += -4.0000 * contract("imAB,ABjn,lnkm->iljk",t2["ooVV"],t2["VVoo"],v["oooo"])
	ws2s2["oooo"] +=  2.0000 * contract("ijaA,bBkl,aAbB->ijkl",t2["oovV"],t2["vVoo"],v["vVvV"])
	ws2s2["oooo"] += -2.0000 * contract("imaA,aAmn,lnjk->iljk",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["oooo"] += -2.0000 * contract("mnaA,aAim,kljn->klij",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["oooo"] +=  1.0000 * contract("ijAB,aCkl,ABaC->ijkl",t2["ooVV"],t2["vVoo"],v["VVvV"])
	ws2s2["oooo"] +=  1.0000 * contract("ijaA,BCkl,aABC->ijkl",t2["oovV"],t2["VVoo"],v["vVVV"])
	ws2s2["oooo"] += -1.0000 * contract("imAB,ABmn,lnjk->iljk",t2["ooVV"],t2["VVoo"],v["oooo"])
	ws2s2["oooo"] += -1.0000 * contract("mnAB,ABim,kljn->klij",t2["ooVV"],t2["VVoo"],v["oooo"])
	ws2s2["oooo"] +=  0.5000 * contract("mnaA,aAij,klmn->klij",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["oooo"] +=  0.5000 * contract("ijAB,CDkl,ABCD->ijkl",t2["ooVV"],t2["VVoo"],v["VVVV"])
	ws2s2["oooo"] +=  0.5000 * contract("ijaA,aAmn,mnkl->ijkl",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["oooo"] +=  0.2500 * contract("mnAB,ABij,klmn->klij",t2["ooVV"],t2["VVoo"],v["oooo"])
	ws2s2["oooo"] +=  0.2500 * contract("ijAB,ABmn,mnkl->ijkl",t2["ooVV"],t2["VVoo"],v["oooo"])
	ws2s2["oooo"] = asym_term(ws2s2["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	ws2s2["ooov"] += -4.0000 * contract("ilaA,bBjl,kAbB->ikja",t2["oovV"],t2["vVoo"],v["oVvV"])
	ws2s2["ooov"] += -4.0000 * contract("ilaA,bAjm,kmlb->ikja",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ooov"] +=  4.0000 * contract("ilaA,ABjm,kmlB->ikja",t2["oovV"],t2["VVoo"],v["oooV"])
	ws2s2["ooov"] +=  4.0000 * contract("ilbA,bBjl,kAaB->ikja",t2["oovV"],t2["vVoo"],v["oVvV"])
	ws2s2["ooov"] +=  4.0000 * contract("ilbA,cAjl,kbac->ikja",t2["oovV"],t2["vVoo"],v["ovvv"])
	ws2s2["ooov"] += -4.0000 * contract("ilbA,ABjl,kbaB->ikja",t2["oovV"],t2["VVoo"],v["ovvV"])
	ws2s2["ooov"] +=  4.0000 * contract("ilbA,bAjm,kmla->ikja",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ooov"] += -4.0000 * contract("ilAB,bAjl,kBab->ikja",t2["ooVV"],t2["vVoo"],v["oVvv"])
	ws2s2["ooov"] +=  4.0000 * contract("ilAB,ACjl,kBaC->ikja",t2["ooVV"],t2["VVoo"],v["oVvV"])
	ws2s2["ooov"] +=  2.0000 * contract("ijaA,bBkl,lAbB->ijka",t2["oovV"],t2["vVoo"],v["oVvV"])
	ws2s2["ooov"] += -2.0000 * contract("ilaA,BCjl,kABC->ikja",t2["oovV"],t2["VVoo"],v["oVVV"])
	ws2s2["ooov"] +=  2.0000 * contract("ilaA,bAlm,kmjb->ikja",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ooov"] += -2.0000 * contract("ilaA,ABlm,kmjB->ikja",t2["oovV"],t2["VVoo"],v["oooV"])
	ws2s2["ooov"] += -2.0000 * contract("lmaA,bAil,jkmb->jkia",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ooov"] +=  2.0000 * contract("lmaA,ABil,jkmB->jkia",t2["oovV"],t2["VVoo"],v["oooV"])
	ws2s2["ooov"] += -2.0000 * contract("ijbA,bBkl,lAaB->ijka",t2["oovV"],t2["vVoo"],v["oVvV"])
	ws2s2["ooov"] += -2.0000 * contract("ijbA,cAkl,lbac->ijka",t2["oovV"],t2["vVoo"],v["ovvv"])
	ws2s2["ooov"] +=  2.0000 * contract("ijbA,ABkl,lbaB->ijka",t2["oovV"],t2["VVoo"],v["ovvV"])
	ws2s2["ooov"] += -2.0000 * contract("ilbA,bAlm,kmja->ikja",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ooov"] +=  2.0000 * contract("ijAB,bAkl,lBab->ijka",t2["ooVV"],t2["vVoo"],v["oVvv"])
	ws2s2["ooov"] += -2.0000 * contract("ijAB,ACkl,lBaC->ijka",t2["ooVV"],t2["VVoo"],v["oVvV"])
	ws2s2["ooov"] +=  2.0000 * contract("ilAB,ABjm,kmla->ikja",t2["ooVV"],t2["VVoo"],v["ooov"])
	ws2s2["ooov"] += -1.0000 * contract("ilAB,ABlm,kmja->ikja",t2["ooVV"],t2["VVoo"],v["ooov"])
	ws2s2["ooov"] +=  1.0000 * contract("lmbA,bAil,jkma->jkia",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ooov"] +=  1.0000 * contract("ijaA,BCkl,lABC->ijka",t2["oovV"],t2["VVoo"],v["oVVV"])
	ws2s2["ooov"] += -0.5000 * contract("ijaA,bAlm,lmkb->ijka",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ooov"] +=  0.5000 * contract("ijaA,ABlm,lmkB->ijka",t2["oovV"],t2["VVoo"],v["oooV"])
	ws2s2["ooov"] += -0.5000 * contract("lmaA,bAlm,jkib->jkia",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ooov"] +=  0.5000 * contract("lmaA,ABlm,jkiB->jkia",t2["oovV"],t2["VVoo"],v["oooV"])
	ws2s2["ooov"] +=  0.5000 * contract("ijbA,bAlm,lmka->ijka",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ooov"] +=  0.5000 * contract("lmAB,ABil,jkma->jkia",t2["ooVV"],t2["VVoo"],v["ooov"])
	ws2s2["ooov"] +=  0.2500 * contract("ijAB,ABlm,lmka->ijka",t2["ooVV"],t2["VVoo"],v["ooov"])
	ws2s2["ooov"] = asym_term(ws2s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	ws2s2["ovoo"] += -4.0000 * contract("ilbA,aBjl,bAkB->iajk",t2["oovV"],t2["vVoo"],v["vVoV"])
	ws2s2["ovoo"] +=  4.0000 * contract("ilbA,bBjl,aAkB->iajk",t2["oovV"],t2["vVoo"],v["vVoV"])
	ws2s2["ovoo"] +=  4.0000 * contract("ilbA,cAjl,abkc->iajk",t2["oovV"],t2["vVoo"],v["vvov"])
	ws2s2["ovoo"] += -4.0000 * contract("ilbA,ABjl,abkB->iajk",t2["oovV"],t2["VVoo"],v["vvoV"])
	ws2s2["ovoo"] += -4.0000 * contract("ilbA,aAjm,mbkl->iajk",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["ovoo"] +=  4.0000 * contract("ilbA,bAjm,makl->iajk",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["ovoo"] += -4.0000 * contract("ilAB,bAjl,aBkb->iajk",t2["ooVV"],t2["vVoo"],v["vVov"])
	ws2s2["ovoo"] +=  4.0000 * contract("ilAB,ACjl,aBkC->iajk",t2["ooVV"],t2["VVoo"],v["vVoV"])
	ws2s2["ovoo"] +=  4.0000 * contract("ilAB,aAjm,mBkl->iajk",t2["ooVV"],t2["vVoo"],v["oVoo"])
	ws2s2["ovoo"] +=  2.0000 * contract("ilbA,aBjk,bAlB->iajk",t2["oovV"],t2["vVoo"],v["vVoV"])
	ws2s2["ovoo"] += -2.0000 * contract("ilbA,bBjk,aAlB->iajk",t2["oovV"],t2["vVoo"],v["vVoV"])
	ws2s2["ovoo"] += -2.0000 * contract("ilbA,cAjk,ablc->iajk",t2["oovV"],t2["vVoo"],v["vvov"])
	ws2s2["ovoo"] +=  2.0000 * contract("ilbA,ABjk,ablB->iajk",t2["oovV"],t2["VVoo"],v["vvoV"])
	ws2s2["ovoo"] += -2.0000 * contract("ilbA,aAlm,mbjk->iajk",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["ovoo"] +=  2.0000 * contract("lmbA,aAil,kbjm->kaij",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["ovoo"] += -2.0000 * contract("lmbA,bAil,kajm->kaij",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["ovoo"] +=  2.0000 * contract("ilAB,bAjk,aBlb->iajk",t2["ooVV"],t2["vVoo"],v["vVov"])
	ws2s2["ovoo"] += -2.0000 * contract("ilAB,ACjk,aBlC->iajk",t2["ooVV"],t2["VVoo"],v["vVoV"])
	ws2s2["ovoo"] += -2.0000 * contract("ilAB,aCjl,ABkC->iajk",t2["ooVV"],t2["vVoo"],v["VVoV"])
	ws2s2["ovoo"] +=  2.0000 * contract("ilAB,ABjm,makl->iajk",t2["ooVV"],t2["VVoo"],v["ovoo"])
	ws2s2["ovoo"] +=  2.0000 * contract("ilAB,aAlm,mBjk->iajk",t2["ooVV"],t2["vVoo"],v["oVoo"])
	ws2s2["ovoo"] += -2.0000 * contract("lmAB,aAil,kBjm->kaij",t2["ooVV"],t2["vVoo"],v["oVoo"])
	ws2s2["ovoo"] +=  1.0000 * contract("ilbA,bAlm,majk->iajk",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["ovoo"] +=  1.0000 * contract("ilAB,aCjk,ABlC->iajk",t2["ooVV"],t2["vVoo"],v["VVoV"])
	ws2s2["ovoo"] += -1.0000 * contract("lmAB,ABil,kajm->kaij",t2["ooVV"],t2["VVoo"],v["ovoo"])
	ws2s2["ovoo"] += -0.5000 * contract("lmbA,aAij,kblm->kaij",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["ovoo"] +=  0.5000 * contract("lmbA,bAij,kalm->kaij",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["ovoo"] += -0.5000 * contract("lmbA,aAlm,kbij->kaij",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["ovoo"] +=  0.5000 * contract("ilAB,ABlm,majk->iajk",t2["ooVV"],t2["VVoo"],v["ovoo"])
	ws2s2["ovoo"] +=  0.5000 * contract("lmAB,aAij,kBlm->kaij",t2["ooVV"],t2["vVoo"],v["oVoo"])
	ws2s2["ovoo"] +=  0.5000 * contract("lmAB,aAlm,kBij->kaij",t2["ooVV"],t2["vVoo"],v["oVoo"])
	ws2s2["ovoo"] +=  0.2500 * contract("lmAB,ABij,kalm->kaij",t2["ooVV"],t2["VVoo"],v["ovoo"])
	ws2s2["ovoo"] = asym_term(ws2s2["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	ws2s2["oovv"] += -4.0000 * contract("ikaA,cAkl,jlbc->ijab",t2["oovV"],t2["vVoo"],v["oovv"])
	ws2s2["oovv"] +=  4.0000 * contract("ikaA,ABkl,jlbB->ijab",t2["oovV"],t2["VVoo"],v["oovV"])
	ws2s2["oovv"] +=  4.0000 * contract("ikaA,jlbB,ABkl->ijab",t2["oovV"],t2["oovV"],v["VVoo"])
	ws2s2["oovv"] += -2.0000 * contract("ijaA,klbB,ABkl->ijab",t2["oovV"],t2["oovV"],v["VVoo"])
	ws2s2["oovv"] += -2.0000 * contract("ikcA,cAkl,jlab->ijab",t2["oovV"],t2["vVoo"],v["oovv"])
	ws2s2["oovv"] +=  1.0000 * contract("ijaA,cAkl,klbc->ijab",t2["oovV"],t2["vVoo"],v["oovv"])
	ws2s2["oovv"] += -1.0000 * contract("ijaA,ABkl,klbB->ijab",t2["oovV"],t2["VVoo"],v["oovV"])
	ws2s2["oovv"] +=  1.0000 * contract("klaA,cAkl,ijbc->ijab",t2["oovV"],t2["vVoo"],v["oovv"])
	ws2s2["oovv"] += -1.0000 * contract("klaA,ABkl,ijbB->ijab",t2["oovV"],t2["VVoo"],v["oovV"])
	ws2s2["oovv"] += -1.0000 * contract("ikAB,ABkl,jlab->ijab",t2["ooVV"],t2["VVoo"],v["oovv"])
	ws2s2["oovv"] +=  0.5000 * contract("ijcA,cAkl,klab->ijab",t2["oovV"],t2["vVoo"],v["oovv"])
	ws2s2["oovv"] +=  0.2500 * contract("ijAB,ABkl,klab->ijab",t2["ooVV"],t2["VVoo"],v["oovv"])
	ws2s2["oovv"] = asym_term(ws2s2["oovv"],"oovv")

	ws2s2["ovov"] += -2.0000 * contract("ikaA,cBjk,bAcB->ibja",t2["oovV"],t2["vVoo"],v["vVvV"])
	ws2s2["ovov"] +=  2.0000 * contract("ikaA,bBjl,lAkB->ibja",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["ovov"] +=  2.0000 * contract("ikaA,cAjl,lbkc->ibja",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["ovov"] += -2.0000 * contract("ikaA,ABjl,lbkB->ibja",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["ovov"] += -2.0000 * contract("ikaA,bBkl,lAjB->ibja",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["ovov"] += -2.0000 * contract("klaA,bBik,jAlB->jbia",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["ovov"] += -2.0000 * contract("klaA,cAik,jblc->jbia",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["ovov"] +=  2.0000 * contract("klaA,ABik,jblB->jbia",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["ovov"] +=  2.0000 * contract("klaA,bAkm,jmil->jbia",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["ovov"] += -2.0000 * contract("ikcA,aBjk,cAbB->iajb",t2["oovV"],t2["vVoo"],v["vVvV"])
	ws2s2["ovov"] +=  2.0000 * contract("ikcA,cBjk,bAaB->ibja",t2["oovV"],t2["vVoo"],v["vVvV"])
	ws2s2["ovov"] +=  2.0000 * contract("ikcA,dAjk,bcad->ibja",t2["oovV"],t2["vVoo"],v["vvvv"])
	ws2s2["ovov"] += -2.0000 * contract("ikcA,ABjk,bcaB->ibja",t2["oovV"],t2["VVoo"],v["vvvV"])
	ws2s2["ovov"] +=  2.0000 * contract("ikcA,aAjl,lckb->iajb",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["ovov"] += -2.0000 * contract("ikcA,cAjl,lbka->ibja",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["ovov"] += -2.0000 * contract("ikcA,aAkl,lcjb->iajb",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["ovov"] += -2.0000 * contract("ikAB,cAjk,bBac->ibja",t2["ooVV"],t2["vVoo"],v["vVvv"])
	ws2s2["ovov"] +=  2.0000 * contract("ikAB,ACjk,bBaC->ibja",t2["ooVV"],t2["VVoo"],v["vVvV"])
	ws2s2["ovov"] += -2.0000 * contract("ikAB,aAjl,lBkb->iajb",t2["ooVV"],t2["vVoo"],v["oVov"])
	ws2s2["ovov"] +=  2.0000 * contract("ikAB,aAkl,lBjb->iajb",t2["ooVV"],t2["vVoo"],v["oVov"])
	ws2s2["ovov"] += -1.0000 * contract("ikaA,BCjk,bABC->ibja",t2["oovV"],t2["VVoo"],v["vVVV"])
	ws2s2["ovov"] += -1.0000 * contract("ikaA,cAkl,lbjc->ibja",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["ovov"] +=  1.0000 * contract("ikaA,ABkl,lbjB->ibja",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["ovov"] += -1.0000 * contract("ikaA,bAlm,lmjk->ibja",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["ovov"] += -1.0000 * contract("klaA,bAim,jmkl->jbia",t2["oovV"],t2["vVoo"],v["oooo"])
	ws2s2["ovov"] += -1.0000 * contract("klaA,bBkl,jAiB->jbia",t2["oovV"],t2["vVoo"],v["oVoV"])
	ws2s2["ovov"] +=  1.0000 * contract("ikcA,cAkl,lbja->ibja",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["ovov"] += -1.0000 * contract("klcA,aAik,jclb->jaib",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["ovov"] +=  1.0000 * contract("klcA,cAik,jbla->jbia",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["ovov"] += -1.0000 * contract("ikAB,aCjk,ABbC->iajb",t2["ooVV"],t2["vVoo"],v["VVvV"])
	ws2s2["ovov"] += -1.0000 * contract("ikAB,ABjl,lbka->ibja",t2["ooVV"],t2["VVoo"],v["ovov"])
	ws2s2["ovov"] +=  1.0000 * contract("klAB,aAik,jBlb->jaib",t2["ooVV"],t2["vVoo"],v["oVov"])
	ws2s2["ovov"] += -0.5000 * contract("klaA,cAkl,jbic->jbia",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["ovov"] +=  0.5000 * contract("klaA,ABkl,jbiB->jbia",t2["oovV"],t2["VVoo"],v["ovoV"])
	ws2s2["ovov"] += -0.5000 * contract("klcA,aAkl,jcib->jaib",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["ovov"] +=  0.5000 * contract("ikAB,ABkl,lbja->ibja",t2["ooVV"],t2["VVoo"],v["ovov"])
	ws2s2["ovov"] +=  0.5000 * contract("klAB,ABik,jbla->jbia",t2["ooVV"],t2["VVoo"],v["ovov"])
	ws2s2["ovov"] +=  0.5000 * contract("klAB,aAkl,jBib->jaib",t2["ooVV"],t2["vVoo"],v["oVov"])

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	ws2s2["vvoo"] +=  4.0000 * contract("aAik,bBjl,klAB->abij",t2["vVoo"],t2["vVoo"],v["ooVV"])
	ws2s2["vvoo"] += -4.0000 * contract("klcA,aAik,bcjl->abij",t2["oovV"],t2["vVoo"],v["vvoo"])
	ws2s2["vvoo"] +=  4.0000 * contract("klAB,aAik,bBjl->abij",t2["ooVV"],t2["vVoo"],v["vVoo"])
	ws2s2["vvoo"] += -2.0000 * contract("aAij,bBkl,klAB->abij",t2["vVoo"],t2["vVoo"],v["ooVV"])
	ws2s2["vvoo"] += -2.0000 * contract("klcA,cAik,abjl->abij",t2["oovV"],t2["vVoo"],v["vvoo"])
	ws2s2["vvoo"] +=  1.0000 * contract("klcA,aAij,bckl->abij",t2["oovV"],t2["vVoo"],v["vvoo"])
	ws2s2["vvoo"] +=  1.0000 * contract("klcA,aAkl,bcij->abij",t2["oovV"],t2["vVoo"],v["vvoo"])
	ws2s2["vvoo"] += -1.0000 * contract("klAB,aAij,bBkl->abij",t2["ooVV"],t2["vVoo"],v["vVoo"])
	ws2s2["vvoo"] += -1.0000 * contract("klAB,ABik,abjl->abij",t2["ooVV"],t2["VVoo"],v["vvoo"])
	ws2s2["vvoo"] += -1.0000 * contract("klAB,aAkl,bBij->abij",t2["ooVV"],t2["vVoo"],v["vVoo"])
	ws2s2["vvoo"] +=  0.5000 * contract("klcA,cAij,abkl->abij",t2["oovV"],t2["vVoo"],v["vvoo"])
	ws2s2["vvoo"] +=  0.2500 * contract("klAB,ABij,abkl->abij",t2["ooVV"],t2["VVoo"],v["vvoo"])
	ws2s2["vvoo"] = asym_term(ws2s2["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	ws2s2["ovvv"] +=  4.0000 * contract("ijaA,bBjk,kAcB->ibac",t2["oovV"],t2["vVoo"],v["oVvV"])
	ws2s2["ovvv"] +=  4.0000 * contract("jkaA,bAjl,ilkc->ibac",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ovvv"] +=  2.0000 * contract("ijaA,dAjk,kcbd->icab",t2["oovV"],t2["vVoo"],v["ovvv"])
	ws2s2["ovvv"] += -2.0000 * contract("ijaA,ABjk,kcbB->icab",t2["oovV"],t2["VVoo"],v["ovvV"])
	ws2s2["ovvv"] += -2.0000 * contract("ijaA,bAkl,kljc->ibac",t2["oovV"],t2["vVoo"],v["ooov"])
	ws2s2["ovvv"] +=  2.0000 * contract("jkaA,bBjk,iAcB->ibac",t2["oovV"],t2["vVoo"],v["oVvV"])
	ws2s2["ovvv"] += -2.0000 * contract("ijdA,aAjk,kdbc->iabc",t2["oovV"],t2["vVoo"],v["ovvv"])
	ws2s2["ovvv"] +=  2.0000 * contract("ijAB,aAjk,kBbc->iabc",t2["ooVV"],t2["vVoo"],v["oVvv"])
	ws2s2["ovvv"] +=  1.0000 * contract("jkaA,dAjk,icbd->icab",t2["oovV"],t2["vVoo"],v["ovvv"])
	ws2s2["ovvv"] += -1.0000 * contract("jkaA,ABjk,icbB->icab",t2["oovV"],t2["VVoo"],v["ovvV"])
	ws2s2["ovvv"] +=  1.0000 * contract("ijdA,dAjk,kcab->icab",t2["oovV"],t2["vVoo"],v["ovvv"])
	ws2s2["ovvv"] += -0.5000 * contract("jkdA,aAjk,idbc->iabc",t2["oovV"],t2["vVoo"],v["ovvv"])
	ws2s2["ovvv"] +=  0.5000 * contract("ijAB,ABjk,kcab->icab",t2["ooVV"],t2["VVoo"],v["ovvv"])
	ws2s2["ovvv"] +=  0.5000 * contract("jkAB,aAjk,iBbc->iabc",t2["ooVV"],t2["vVoo"],v["oVvv"])
	ws2s2["ovvv"] = asym_term(ws2s2["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	ws2s2["vvov"] +=  4.0000 * contract("jkaA,bBij,cAkB->bcia",t2["oovV"],t2["vVoo"],v["vVoV"])
	ws2s2["vvov"] +=  4.0000 * contract("jkaA,bAjl,lcik->bcia",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["vvov"] += -2.0000 * contract("jkaA,dAij,bckd->bcia",t2["oovV"],t2["vVoo"],v["vvov"])
	ws2s2["vvov"] +=  2.0000 * contract("jkaA,ABij,bckB->bcia",t2["oovV"],t2["VVoo"],v["vvoV"])
	ws2s2["vvov"] += -2.0000 * contract("jkaA,bAil,lcjk->bcia",t2["oovV"],t2["vVoo"],v["ovoo"])
	ws2s2["vvov"] +=  2.0000 * contract("jkaA,bBjk,cAiB->bcia",t2["oovV"],t2["vVoo"],v["vVoV"])
	ws2s2["vvov"] +=  2.0000 * contract("jkdA,aAij,cdkb->acib",t2["oovV"],t2["vVoo"],v["vvov"])
	ws2s2["vvov"] += -2.0000 * contract("jkAB,aAij,cBkb->acib",t2["ooVV"],t2["vVoo"],v["vVov"])
	ws2s2["vvov"] +=  1.0000 * contract("jkdA,dAij,bcka->bcia",t2["oovV"],t2["vVoo"],v["vvov"])
	ws2s2["vvov"] +=  1.0000 * contract("jkdA,aAjk,cdib->acib",t2["oovV"],t2["vVoo"],v["vvov"])
	ws2s2["vvov"] += -1.0000 * contract("jkAB,aAjk,cBib->acib",t2["ooVV"],t2["vVoo"],v["vVov"])
	ws2s2["vvov"] += -0.5000 * contract("jkaA,dAjk,bcid->bcia",t2["oovV"],t2["vVoo"],v["vvov"])
	ws2s2["vvov"] +=  0.5000 * contract("jkaA,ABjk,bciB->bcia",t2["oovV"],t2["VVoo"],v["vvoV"])
	ws2s2["vvov"] +=  0.5000 * contract("jkAB,ABij,bcka->bcia",t2["ooVV"],t2["VVoo"],v["vvov"])
	ws2s2["vvov"] = asym_term(ws2s2["vvov"],"vvov")

	# vvvv = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_virt_int))
	ws2s2["vvvv"] +=  8.0000 * contract("ijaA,bAik,kdjc->bdac",t2["oovV"],t2["vVoo"],v["ovov"])
	ws2s2["vvvv"] += -4.0000 * contract("ijaA,bBij,dAcB->bdac",t2["oovV"],t2["vVoo"],v["vVvV"])
	ws2s2["vvvv"] +=  1.0000 * contract("ijaA,eAij,cdbe->cdab",t2["oovV"],t2["vVoo"],v["vvvv"])
	ws2s2["vvvv"] += -1.0000 * contract("ijaA,ABij,cdbB->cdab",t2["oovV"],t2["VVoo"],v["vvvV"])
	ws2s2["vvvv"] +=  1.0000 * contract("ijeA,aAij,debc->adbc",t2["oovV"],t2["vVoo"],v["vvvv"])
	ws2s2["vvvv"] += -1.0000 * contract("ijAB,aAij,dBbc->adbc",t2["ooVV"],t2["vVoo"],v["vVvv"])
	ws2s2["vvvv"] = asym_term(ws2s2["vvvv"],"vvvv")

	if(inc_3_body):
		ws2s2["oooooo"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ))
		ws2s2["ooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["oovooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		ws2s2["oooovv"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["oovoov"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s2["ovvooo"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2s2["ooovvv"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["oovovv"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s2["ovvoov"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s2["vvvooo"] = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2s2["oovvvv"] = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["ovvovv"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s2["vvvoov"] = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s2["ovvvvv"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["vvvovv"] = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_virt_int,n_virt_int))

		# oooooo = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ))
		ws2s2["oooooo"] +=  2.0000 * contract("ijaA,aBkl,nAmB->ijnklm",t2["oovV"],t2["vVoo"],v["oVoV"])
		ws2s2["oooooo"] +=  2.0000 * contract("ijaA,bAkl,namb->ijnklm",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["oooooo"] += -2.0000 * contract("ijaA,ABkl,namB->ijnklm",t2["oovV"],t2["VVoo"],v["ovoV"])
		ws2s2["oooooo"] += -2.0000 * contract("ijAB,aAkl,nBma->ijnklm",t2["ooVV"],t2["vVoo"],v["oVov"])
		ws2s2["oooooo"] +=  2.0000 * contract("ijAB,ACkl,nBmC->ijnklm",t2["ooVV"],t2["VVoo"],v["oVoV"])
		ws2s2["oooooo"] +=  1.0000 * contract("ijaA,aAkp,nplm->ijnklm",t2["oovV"],t2["vVoo"],v["oooo"])
		ws2s2["oooooo"] +=  1.0000 * contract("ipaA,aAjk,mnlp->imnjkl",t2["oovV"],t2["vVoo"],v["oooo"])
		ws2s2["oooooo"] +=  0.5000 * contract("ijAB,ABkp,nplm->ijnklm",t2["ooVV"],t2["VVoo"],v["oooo"])
		ws2s2["oooooo"] +=  0.5000 * contract("ipAB,ABjk,mnlp->imnjkl",t2["ooVV"],t2["VVoo"],v["oooo"])
		ws2s2["oooooo"] = asym_term(ws2s2["oooooo"],"oooooo")

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["ooooov"] += -(2./3.) * contract("ijaA,bBkl,mAbB->ijmkla",t2["oovV"],t2["vVoo"],v["oVvV"])
		ws2s2["ooooov"] += -(2./3.) * contract("ijaA,bAkn,mnlb->ijmkla",t2["oovV"],t2["vVoo"],v["ooov"])
		ws2s2["ooooov"] +=  (2./3.) * contract("ijaA,ABkn,mnlB->ijmkla",t2["oovV"],t2["VVoo"],v["oooV"])
		ws2s2["ooooov"] +=  (2./3.) * contract("inaA,bAjk,lmnb->ilmjka",t2["oovV"],t2["vVoo"],v["ooov"])
		ws2s2["ooooov"] += -(2./3.) * contract("inaA,ABjk,lmnB->ilmjka",t2["oovV"],t2["VVoo"],v["oooV"])
		ws2s2["ooooov"] += -(2./3.) * contract("inaA,bAjn,lmkb->ilmjka",t2["oovV"],t2["vVoo"],v["ooov"])
		ws2s2["ooooov"] +=  (2./3.) * contract("inaA,ABjn,lmkB->ilmjka",t2["oovV"],t2["VVoo"],v["oooV"])
		ws2s2["ooooov"] +=  (2./3.) * contract("ijbA,bBkl,mAaB->ijmkla",t2["oovV"],t2["vVoo"],v["oVvV"])
		ws2s2["ooooov"] +=  (2./3.) * contract("ijbA,cAkl,mbac->ijmkla",t2["oovV"],t2["vVoo"],v["ovvv"])
		ws2s2["ooooov"] += -(2./3.) * contract("ijbA,ABkl,mbaB->ijmkla",t2["oovV"],t2["VVoo"],v["ovvV"])
		ws2s2["ooooov"] +=  (2./3.) * contract("ijbA,bAkn,mnla->ijmkla",t2["oovV"],t2["vVoo"],v["ooov"])
		ws2s2["ooooov"] += -(2./3.) * contract("ijAB,bAkl,mBab->ijmkla",t2["ooVV"],t2["vVoo"],v["oVvv"])
		ws2s2["ooooov"] +=  (2./3.) * contract("ijAB,ACkl,mBaC->ijmkla",t2["ooVV"],t2["VVoo"],v["oVvV"])
		ws2s2["ooooov"] += -(1./3.) * contract("ijaA,BCkl,mABC->ijmkla",t2["oovV"],t2["VVoo"],v["oVVV"])
		ws2s2["ooooov"] += -(1./3.) * contract("inbA,bAjk,lmna->ilmjka",t2["oovV"],t2["vVoo"],v["ooov"])
		ws2s2["ooooov"] +=  (1./3.) * contract("ijAB,ABkn,mnla->ijmkla",t2["ooVV"],t2["VVoo"],v["ooov"])
		ws2s2["ooooov"] += -(1./6.) * contract("inAB,ABjk,lmna->ilmjka",t2["ooVV"],t2["VVoo"],v["ooov"])
		ws2s2["ooooov"] = asym_term(ws2s2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		ws2s2["oovooo"] += -(2./3.) * contract("ijbA,aBkl,bAmB->ijaklm",t2["oovV"],t2["vVoo"],v["vVoV"])
		ws2s2["oovooo"] +=  (2./3.) * contract("ijbA,bBkl,aAmB->ijaklm",t2["oovV"],t2["vVoo"],v["vVoV"])
		ws2s2["oovooo"] +=  (2./3.) * contract("ijbA,cAkl,abmc->ijaklm",t2["oovV"],t2["vVoo"],v["vvov"])
		ws2s2["oovooo"] += -(2./3.) * contract("ijbA,ABkl,abmB->ijaklm",t2["oovV"],t2["VVoo"],v["vvoV"])
		ws2s2["oovooo"] +=  (2./3.) * contract("ijbA,aAkn,nblm->ijaklm",t2["oovV"],t2["vVoo"],v["ovoo"])
		ws2s2["oovooo"] += -(2./3.) * contract("inbA,aAjk,mbln->imajkl",t2["oovV"],t2["vVoo"],v["ovoo"])
		ws2s2["oovooo"] +=  (2./3.) * contract("inbA,bAjk,maln->imajkl",t2["oovV"],t2["vVoo"],v["ovoo"])
		ws2s2["oovooo"] += -(2./3.) * contract("inbA,aAjn,mbkl->imajkl",t2["oovV"],t2["vVoo"],v["ovoo"])
		ws2s2["oovooo"] += -(2./3.) * contract("ijAB,bAkl,aBmb->ijaklm",t2["ooVV"],t2["vVoo"],v["vVov"])
		ws2s2["oovooo"] +=  (2./3.) * contract("ijAB,ACkl,aBmC->ijaklm",t2["ooVV"],t2["VVoo"],v["vVoV"])
		ws2s2["oovooo"] += -(2./3.) * contract("ijAB,aAkn,nBlm->ijaklm",t2["ooVV"],t2["vVoo"],v["oVoo"])
		ws2s2["oovooo"] +=  (2./3.) * contract("inAB,aAjk,mBln->imajkl",t2["ooVV"],t2["vVoo"],v["oVoo"])
		ws2s2["oovooo"] +=  (2./3.) * contract("inAB,aAjn,mBkl->imajkl",t2["ooVV"],t2["vVoo"],v["oVoo"])
		ws2s2["oovooo"] += -(1./3.) * contract("ijbA,bAkn,nalm->ijaklm",t2["oovV"],t2["vVoo"],v["ovoo"])
		ws2s2["oovooo"] += -(1./3.) * contract("ijAB,aCkl,ABmC->ijaklm",t2["ooVV"],t2["vVoo"],v["VVoV"])
		ws2s2["oovooo"] +=  (1./3.) * contract("inAB,ABjk,maln->imajkl",t2["ooVV"],t2["VVoo"],v["ovoo"])
		ws2s2["oovooo"] += -(1./6.) * contract("ijAB,ABkn,nalm->ijaklm",t2["ooVV"],t2["VVoo"],v["ovoo"])
		ws2s2["oovooo"] = asym_term(ws2s2["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["oooovv"] += -(4./3.) * contract("ijaA,kmbB,ABlm->ijklab",t2["oovV"],t2["oovV"],v["VVoo"])
		ws2s2["oooovv"] +=  (2./3.) * contract("ijaA,cAkm,lmbc->ijlkab",t2["oovV"],t2["vVoo"],v["oovv"])
		ws2s2["oooovv"] += -(2./3.) * contract("ijaA,ABkm,lmbB->ijlkab",t2["oovV"],t2["VVoo"],v["oovV"])
		ws2s2["oooovv"] +=  (2./3.) * contract("imaA,cAjm,klbc->ikljab",t2["oovV"],t2["vVoo"],v["oovv"])
		ws2s2["oooovv"] += -(2./3.) * contract("imaA,ABjm,klbB->ikljab",t2["oovV"],t2["VVoo"],v["oovV"])
		ws2s2["oooovv"] +=  (1./3.) * contract("ijcA,cAkm,lmab->ijlkab",t2["oovV"],t2["vVoo"],v["oovv"])
		ws2s2["oooovv"] +=  (1./6.) * contract("ijAB,ABkm,lmab->ijlkab",t2["ooVV"],t2["VVoo"],v["oovv"])
		ws2s2["oooovv"] = asym_term(ws2s2["oooovv"],"oooovv")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s2["oovoov"] += -(8./9.) * contract("imaA,bBjm,lAkB->ilbjka",t2["oovV"],t2["vVoo"],v["oVoV"])
		ws2s2["oovoov"] +=  (8./9.) * contract("imaA,bAjn,lnkm->ilbjka",t2["oovV"],t2["vVoo"],v["oooo"])
		ws2s2["oovoov"] +=  (4./9.) * contract("ijaA,bBkm,mAlB->ijbkla",t2["oovV"],t2["vVoo"],v["oVoV"])
		ws2s2["oovoov"] +=  (4./9.) * contract("imaA,bBjk,lAmB->ilbjka",t2["oovV"],t2["vVoo"],v["oVoV"])
		ws2s2["oovoov"] +=  (4./9.) * contract("imaA,cAjk,lbmc->ilbjka",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["oovoov"] += -(4./9.) * contract("imaA,ABjk,lbmB->ilbjka",t2["oovV"],t2["VVoo"],v["ovoV"])
		ws2s2["oovoov"] += -(4./9.) * contract("imaA,cAjm,lbkc->ilbjka",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["oovoov"] +=  (4./9.) * contract("imaA,ABjm,lbkB->ilbjka",t2["oovV"],t2["VVoo"],v["ovoV"])
		ws2s2["oovoov"] +=  (4./9.) * contract("ijcA,aAkm,mclb->ijaklb",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["oovoov"] += -(4./9.) * contract("imcA,aAjm,lckb->ilajkb",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["oovoov"] += -(4./9.) * contract("ijAB,aAkm,mBlb->ijaklb",t2["ooVV"],t2["vVoo"],v["oVov"])
		ws2s2["oovoov"] +=  (4./9.) * contract("imAB,aAjm,lBkb->ilajkb",t2["ooVV"],t2["vVoo"],v["oVov"])
		ws2s2["oovoov"] += -(2./9.) * contract("ijaA,cBkl,bAcB->ijbkla",t2["oovV"],t2["vVoo"],v["vVvV"])
		ws2s2["oovoov"] +=  (2./9.) * contract("ijaA,cAkm,mblc->ijbkla",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["oovoov"] += -(2./9.) * contract("ijaA,ABkm,mblB->ijbkla",t2["oovV"],t2["VVoo"],v["ovoV"])
		ws2s2["oovoov"] +=  (2./9.) * contract("imaA,bAmn,lnjk->ilbjka",t2["oovV"],t2["vVoo"],v["oooo"])
		ws2s2["oovoov"] +=  (2./9.) * contract("mnaA,bAim,kljn->klbija",t2["oovV"],t2["vVoo"],v["oooo"])
		ws2s2["oovoov"] += -(2./9.) * contract("ijcA,aBkl,cAbB->ijaklb",t2["oovV"],t2["vVoo"],v["vVvV"])
		ws2s2["oovoov"] +=  (2./9.) * contract("ijcA,cBkl,bAaB->ijbkla",t2["oovV"],t2["vVoo"],v["vVvV"])
		ws2s2["oovoov"] +=  (2./9.) * contract("ijcA,dAkl,bcad->ijbkla",t2["oovV"],t2["vVoo"],v["vvvv"])
		ws2s2["oovoov"] += -(2./9.) * contract("ijcA,ABkl,bcaB->ijbkla",t2["oovV"],t2["VVoo"],v["vvvV"])
		ws2s2["oovoov"] += -(2./9.) * contract("ijcA,cAkm,mbla->ijbkla",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["oovoov"] +=  (2./9.) * contract("imcA,aAjk,lcmb->ilajkb",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["oovoov"] += -(2./9.) * contract("imcA,cAjk,lbma->ilbjka",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["oovoov"] += -(2./9.) * contract("ijAB,cAkl,bBac->ijbkla",t2["ooVV"],t2["vVoo"],v["vVvv"])
		ws2s2["oovoov"] +=  (2./9.) * contract("ijAB,ACkl,bBaC->ijbkla",t2["ooVV"],t2["VVoo"],v["vVvV"])
		ws2s2["oovoov"] += -(2./9.) * contract("imAB,aAjk,lBmb->ilajkb",t2["ooVV"],t2["vVoo"],v["oVov"])
		ws2s2["oovoov"] += -(1./9.) * contract("ijAB,ABkm,mbla->ijbkla",t2["ooVV"],t2["VVoo"],v["ovov"])
		ws2s2["oovoov"] += -(1./9.) * contract("ijaA,BCkl,bABC->ijbkla",t2["oovV"],t2["VVoo"],v["vVVV"])
		ws2s2["oovoov"] += -(1./9.) * contract("ijAB,aCkl,ABbC->ijaklb",t2["ooVV"],t2["vVoo"],v["VVvV"])
		ws2s2["oovoov"] += -(1./9.) * contract("imAB,ABjk,lbma->ilbjka",t2["ooVV"],t2["VVoo"],v["ovov"])
		ws2s2["oovoov"] += -(1./18.) * contract("ijaA,bAmn,mnkl->ijbkla",t2["oovV"],t2["vVoo"],v["oooo"])
		ws2s2["oovoov"] += -(1./18.) * contract("mnaA,bAij,klmn->klbija",t2["oovV"],t2["vVoo"],v["oooo"])
		ws2s2["oovoov"] = asym_term(ws2s2["oovoov"],"oovoov")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2s2["ovvooo"] += -(4./3.) * contract("aAij,bBkm,lmAB->labijk",t2["vVoo"],t2["vVoo"],v["ooVV"])
		ws2s2["ovvooo"] +=  (2./3.) * contract("imcA,aAjk,bclm->iabjkl",t2["oovV"],t2["vVoo"],v["vvoo"])
		ws2s2["ovvooo"] +=  (2./3.) * contract("imcA,aAjm,bckl->iabjkl",t2["oovV"],t2["vVoo"],v["vvoo"])
		ws2s2["ovvooo"] += -(2./3.) * contract("imAB,aAjk,bBlm->iabjkl",t2["ooVV"],t2["vVoo"],v["vVoo"])
		ws2s2["ovvooo"] += -(2./3.) * contract("imAB,aAjm,bBkl->iabjkl",t2["ooVV"],t2["vVoo"],v["vVoo"])
		ws2s2["ovvooo"] +=  (1./3.) * contract("imcA,cAjk,ablm->iabjkl",t2["oovV"],t2["vVoo"],v["vvoo"])
		ws2s2["ovvooo"] +=  (1./6.) * contract("imAB,ABjk,ablm->iabjkl",t2["ooVV"],t2["VVoo"],v["vvoo"])
		ws2s2["ovvooo"] = asym_term(ws2s2["ovvooo"],"ovvooo")

		# ooovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["ooovvv"] += 4.0000 * contract("ijaA,klbB,ABlc->ijkabc",t2["oovV"],t2["oovV"],v["VVov"])
		ws2s2["ooovvv"] = asym_term(ws2s2["ooovvv"],"ooovvv")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s2["oovovv"] +=  (8./9.) * contract("ilaA,bBjl,kAcB->ikbjac",t2["oovV"],t2["vVoo"],v["oVvV"])
		ws2s2["oovovv"] +=  (8./9.) * contract("ilaA,bAjm,kmlc->ikbjac",t2["oovV"],t2["vVoo"],v["ooov"])
		ws2s2["oovovv"] += -(4./9.) * contract("ijaA,bBkl,lAcB->ijbkac",t2["oovV"],t2["vVoo"],v["oVvV"])
		ws2s2["oovovv"] +=  (4./9.) * contract("ilaA,dAjl,kcbd->ikcjab",t2["oovV"],t2["vVoo"],v["ovvv"])
		ws2s2["oovovv"] += -(4./9.) * contract("ilaA,ABjl,kcbB->ikcjab",t2["oovV"],t2["VVoo"],v["ovvV"])
		ws2s2["oovovv"] += -(4./9.) * contract("ilaA,bAlm,kmjc->ikbjac",t2["oovV"],t2["vVoo"],v["ooov"])
		ws2s2["oovovv"] += -(2./9.) * contract("ijaA,dAkl,lcbd->ijckab",t2["oovV"],t2["vVoo"],v["ovvv"])
		ws2s2["oovovv"] +=  (2./9.) * contract("ijaA,ABkl,lcbB->ijckab",t2["oovV"],t2["VVoo"],v["ovvV"])
		ws2s2["oovovv"] +=  (2./9.) * contract("lmaA,bAil,jkmc->jkbiac",t2["oovV"],t2["vVoo"],v["ooov"])
		ws2s2["oovovv"] +=  (2./9.) * contract("ijdA,aAkl,ldbc->ijakbc",t2["oovV"],t2["vVoo"],v["ovvv"])
		ws2s2["oovovv"] += -(2./9.) * contract("ildA,aAjl,kdbc->ikajbc",t2["oovV"],t2["vVoo"],v["ovvv"])
		ws2s2["oovovv"] += -(2./9.) * contract("ijAB,aAkl,lBbc->ijakbc",t2["ooVV"],t2["vVoo"],v["oVvv"])
		ws2s2["oovovv"] +=  (2./9.) * contract("ilAB,aAjl,kBbc->ikajbc",t2["ooVV"],t2["vVoo"],v["oVvv"])
		ws2s2["oovovv"] +=  (1./9.) * contract("ijaA,bAlm,lmkc->ijbkac",t2["oovV"],t2["vVoo"],v["ooov"])
		ws2s2["oovovv"] += -(1./9.) * contract("ijdA,dAkl,lcab->ijckab",t2["oovV"],t2["vVoo"],v["ovvv"])
		ws2s2["oovovv"] += -(1./18.) * contract("ijAB,ABkl,lcab->ijckab",t2["ooVV"],t2["VVoo"],v["ovvv"])
		ws2s2["oovovv"] = asym_term(ws2s2["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s2["ovvoov"] +=  (8./9.) * contract("ilaA,bBjl,cAkB->ibcjka",t2["oovV"],t2["vVoo"],v["vVoV"])
		ws2s2["ovvoov"] +=  (8./9.) * contract("ilaA,bAjm,mckl->ibcjka",t2["oovV"],t2["vVoo"],v["ovoo"])
		ws2s2["ovvoov"] += -(4./9.) * contract("ilaA,bBjk,cAlB->ibcjka",t2["oovV"],t2["vVoo"],v["vVoV"])
		ws2s2["ovvoov"] += -(4./9.) * contract("lmaA,bAil,kcjm->kbcija",t2["oovV"],t2["vVoo"],v["ovoo"])
		ws2s2["ovvoov"] +=  (4./9.) * contract("ildA,aAjl,cdkb->iacjkb",t2["oovV"],t2["vVoo"],v["vvov"])
		ws2s2["ovvoov"] += -(4./9.) * contract("ilAB,aAjl,cBkb->iacjkb",t2["ooVV"],t2["vVoo"],v["vVov"])
		ws2s2["ovvoov"] +=  (2./9.) * contract("ilaA,dAjk,bcld->ibcjka",t2["oovV"],t2["vVoo"],v["vvov"])
		ws2s2["ovvoov"] += -(2./9.) * contract("ilaA,ABjk,bclB->ibcjka",t2["oovV"],t2["VVoo"],v["vvoV"])
		ws2s2["ovvoov"] += -(2./9.) * contract("ilaA,dAjl,bckd->ibcjka",t2["oovV"],t2["vVoo"],v["vvov"])
		ws2s2["ovvoov"] +=  (2./9.) * contract("ilaA,ABjl,bckB->ibcjka",t2["oovV"],t2["VVoo"],v["vvoV"])
		ws2s2["ovvoov"] +=  (2./9.) * contract("ilaA,bAlm,mcjk->ibcjka",t2["oovV"],t2["vVoo"],v["ovoo"])
		ws2s2["ovvoov"] += -(2./9.) * contract("ildA,aAjk,cdlb->iacjkb",t2["oovV"],t2["vVoo"],v["vvov"])
		ws2s2["ovvoov"] +=  (2./9.) * contract("ilAB,aAjk,cBlb->iacjkb",t2["ooVV"],t2["vVoo"],v["vVov"])
		ws2s2["ovvoov"] +=  (1./9.) * contract("lmaA,bAij,kclm->kbcija",t2["oovV"],t2["vVoo"],v["ovoo"])
		ws2s2["ovvoov"] += -(1./9.) * contract("ildA,dAjk,bcla->ibcjka",t2["oovV"],t2["vVoo"],v["vvov"])
		ws2s2["ovvoov"] += -(1./18.) * contract("ilAB,ABjk,bcla->ibcjka",t2["ooVV"],t2["VVoo"],v["vvov"])
		ws2s2["ovvoov"] = asym_term(ws2s2["ovvoov"],"ovvoov")

		# vvvooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2s2["vvvooo"] +=  4.0000 * contract("aAij,bBkl,lcAB->abcijk",t2["vVoo"],t2["vVoo"],v["ovVV"])
		ws2s2["vvvooo"] = asym_term(ws2s2["vvvooo"],"vvvooo")

		# oovvvv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["oovvvv"] +=  (2./3.) * contract("ikaA,bAkl,jlcd->ijbacd",t2["oovV"],t2["vVoo"],v["oovv"])
		ws2s2["oovvvv"] += -(1./6.) * contract("ijaA,bAkl,klcd->ijbacd",t2["oovV"],t2["vVoo"],v["oovv"])
		ws2s2["oovvvv"] = asym_term(ws2s2["oovvvv"],"oovvvv")

		# ovvovv = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s2["ovvovv"] += -(8./9.) * contract("ikaA,bBjk,dAcB->ibdjac",t2["oovV"],t2["vVoo"],v["vVvV"])
		ws2s2["ovvovv"] +=  (8./9.) * contract("ikaA,bAjl,ldkc->ibdjac",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["ovvovv"] += -(4./9.) * contract("ikaA,bAkl,ldjc->ibdjac",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["ovvovv"] += -(4./9.) * contract("klaA,bAik,jdlc->jbdiac",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["ovvovv"] +=  (2./9.) * contract("ikaA,eAjk,cdbe->icdjab",t2["oovV"],t2["vVoo"],v["vvvv"])
		ws2s2["ovvovv"] += -(2./9.) * contract("ikaA,ABjk,cdbB->icdjab",t2["oovV"],t2["VVoo"],v["vvvV"])
		ws2s2["ovvovv"] +=  (2./9.) * contract("ikeA,aAjk,debc->iadjbc",t2["oovV"],t2["vVoo"],v["vvvv"])
		ws2s2["ovvovv"] += -(2./9.) * contract("ikAB,aAjk,dBbc->iadjbc",t2["ooVV"],t2["vVoo"],v["vVvv"])
		ws2s2["ovvovv"] = asym_term(ws2s2["ovvovv"],"ovvovv")

		#vvvoov = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s2["vvvoov"] +=  (2./3.) * contract("klaA,bAik,cdjl->bcdija",t2["oovV"],t2["vVoo"],v["vvoo"])
		ws2s2["vvvoov"] += -(1./6.) * contract("klaA,bAij,cdkl->bcdija",t2["oovV"],t2["vVoo"],v["vvoo"])
		ws2s2["vvvoov"] = asym_term(ws2s2["vvvoov"],"vvvoov")

		# ovvvvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["ovvvvv"] +=  (2./3.) * contract("ijaA,bAjk,kecd->ibeacd",t2["oovV"],t2["vVoo"],v["ovvv"])
		ws2s2["ovvvvv"] = asym_term(ws2s2["ovvvvv"],"ovvvvv") 

		# vvvovv = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s2["vvvovv"] +=  (2./3.) * contract("jkaA,bAij,dekc->bdeiac",t2["oovV"],t2["vVoo"],v["vvov"])
		ws2s2["vvvovv"] = asym_term(ws2s2["vvvovv"],"vvvovv")

	if(inc_4_body):
		ws2s2["ooooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["ooovoooo"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		ws2s2["oooooovv"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["ooovooov"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["oovvoooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		ws2s2["ooooovvv"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["ooovoovv"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["oovvooov"] = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["ovvvoooo"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		ws2s2["oooovvvv"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["ooovovvv"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["oovvoovv"] = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["ovvvooov"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["vvvvoooo"] = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		ws2s2["oovvovvv"] = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["ovvvoovv"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int,n_virt_int))

		# ooooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["ooooooov"] += -0.1250 * contract("ijaA,bAkl,npmb->ijnpklma",t2["oovV"],t2["vVoo"],v["ooov"])
		ws2s2["ooooooov"] +=  0.1250 * contract("ijaA,ABkl,npmB->ijnpklma",t2["oovV"],t2["VVoo"],v["oooV"])
		ws2s2["ooooooov"] = asym_term(ws2s2["ooooooov"],"ooooooov")

		# ooovoooo = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		ws2s2["ooovoooo"] += -0.1250 * contract("ijbA,aAkl,pbmn->ijpaklmn",t2["oovV"],t2["vVoo"],v["ovoo"])
		ws2s2["ooovoooo"] +=  0.1250 * contract("ijAB,aAkl,pBmn->ijpaklmn",t2["ooVV"],t2["vVoo"],v["oVoo"])
		ws2s2["ooovoooo"] = asym_term(ws2s2["ooovoooo"],"ooovoooo")

		# oooooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["oooooovv"] +=  (1./12.) * contract("ijaA,cAkl,mnbc->ijmnklab",t2["oovV"],t2["vVoo"],v["oovv"])
		ws2s2["oooooovv"] += -(1./12.) * contract("ijaA,ABkl,mnbB->ijmnklab",t2["oovV"],t2["VVoo"],v["oovV"])
		ws2s2["oooooovv"] += -(1./12.) * contract("ijaA,klbB,ABmn->ijklmnab",t2["oovV"],t2["oovV"],v["VVoo"])
		ws2s2["oooooovv"] = asym_term(ws2s2["oooooovv"],"oooooovv")

		# ooovooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["ooovooov"] += -0.1250 * contract("ijaA,bBkl,nAmB->ijnbklma",t2["oovV"],t2["vVoo"],v["oVoV"])
		ws2s2["ooovooov"] += -0.0625 * contract("ijaA,cAkl,nbmc->ijnbklma",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["ooovooov"] +=  0.0625 * contract("ijaA,ABkl,nbmB->ijnbklma",t2["oovV"],t2["VVoo"],v["ovoV"])
		ws2s2["ooovooov"] += -0.0625 * contract("ijaA,bAkp,nplm->ijnbklma",t2["oovV"],t2["vVoo"],v["oooo"])
		ws2s2["ooovooov"] += -0.0625 * contract("ipaA,bAjk,mnlp->imnbjkla",t2["oovV"],t2["vVoo"],v["oooo"])
		ws2s2["ooovooov"] += -0.0625 * contract("ijcA,aAkl,ncmb->ijnaklmb",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["ooovooov"] +=  0.0625 * contract("ijAB,aAkl,nBmb->ijnaklmb",t2["ooVV"],t2["vVoo"],v["oVov"])
		ws2s2["ooovooov"] = asym_term(ws2s2["ooovooov"],"ooovooov")

		# oovvoooo = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		ws2s2["oovvoooo"] += -(1./12.) * contract("aAij,bBkl,mnAB->mnabijkl",t2["vVoo"],t2["vVoo"],v["ooVV"])
		ws2s2["oovvoooo"] +=  (1./12.) * contract("ijcA,aAkl,bcmn->ijabklmn",t2["oovV"],t2["vVoo"],v["vvoo"])
		ws2s2["oovvoooo"] += -(1./12.) * contract("ijAB,aAkl,bBmn->ijabklmn",t2["ooVV"],t2["vVoo"],v["vVoo"])
		ws2s2["oovvoooo"] = asym_term(ws2s2["oovvoooo"],"oovvoooo")

		# ooooovvv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["ooooovvv"] += -0.2500 * contract("ijaA,klbB,ABmc->ijklmabc",t2["oovV"],t2["oovV"],v["VVov"])
		ws2s2["ooooovvv"] = asym_term(ws2s2["ooooovvv"],"ooooovvv")

		# ooovoovv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["ooovoovv"] +=  (1./12.) * contract("ijaA,bBkl,mAcB->ijmbklac",t2["oovV"],t2["vVoo"],v["oVvV"])
		ws2s2["ooovoovv"] +=  (1./12.) * contract("ijaA,bAkn,mnlc->ijmbklac",t2["oovV"],t2["vVoo"],v["ooov"])
		ws2s2["ooovoovv"] +=  (1./24.) * contract("ijaA,dAkl,mcbd->ijmcklab",t2["oovV"],t2["vVoo"],v["ovvv"])
		ws2s2["ooovoovv"] += -(1./24.) * contract("ijaA,ABkl,mcbB->ijmcklab",t2["oovV"],t2["VVoo"],v["ovvV"])
		ws2s2["ooovoovv"] += -(1./24.) * contract("inaA,bAjk,lmnc->ilmbjkac",t2["oovV"],t2["vVoo"],v["ooov"])
		ws2s2["ooovoovv"] += -(1./48.) * contract("ijdA,aAkl,mdbc->ijmaklbc",t2["oovV"],t2["vVoo"],v["ovvv"])
		ws2s2["ooovoovv"] +=  (1./48.) * contract("ijAB,aAkl,mBbc->ijmaklbc",t2["ooVV"],t2["vVoo"],v["oVvv"])
		ws2s2["ooovoovv"] = asym_term(ws2s2["ooovoovv"],"ooovoovv")

		# oovvooov = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["oovvooov"] +=  (1./12.) * contract("ijaA,bBkl,cAmB->ijbcklma",t2["oovV"],t2["vVoo"],v["vVoV"])
		ws2s2["oovvooov"] +=  (1./12.) * contract("inaA,bAjk,mcln->imbcjkla",t2["oovV"],t2["vVoo"],v["ovoo"])
		ws2s2["oovvooov"] += -(1./24.) * contract("ijaA,bAkn,nclm->ijbcklma",t2["oovV"],t2["vVoo"],v["ovoo"])
		ws2s2["oovvooov"] +=  (1./24.) * contract("ijdA,aAkl,cdmb->ijacklmb",t2["oovV"],t2["vVoo"],v["vvov"])
		ws2s2["oovvooov"] += -(1./24.) * contract("ijAB,aAkl,cBmb->ijacklmb",t2["ooVV"],t2["vVoo"],v["vVov"])
		ws2s2["oovvooov"] += -(1./48.) * contract("ijaA,dAkl,bcmd->ijbcklma",t2["oovV"],t2["vVoo"],v["vvov"])
		ws2s2["oovvooov"] +=  (1./48.) * contract("ijaA,ABkl,bcmB->ijbcklma",t2["oovV"],t2["VVoo"],v["vvoV"])
		ws2s2["oovvooov"] = asym_term(ws2s2["oovvooov"],"oovvooov")

		# ovvvoooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		ws2s2["ovvvoooo"] += -0.2500 * contract("aAij,bBkl,mcAB->mabcijkl",t2["vVoo"],t2["vVoo"],v["ovVV"])
		ws2s2["ovvvoooo"] = asym_term(ws2s2["ovvvoooo"],"ovvvoooo")

		# oooovvvv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["oooovvvv"] += -0.5000 * contract("ijaA,klbB,ABcd->ijklabcd",t2["oovV"],t2["oovV"],v["VVvv"])
		ws2s2["oooovvvv"] = asym_term(ws2s2["oooovvvv"],"oooovvvv")

		# ooovovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["ooovovvv"] += -0.0625 * contract("ijaA,bAkm,lmcd->ijlbkacd",t2["oovV"],t2["vVoo"],v["oovv"])
		ws2s2["ooovovvv"] = asym_term(ws2s2["ooovovvv"],"ooovovvv")

		# oovvoovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["oovvoovv"] += -(1./18.) * contract("ijaA,bBkl,dAcB->ijbdklac",t2["oovV"],t2["vVoo"],v["vVvV"])
		ws2s2["oovvoovv"] +=  (1./18.) * contract("ijaA,bAkm,mdlc->ijbdklac",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["oovvoovv"] +=  (1./18.) * contract("imaA,bAjk,ldmc->ilbdjkac",t2["oovV"],t2["vVoo"],v["ovov"])
		ws2s2["oovvoovv"] +=  (1./72.) * contract("ijaA,eAkl,cdbe->ijcdklab",t2["oovV"],t2["vVoo"],v["vvvv"])
		ws2s2["oovvoovv"] += -(1./72.) * contract("ijaA,ABkl,cdbB->ijcdklab",t2["oovV"],t2["VVoo"],v["vvvV"])
		ws2s2["oovvoovv"] +=  (1./72.) * contract("ijeA,aAkl,debc->ijadklbc",t2["oovV"],t2["vVoo"],v["vvvv"])
		ws2s2["oovvoovv"] += -(1./72.) * contract("ijAB,aAkl,dBbc->ijadklbc",t2["ooVV"],t2["vVoo"],v["vVvv"])
		ws2s2["oovvoovv"] = asym_term(ws2s2["oovvoovv"],"oovvoovv")

		# ovvvooov = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["ovvvooov"] += -0.0625 * contract("imaA,bAjk,cdlm->ibcdjkla",t2["oovV"],t2["vVoo"],v["vvoo"])
		ws2s2["ovvvooov"] = asym_term(ws2s2["ovvvooov"],"ovvvooov")

		# vvvvoooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		ws2s2["vvvvoooo"] += -0.5000 * contract("aAij,bBkl,cdAB->abcdijkl",t2["vVoo"],t2["vVoo"],v["vvVV"])
		ws2s2["vvvvoooo"] = asym_term(ws2s2["vvvvoooo"],"vvvvoooo")

		# oovvovvv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["oovvovvv"] += -(1./24.) * contract("ijaA,bAkl,lecd->ijbekacd",t2["oovV"],t2["vVoo"],v["ovvv"])
		ws2s2["oovvovvv"] = asym_term(ws2s2["oovvovvv"],"oovvovvv")

		# ovvvoovv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["ovvvoovv"] += -(1./24.) * contract("ilaA,bAjk,delc->ibdejkac",t2["oovV"],t2["vVoo"],v["vvov"])
		ws2s2["ovvvoovv"] = asym_term(ws2s2["ovvvoovv"],"ovvvoovv")

	return ws2s2 

def fn_s1_s1_s1(f,t1):
	# [[[Fn,S_1ext],S_1ext],S_1ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize
	fs1s1s1 = {
		"c": 0.0,
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ))
	}
	# Populate [[[Fn,S_1ext],S_1ext],S_1ext]
	fs1s1s1["c"] += -4.000 * contract("Ai,iB,jA,Bj->",f["Vo"],t1["oV"],t1["oV"],t1["Vo"])
	fs1s1s1["c"] += -4.000 * contract("iA,jB,Aj,Bi->",f["oV"],t1["oV"],t1["Vo"],t1["Vo"])

	fs1s1s1["oo"] += -1.000 * contract("Ai,jB,kA,Bk->ji",f["Vo"],t1["oV"],t1["oV"],t1["Vo"])
	fs1s1s1["oo"] += -3.000 * contract("Ak,kB,iA,Bj->ij",f["Vo"],t1["oV"],t1["oV"],t1["Vo"])
	fs1s1s1["oo"] += -1.000 * contract("iA,kB,Bj,Ak->ij",f["oV"],t1["oV"],t1["Vo"],t1["Vo"])
	fs1s1s1["oo"] += -3.000 * contract("kA,iB,Aj,Bk->ij",f["oV"],t1["oV"],t1["Vo"],t1["Vo"])

	fs1s1s1["ov"] += -1.000 * contract("Aa,iB,jA,Bj->ia",f["Vv"],t1["oV"],t1["oV"],t1["Vo"])

	fs1s1s1["vo"] += -1.000 * contract("aA,jB,Bi,Aj->ai",f["vV"],t1["oV"],t1["Vo"],t1["Vo"])

	return fs1s1s1

def fn_s1_s1_s2(f,t1,t2):
	# [[[Fn,S_1ext],S_1ext],S_2ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize
	fs1s1s2 = {
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	} 
	# Populate [[[Fn,S_1ext],S_1ext],S_2ext]
	fs1s1s2["ov"] += -2.000 * contract("jA,Ak,Bj,ikaB->ia",f["oV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs1s1s2["ov"] += -1.000 * contract("Aj,kA,Bk,ijaB->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs1s1s2["ov"] += -1.000 * contract("Aj,jB,Bk,ikaA->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])

	fs1s1s2["vo"] += -2.000 * contract("Aj,jB,kA,aBik->ai",f["Vo"],t1["oV"],t1["oV"],t2["vVoo"])
	fs1s1s2["vo"] += -1.000 * contract("jA,kB,Bj,aAik->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs1s1s2["vo"] += -1.000 * contract("jA,kB,Ak,aBij->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])

	fs1s1s2["ooov"] +=  0.500 * contract("lA,Ai,Bl,jkaB->jkia",f["oV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs1s1s2["ooov"] +=  0.250 * contract("Ai,lA,Bl,jkaB->jkia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs1s1s2["ooov"] +=  0.250 * contract("Al,lB,Bi,jkaA->jkia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])

	fs1s1s2["ovoo"] +=  0.500 * contract("Al,lB,iA,aBjk->iajk",f["Vo"],t1["oV"],t1["oV"],t2["vVoo"])
	fs1s1s2["ovoo"] +=  0.250 * contract("iA,lB,Al,aBjk->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs1s1s2["ovoo"] +=  0.250 * contract("lA,iB,Bl,aAjk->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])

	fs1s1s2["oovv"] +=  0.250 * contract("Aa,kA,Bk,ijbB->ijab",f["Vv"],t1["oV"],t1["Vo"],t2["oovV"])
	fs1s1s2["oovv"] += -0.250 * contract("Aa,kA,Bk,ijbB->ijba",f["Vv"],t1["oV"],t1["Vo"],t2["oovV"])

	fs1s1s2["vvoo"] +=  0.250 * contract("aA,kB,Ak,bBij->abij",f["vV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs1s1s2["vvoo"] += -0.250 * contract("aA,kB,Ak,bBij->baij",f["vV"],t1["oV"],t1["Vo"],t2["vVoo"])

	return fs1s1s2 

def fn_s1_s2_s1(f,t1,t2):
	# [[[Fn,S_1ext],S_2ext],S_1ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize
	fs1s2s1 = {
		"c": 0.0,
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"vv": np.zeros((n_virt_int,n_virt_int)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"ovov": np.zeros((n_occ,n_virt_int,n_occ,n_virt_int))
	} 
	# Populate [[[Fn,S_1ext],S_2ext],S_1ext]
	fs1s2s1["c"] +=  1.000 * contract("ji,Ak,Bj,ikAB->",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs1s2s1["c"] +=  1.000 * contract("ji,kA,iB,ABjk->",f["oo"],t1["oV"],t1["oV"],t2["VVoo"])
	fs1s2s1["c"] += -1.000 * contract("Aa,iB,jA,aBij->",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"])
	fs1s2s1["c"] += -1.000 * contract("aA,Bi,Aj,ijaB->",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs1s2s1["c"] += -1.000 * contract("BA,Ci,Aj,ijBC->",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs1s2s1["c"] += -1.000 * contract("BA,iC,jB,ACij->",f["VV"],t1["oV"],t1["oV"],t2["VVoo"])

	fs1s2s1["oo"] +=  1.000 * contract("ki,Al,Bk,jlAB->ji",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs1s2s1["oo"] +=  1.000 * contract("ik,lA,kB,ABjl->ij",f["oo"],t1["oV"],t1["oV"],t2["VVoo"])
	fs1s2s1["oo"] += -1.000 * contract("lk,Ai,Bl,jkAB->ji",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs1s2s1["oo"] += -1.000 * contract("lk,iA,kB,ABjl->ij",f["oo"],t1["oV"],t1["oV"],t2["VVoo"])
	fs1s2s1["oo"] += -1.000 * contract("Aa,iB,kA,aBjk->ij",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"])
	fs1s2s1["oo"] +=  1.000 * contract("Aa,kB,iA,aBjk->ij",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"])
	fs1s2s1["oo"] +=  1.000 * contract("aA,Ai,Bk,jkaB->ji",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs1s2s1["oo"] += -1.000 * contract("aA,Bi,Ak,jkaB->ji",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs1s2s1["oo"] +=  1.000 * contract("BA,Ai,Ck,jkBC->ji",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs1s2s1["oo"] += -1.000 * contract("BA,Ci,Ak,jkBC->ji",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs1s2s1["oo"] += -1.000 * contract("BA,iC,kB,ACjk->ij",f["VV"],t1["oV"],t1["oV"],t2["VVoo"])
	fs1s2s1["oo"] +=  1.000 * contract("BA,kC,iB,ACjk->ij",f["VV"],t1["oV"],t1["oV"],t2["VVoo"])

	fs1s2s1["ov"] += -1.000 * contract("Aj,iA,Bk,jkaB->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs1s2s1["ov"] += -1.000 * contract("Aj,kA,Bk,ijaB->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs1s2s1["ov"] += -1.000 * contract("Aj,jB,Bk,ikaA->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs1s2s1["ov"] +=  1.000 * contract("ja,Ak,Bj,ikAB->ia",f["ov"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs1s2s1["ov"] +=  1.000 * contract("iA,Bj,Ak,jkaB->ia",f["oV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs1s2s1["ov"] += -2.000 * contract("jA,Ak,Bj,ikaB->ia",f["oV"],t1["Vo"],t1["Vo"],t2["oovV"])

	fs1s2s1["vo"] +=  1.000 * contract("Ai,jB,kA,aBjk->ai",f["Vo"],t1["oV"],t1["oV"],t2["vVoo"])
	fs1s2s1["vo"] +=  1.000 * contract("aj,kA,jB,ABik->ai",f["vo"],t1["oV"],t1["oV"],t2["VVoo"])
	fs1s2s1["vo"] += -2.000 * contract("Aj,jB,kA,aBik->ai",f["Vo"],t1["oV"],t1["oV"],t2["vVoo"])
	fs1s2s1["vo"] += -1.000 * contract("jA,kB,Ai,aBjk->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs1s2s1["vo"] += -1.000 * contract("jA,kB,Bj,aAik->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs1s2s1["vo"] += -1.000 * contract("jA,kB,Ak,aBij->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])

	fs1s2s1["vv"] += 1.000 * contract("Aa,iB,jA,bBij->ba",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"])
	fs1s2s1["vv"] += 1.000 * contract("aA,Bi,Aj,ijbB->ab",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs1s2s1["oooo"] +=  0.500 * contract("mi,Aj,Bm,klAB->klij",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs1s2s1["oooo"] +=  0.500 * contract("im,jA,mB,ABkl->ijkl",f["oo"],t1["oV"],t1["oV"],t2["VVoo"])
	fs1s2s1["oooo"] += -0.500 * contract("Aa,iB,jA,aBkl->ijkl",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"])
	fs1s2s1["oooo"] += -0.500 * contract("aA,Bi,Aj,klaB->klij",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs1s2s1["oooo"] += -0.500 * contract("BA,Ci,Aj,klBC->klij",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs1s2s1["oooo"] += -0.500 * contract("BA,iC,jB,ACkl->ijkl",f["VV"],t1["oV"],t1["oV"],t2["VVoo"])
	fs1s2s1["oooo"] = asym_term(fs1s2s1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs1s2s1["ooov"] += -0.500 * contract("Al,iA,Bj,klaB->ikja",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs1s2s1["ooov"] += -0.500 * contract("iA,Bj,Al,klaB->ikja",f["oV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs1s2s1["ooov"] +=  0.250 * contract("Al,lB,Bi,jkaA->jkia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs1s2s1["ooov"] += -0.250 * contract("la,Ai,Bl,jkAB->jkia",f["ov"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs1s2s1["ooov"] +=  0.250 * contract("lA,Ai,Bl,jkaB->jkia",f["oV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs1s2s1["ooov"] = asym_term(fs1s2s1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs1s2s1["ovoo"] += -0.500 * contract("Ai,jB,lA,aBkl->jaik",f["Vo"],t1["oV"],t1["oV"],t2["vVoo"])
	fs1s2s1["ovoo"] += -0.500 * contract("lA,iB,Aj,aBkl->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs1s2s1["ovoo"] += -0.250 * contract("al,iA,lB,ABjk->iajk",f["vo"],t1["oV"],t1["oV"],t2["VVoo"])
	fs1s2s1["ovoo"] +=  0.250 * contract("Al,lB,iA,aBjk->iajk",f["Vo"],t1["oV"],t1["oV"],t2["vVoo"])
	fs1s2s1["ovoo"] +=  0.250 * contract("lA,iB,Bl,aAjk->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs1s2s1["ovoo"] = asym_term(fs1s2s1["ovoo"],"ovoo")

	fs1s2s1["ovov"] += 0.250 * contract("Aa,iB,kA,bBjk->ibja",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"])
	fs1s2s1["ovov"] += 0.250 * contract("aA,Bi,Ak,jkbB->jaib",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"])

	return fs1s2s1

def fn_s1_s2_s2(f,t1,t2,inc_3_body=True):
	# [[[Fn,S_1ext],S_2ext],S_2ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize
	fs1s2s2 = {
		"c": 0.0,
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"vv": np.zeros((n_virt_int,n_virt_int)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"ovov": np.zeros((n_occ,n_virt_int,n_occ,n_virt_int)),
		"ovvv": np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int)),
		"vvov": np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	}
	# Populate [[[Fn,S_1ext],S_2ext],S_2ext]
	fs1s2s2["c"] += -2.000 * contract("Ai,jA,ikaB,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["c"] += -2.000 * contract("iA,Aj,jkaB,aBik->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["c"] += -1.000 * contract("ai,iA,jkaB,ABjk->",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs1s2s2["c"] += -1.000 * contract("Ai,jA,ikBC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["c"] += -1.000 * contract("Ai,iB,jkaA,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["c"] += -1.000 * contract("Ai,iB,jkAC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["c"] += -1.000 * contract("ia,Ai,jkAB,aBjk->",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs1s2s2["c"] += -1.000 * contract("iA,Bi,jkaB,aAjk->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["c"] += -1.000 * contract("iA,Bi,jkBC,ACjk->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["c"] += -1.000 * contract("iA,Aj,jkBC,BCik->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])

	fs1s2s2["oo"] += -2.000 * contract("ak,kA,ilaB,ABjl->ij",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs1s2s2["oo"] += -2.000 * contract("Ak,lA,ikaB,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oo"] += -2.000 * contract("Ak,kB,ilaA,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oo"] += -2.000 * contract("Ak,kB,ilAC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oo"] += -2.000 * contract("ka,Ak,ilAB,aBjl->ij",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs1s2s2["oo"] += -2.000 * contract("kA,Bk,ilaB,aAjl->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oo"] += -2.000 * contract("kA,Bk,ilBC,ACjl->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oo"] += -2.000 * contract("kA,Al,ilaB,aBjk->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oo"] += -1.000 * contract("Ai,kA,jlaB,aBkl->ji",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oo"] += -1.000 * contract("Ak,iA,klaB,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oo"] += -1.000 * contract("Ak,lA,ikBC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oo"] += -1.000 * contract("iA,Ak,klaB,aBjl->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oo"] += -1.000 * contract("kA,Ai,jlaB,aBkl->ji",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oo"] += -1.000 * contract("kA,Al,ilBC,BCjk->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oo"] += -0.500 * contract("kA,Ai,jlBC,BCkl->ji",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oo"] += -0.500 * contract("Ai,kA,jlBC,BCkl->ji",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oo"] += -0.500 * contract("Ak,iA,klBC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oo"] += -0.500 * contract("iA,Ak,klBC,BCjl->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])

	fs1s2s2["ov"] += -1.000 * contract("kj,jA,ilaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs1s2s2["ov"] += -1.000 * contract("Aa,jA,ikbB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ov"] +=  1.000 * contract("Ab,jA,ikaB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ov"] +=  1.000 * contract("BA,jB,ikaC,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs1s2s2["ov"] +=  0.500 * contract("ij,jA,klaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs1s2s2["ov"] += -0.500 * contract("Aa,jA,ikBC,BCjk->ia",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["ov"] += -0.500 * contract("Ab,iA,jkaB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ov"] += -0.500 * contract("BA,iB,jkaC,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])

	fs1s2s2["vo"] += -1.000 * contract("kj,Ak,jlAB,aBil->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs1s2s2["vo"] += -1.000 * contract("aA,Aj,jkbB,bBik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["vo"] +=  1.000 * contract("bA,Aj,jkbB,aBik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["vo"] +=  1.000 * contract("BA,Aj,jkBC,aCik->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs1s2s2["vo"] +=  0.500 * contract("ji,Aj,klAB,aBkl->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs1s2s2["vo"] += -0.500 * contract("aA,Aj,jkBC,BCik->ai",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["vo"] += -0.500 * contract("bA,Ai,jkbB,aBjk->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["vo"] += -0.500 * contract("BA,Ai,jkBC,aCjk->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])

	fs1s2s2["vv"] +=  2.000 * contract("Ai,jA,ikaB,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["vv"] +=  2.000 * contract("iA,Aj,jkaB,bBik->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["vv"] +=  1.000 * contract("Ai,iB,jkaA,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["vv"] +=  1.000 * contract("iA,Bi,jkaB,bAjk->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["vv"] +=  0.500 * contract("ai,iA,jkbB,ABjk->ab",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs1s2s2["vv"] +=  0.500 * contract("ia,Ai,jkAB,bBjk->ba",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs1s2s2["oooo"] +=  0.500 * contract("Ai,mA,jkaB,aBlm->jkil",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oooo"] += -0.500 * contract("am,mA,ijaB,ABkl->ijkl",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs1s2s2["oooo"] +=  0.500 * contract("Am,iA,jmaB,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oooo"] += -0.500 * contract("Am,mB,ijaA,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oooo"] += -0.500 * contract("Am,mB,ijAC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oooo"] += -0.500 * contract("ma,Am,ijAB,aBkl->ijkl",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs1s2s2["oooo"] +=  0.500 * contract("iA,Am,jmaB,aBkl->ijkl",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oooo"] +=  0.500 * contract("mA,Ai,jkaB,aBlm->jkil",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oooo"] += -0.500 * contract("mA,Bm,ijaB,aAkl->ijkl",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["oooo"] += -0.500 * contract("mA,Bm,ijBC,ACkl->ijkl",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oooo"] +=  0.250 * contract("Ai,mA,jkBC,BClm->jkil",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oooo"] +=  0.250 * contract("Am,iA,jmBC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oooo"] +=  0.250 * contract("iA,Am,jmBC,BCkl->ijkl",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oooo"] +=  0.250 * contract("mA,Ai,jkBC,BClm->jkil",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["oooo"] = asym_term(fs1s2s2["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs1s2s2["ooov"] += -0.500 * contract("il,lA,jmaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs1s2s2["ooov"] +=  0.500 * contract("Ab,iA,jlaB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ooov"] +=  0.500 * contract("BA,iB,jlaC,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs1s2s2["ooov"] += -0.250 * contract("ml,lA,ijaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs1s2s2["ooov"] += -0.250 * contract("Aa,lA,ijbB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ooov"] +=  0.250 * contract("Ab,lA,ijaB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ooov"] +=  0.250 * contract("BA,lB,ijaC,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs1s2s2["ooov"] += -0.125 * contract("Aa,lA,ijBC,BCkl->ijka",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["ooov"] = asym_term(fs1s2s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs1s2s2["ovoo"] += -0.500 * contract("li,Al,jmAB,aBkm->jaik",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs1s2s2["ovoo"] +=  0.500 * contract("bA,Ai,jlbB,aBkl->jaik",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ovoo"] +=  0.500 * contract("BA,Ai,jlBC,aCkl->jaik",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs1s2s2["ovoo"] += -0.250 * contract("ml,Am,ilAB,aBjk->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs1s2s2["ovoo"] += -0.250 * contract("aA,Al,ilbB,bBjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ovoo"] +=  0.250 * contract("bA,Al,ilbB,aBjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ovoo"] +=  0.250 * contract("BA,Al,ilBC,aCjk->iajk",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs1s2s2["ovoo"] += -0.125 * contract("aA,Al,ilBC,BCjk->iajk",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs1s2s2["ovoo"] = asym_term(fs1s2s2["ovoo"],"ovoo")

	fs1s2s2["ovov"] += 0.500 * contract("Ak,lA,ikaB,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ovov"] += 0.500 * contract("Ak,kB,ilaA,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ovov"] += 0.500 * contract("kA,Bk,ilaB,bAjl->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ovov"] += 0.500 * contract("kA,Al,ilaB,bBjk->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ovov"] += 0.250 * contract("Ai,kA,jlaB,bBkl->jbia",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ovov"] += 0.250 * contract("ak,kA,ilbB,ABjl->iajb",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs1s2s2["ovov"] += 0.250 * contract("Ak,iA,klaB,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ovov"] += 0.250 * contract("ka,Ak,ilAB,bBjl->ibja",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs1s2s2["ovov"] += 0.250 * contract("iA,Ak,klaB,bBjl->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ovov"] += 0.250 * contract("kA,Ai,jlaB,bBkl->jbia",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	fs1s2s2["ovvv"] += 0.500 * contract("Aa,jA,ikbB,cBjk->icab",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs1s2s2["ovvv"] = asym_term(fs1s2s2["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	fs1s2s2["vvov"] += 0.500 * contract("aA,Aj,jkbB,cBik->acib",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs1s2s2["vvov"] = asym_term(fs1s2s2["vvov"],"vvov")

	if(inc_3_body):
		fs1s2s2["ooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs1s2s2["oovooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs1s2s2["oovoov"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs1s2s2["oovovv"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs1s2s2["ovvoov"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs1s2s2["ooooov"] +=  (1./12.) * contract("in,nA,jkaB,ABlm->ijklma",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs1s2s2["ooooov"] += -(1./12.) * contract("Ab,iA,jkaB,bBlm->ijklma",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs1s2s2["ooooov"] += -(1./12.) * contract("BA,iB,jkaC,AClm->ijklma",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs1s2s2["ooooov"] = asym_term(fs1s2s2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs1s2s2["oovooo"] +=  (1./12.) * contract("ni,An,jkAB,aBlm->jkailm",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs1s2s2["oovooo"] += -(1./12.) * contract("bA,Ai,jkbB,aBlm->jkailm",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs1s2s2["oovooo"] += -(1./12.) * contract("BA,Ai,jkBC,aClm->jkailm",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs1s2s2["oovooo"] = asym_term(fs1s2s2["oovooo"],"oovooo")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs1s2s2["oovoov"] += -(1./18.) * contract("Ai,mA,jkaB,bBlm->jkbila",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs1s2s2["oovoov"] += -(1./18.) * contract("Am,iA,jmaB,bBkl->ijbkla",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs1s2s2["oovoov"] +=  (1./18.) * contract("Am,mB,ijaA,bBkl->ijbkla",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs1s2s2["oovoov"] += -(1./18.) * contract("iA,Am,jmaB,bBkl->ijbkla",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs1s2s2["oovoov"] += -(1./18.) * contract("mA,Ai,jkaB,bBlm->jkbila",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs1s2s2["oovoov"] +=  (1./18.) * contract("mA,Bm,ijaB,bAkl->ijbkla",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs1s2s2["oovoov"] +=  (1./36.) * contract("am,mA,ijbB,ABkl->ijaklb",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs1s2s2["oovoov"] +=  (1./36.) * contract("ma,Am,ijAB,bBkl->ijbkla",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs1s2s2["oovoov"] = asym_term(fs1s2s2["oovoov"],"oovoov")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs1s2s2["oovovv"] +=  (1./18.) * contract("Aa,lA,ijbB,cBkl->ijckab",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs1s2s2["oovovv"] = asym_term(fs1s2s2["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		fs1s2s2["ovvoov"] +=  (1./18.) * contract("aA,Al,ilbB,cBjk->iacjkb",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs1s2s2["ovvoov"] = asym_term(fs1s2s2["ovvoov"],"ovvoov")

	return fs1s2s2 

def fn_s2_s1_s1(f,t1,t2):
	# [[[Fn,S_2ext],S_1ext],S_1ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize
	fs2s1s1 = {
		"c": 0.0,
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	} 
	# Populate [[[Fn,S_2ext],S_1ext],S_1ext]
	fs2s1s1["c"] +=  2.000 * contract("ji,Ak,Bj,ikAB->",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1s1["c"] +=  2.000 * contract("ji,kA,iB,ABjk->",f["oo"],t1["oV"],t1["oV"],t2["VVoo"])
	fs2s1s1["c"] += -2.000 * contract("Aa,iB,jA,aBij->",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"])
	fs2s1s1["c"] += -2.000 * contract("aA,Bi,Aj,ijaB->",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs2s1s1["c"] += -2.000 * contract("BA,Ci,Aj,ijBC->",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1s1["c"] += -2.000 * contract("BA,iC,jB,ACij->",f["VV"],t1["oV"],t1["oV"],t2["VVoo"])

	fs2s1s1["oo"] += -2.000 * contract("ki,jA,lB,ABkl->ji",f["oo"],t1["oV"],t1["oV"],t2["VVoo"])
	fs2s1s1["oo"] += -2.000 * contract("ik,Aj,Bl,klAB->ij",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1s1["oo"] += -2.000 * contract("lk,Ai,Bl,jkAB->ji",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1s1["oo"] += -2.000 * contract("lk,iA,kB,ABjl->ij",f["oo"],t1["oV"],t1["oV"],t2["VVoo"])
	fs2s1s1["oo"] += -2.000 * contract("Aa,iB,kA,aBjk->ij",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"])
	fs2s1s1["oo"] +=  2.000 * contract("Aa,kB,iA,aBjk->ij",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"])
	fs2s1s1["oo"] +=  2.000 * contract("aA,Ai,Bk,jkaB->ji",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs2s1s1["oo"] += -2.000 * contract("aA,Bi,Ak,jkaB->ji",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs2s1s1["oo"] +=  2.000 * contract("BA,Ai,Ck,jkBC->ji",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1s1["oo"] += -2.000 * contract("BA,Ci,Ak,jkBC->ji",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1s1["oo"] += -2.000 * contract("BA,iC,kB,ACjk->ij",f["VV"],t1["oV"],t1["oV"],t2["VVoo"])
	fs2s1s1["oo"] +=  2.000 * contract("BA,kC,iB,ACjk->ij",f["VV"],t1["oV"],t1["oV"],t2["VVoo"])

	fs2s1s1["ov"] += -2.000 * contract("Aj,iA,Bk,jkaB->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs2s1s1["ov"] += -2.000 * contract("Aj,kA,Bk,ijaB->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs2s1s1["ov"] +=  1.000 * contract("Aj,iB,Bk,jkaA->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs2s1s1["ov"] += -2.000 * contract("Aj,jB,Bk,ikaA->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs2s1s1["ov"] += -2.000 * contract("ja,iA,kB,ABjk->ia",f["ov"],t1["oV"],t1["oV"],t2["VVoo"])

	fs2s1s1["vo"] += -2.000 * contract("aj,Ai,Bk,jkAB->ai",f["vo"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1s1["vo"] += -2.000 * contract("jA,kB,Ai,aBjk->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs2s1s1["vo"] +=  1.000 * contract("jA,kB,Bi,aAjk->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs2s1s1["vo"] += -2.000 * contract("jA,kB,Bj,aAik->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs2s1s1["vo"] += -2.000 * contract("jA,kB,Ak,aBij->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs2s1s1["oooo"] +=  1.000 * contract("mi,jA,kB,ABlm->jkil",f["oo"],t1["oV"],t1["oV"],t2["VVoo"])
	fs2s1s1["oooo"] +=  1.000 * contract("im,Aj,Bk,lmAB->iljk",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1s1["oooo"] += -1.000 * contract("Aa,iB,jA,aBkl->ijkl",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"])
	fs2s1s1["oooo"] += -1.000 * contract("aA,Bi,Aj,klaB->klij",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"])
	fs2s1s1["oooo"] += -1.000 * contract("BA,Ci,Aj,klBC->klij",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1s1["oooo"] += -1.000 * contract("BA,iC,jB,ACkl->ijkl",f["VV"],t1["oV"],t1["oV"],t2["VVoo"])
	fs2s1s1["oooo"] = asym_term(fs2s1s1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2s1s1["ooov"] += -1.000 * contract("Al,iA,Bj,klaB->ikja",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs2s1s1["ooov"] += -0.500 * contract("Ai,jB,Bl,klaA->jkia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs2s1s1["ooov"] += -0.500 * contract("la,iA,jB,ABkl->ijka",f["ov"],t1["oV"],t1["oV"],t2["VVoo"])
	fs2s1s1["ooov"] +=  0.250 * contract("Al,lB,Bi,jkaA->jkia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"])
	fs2s1s1["ooov"] = asym_term(fs2s1s1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2s1s1["ovoo"] += -1.000 * contract("lA,iB,Aj,aBkl->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs2s1s1["ovoo"] += -0.500 * contract("al,Ai,Bj,klAB->kaij",f["vo"],t1["Vo"],t1["Vo"],t2["ooVV"])
	fs2s1s1["ovoo"] += -0.500 * contract("iA,lB,Bj,aAkl->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs2s1s1["ovoo"] +=  0.250 * contract("lA,iB,Bl,aAjk->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs2s1s1["ovoo"] = asym_term(fs2s1s1["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	fs2s1s1["oovv"] += -1.000 * contract("Aa,iB,Bk,jkbA->ijab",f["Vv"],t1["oV"],t1["Vo"],t2["oovV"])
	fs2s1s1["oovv"] = asym_term(fs2s1s1["oovv"],"oovv")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	fs2s1s1["vvoo"] += -1.000 * contract("aA,kB,Bi,bAjk->abij",f["vV"],t1["oV"],t1["Vo"],t2["vVoo"])
	fs2s1s1["vvoo"] = asym_term(fs2s1s1["vvoo"],"vvoo")

	return fs2s1s1 

def fn_s2_s1_s2(f,t1,t2,inc_3_body=True):
	# [[[Fn,S_2ext],S_1ext],S_2ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize
	fs2s1s2 = {
		"c": 0.0,
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"vv": np.zeros((n_virt_int,n_virt_int)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"ovov": np.zeros((n_occ,n_virt_int,n_occ,n_virt_int)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	} 
	# Populate [[[Fn,S_2ext],S_1ext],S_2ext]
	fs2s1s2["c"] += -0.500 * contract("ai,iA,jkaB,ABjk->",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["c"] += -1.000 * contract("Ai,jA,ikaB,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["c"] += -0.500 * contract("Ai,jA,ikBC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s1s2["c"] += -0.500 * contract("Ai,iB,jkaA,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["c"] += -0.500 * contract("Ai,iB,jkAC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s1s2["c"] += -0.500 * contract("ia,Ai,jkAB,aBjk->",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["c"] += -0.500 * contract("iA,Bi,jkaB,aAjk->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["c"] += -0.500 * contract("iA,Bi,jkBC,ACjk->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s1s2["c"] += -1.000 * contract("iA,Aj,jkaB,aBik->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["c"] += -0.500 * contract("iA,Aj,jkBC,BCik->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])

	fs2s1s2["oo"] += -1.000 * contract("ak,kA,ilaB,ABjl->ij",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["oo"] += -1.000 * contract("Ak,iA,klaB,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["oo"] += -0.500 * contract("Ak,iA,klBC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s1s2["oo"] += -1.000 * contract("Ak,lA,ikaB,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["oo"] += -0.500 * contract("Ak,lA,ikBC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s1s2["oo"] += -1.000 * contract("Ak,kB,ilaA,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["oo"] += -1.000 * contract("Ak,kB,ilAC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s1s2["oo"] += -1.000 * contract("ka,Ak,ilAB,aBjl->ij",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["oo"] += -1.000 * contract("kA,Ai,jlaB,aBkl->ji",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["oo"] += -0.500 * contract("kA,Ai,jlBC,BCkl->ji",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s1s2["oo"] += -1.000 * contract("kA,Bk,ilaB,aAjl->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["oo"] += -1.000 * contract("kA,Bk,ilBC,ACjl->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s1s2["oo"] += -1.000 * contract("kA,Al,ilaB,aBjk->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["oo"] += -0.500 * contract("kA,Al,ilBC,BCjk->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])

	fs2s1s2["ov"] +=  1.000 * contract("kj,iA,jlaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ov"] += -1.000 * contract("kj,jA,ilaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ov"] +=  1.000 * contract("kj,lA,ijaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ov"] += -0.500 * contract("Ab,iA,jkaB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ov"] +=  1.000 * contract("Ab,jA,ikaB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ov"] +=  0.500 * contract("Ab,iB,jkaA,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ov"] += -1.000 * contract("Ab,jB,ikaA,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ov"] += -0.500 * contract("BA,iB,jkaC,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ov"] +=  1.000 * contract("BA,jB,ikaC,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ov"] +=  0.500 * contract("BA,iC,jkaB,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ov"] += -1.000 * contract("BA,jC,ikaB,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])

	fs2s1s2["vo"] +=  1.000 * contract("kj,Ai,jlAB,aBkl->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["vo"] += -1.000 * contract("kj,Ak,jlAB,aBil->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["vo"] +=  1.000 * contract("kj,Al,jlAB,aBik->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["vo"] += -0.500 * contract("bA,Ai,jkbB,aBjk->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["vo"] +=  0.500 * contract("bA,Bi,jkbB,aAjk->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["vo"] +=  1.000 * contract("bA,Aj,jkbB,aBik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["vo"] += -1.000 * contract("bA,Bj,jkbB,aAik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["vo"] += -0.500 * contract("BA,Ai,jkBC,aCjk->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["vo"] +=  0.500 * contract("BA,Ci,jkBC,aAjk->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["vo"] +=  1.000 * contract("BA,Aj,jkBC,aCik->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["vo"] += -1.000 * contract("BA,Cj,jkBC,aAik->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])

	fs2s1s2["vv"] += 1.000 * contract("Ai,jA,ikaB,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["vv"] += 0.500 * contract("Ai,iB,jkaA,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["vv"] += 0.500 * contract("iA,Bi,jkaB,bAjk->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["vv"] += 1.000 * contract("iA,Aj,jkaB,bBik->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs2s1s2["oooo"] +=  0.500 * contract("Am,iA,jmaB,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["oooo"] +=  0.500 * contract("mA,Ai,jkaB,aBlm->jkil",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["oooo"] += -0.250 * contract("am,mA,ijaB,ABkl->ijkl",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["oooo"] +=  0.250 * contract("Am,iA,jmBC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s1s2["oooo"] += -0.250 * contract("Am,mB,ijaA,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["oooo"] += -0.250 * contract("Am,mB,ijAC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s1s2["oooo"] += -0.250 * contract("ma,Am,ijAB,aBkl->ijkl",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["oooo"] +=  0.250 * contract("mA,Ai,jkBC,BClm->jkil",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s1s2["oooo"] += -0.250 * contract("mA,Bm,ijaB,aAkl->ijkl",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["oooo"] += -0.250 * contract("mA,Bm,ijBC,ACkl->ijkl",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s1s2["oooo"] = asym_term(fs2s1s2["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2s1s2["ooov"] += -0.500 * contract("li,jA,kmaB,ABlm->jkia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ooov"] += -0.500 * contract("ml,iA,jlaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ooov"] +=  0.500 * contract("Ab,iA,jlaB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ooov"] += -0.500 * contract("Ab,iB,jlaA,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ooov"] +=  0.500 * contract("BA,iB,jlaC,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ooov"] += -0.500 * contract("BA,iC,jlaB,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ooov"] += -0.250 * contract("li,mA,jkaB,ABlm->jkia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ooov"] += -0.250 * contract("ml,lA,ijaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ooov"] +=  0.250 * contract("Ab,lA,ijaB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ooov"] += -0.250 * contract("Ab,lB,ijaA,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ooov"] +=  0.250 * contract("BA,lB,ijaC,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ooov"] += -0.250 * contract("BA,lC,ijaB,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["ooov"] = asym_term(fs2s1s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2s1s2["ovoo"] += -0.500 * contract("il,Aj,lmAB,aBkm->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["ovoo"] += -0.500 * contract("ml,Ai,jlAB,aBkm->jaik",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["ovoo"] +=  0.500 * contract("bA,Ai,jlbB,aBkl->jaik",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ovoo"] += -0.500 * contract("bA,Bi,jlbB,aAkl->jaik",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ovoo"] +=  0.500 * contract("BA,Ai,jlBC,aCkl->jaik",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["ovoo"] += -0.500 * contract("BA,Ci,jlBC,aAkl->jaik",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["ovoo"] += -0.250 * contract("il,Am,lmAB,aBjk->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["ovoo"] += -0.250 * contract("ml,Am,ilAB,aBjk->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["ovoo"] +=  0.250 * contract("bA,Al,ilbB,aBjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ovoo"] += -0.250 * contract("bA,Bl,ilbB,aAjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ovoo"] +=  0.250 * contract("BA,Al,ilBC,aCjk->iajk",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["ovoo"] += -0.250 * contract("BA,Cl,ilBC,aAjk->iajk",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["ovoo"] = asym_term(fs2s1s2["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	fs2s1s2["oovv"] += -2.000 * contract("Ak,Bl,ikaB,jlbA->ijab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"])
	fs2s1s2["oovv"] += -1.000 * contract("ka,iA,jlbB,ABkl->ijab",f["ov"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["oovv"] +=  0.500 * contract("Ak,Bl,ijaB,klbA->ijab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"])
	fs2s1s2["oovv"] +=  0.500 * contract("Ak,Bl,klaB,ijbA->ijab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"])
	fs2s1s2["oovv"] += -0.500 * contract("ka,lA,ijbB,ABkl->ijab",f["ov"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s1s2["oovv"] = asym_term(fs2s1s2["oovv"],"oovv")

	fs2s1s2["ovov"] += 0.250 * contract("Ak,iA,klaB,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ovov"] += 0.250 * contract("Ak,lA,ikaB,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ovov"] += 0.250 * contract("Ak,kB,ilaA,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ovov"] += 0.250 * contract("kA,Ai,jlaB,bBkl->jbia",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ovov"] += 0.250 * contract("kA,Bk,ilaB,bAjl->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s1s2["ovov"] += 0.250 * contract("kA,Al,ilaB,bBjk->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	fs2s1s2["vvoo"] += -2.000 * contract("kA,lB,aAil,bBjk->abij",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"])
	fs2s1s2["vvoo"] += -1.000 * contract("ak,Ai,klAB,bBjl->abij",f["vo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["vvoo"] += -0.500 * contract("ak,Al,klAB,bBij->abij",f["vo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s1s2["vvoo"] += -0.500 * contract("kA,lB,aAij,bBkl->abij",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"])
	fs2s1s2["vvoo"] +=  0.500 * contract("kA,lB,aBij,bAkl->abij",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"])
	fs2s1s2["vvoo"] = asym_term(fs2s1s2["vvoo"],"vvoo")
	
	if(inc_3_body):
		fs2s1s2["ooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s1s2["oovooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s1s2["oooovv"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s1s2["oovoov"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s1s2["ovvooo"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s1s2["ooovvv"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		fs2s1s2["vvvooo"] = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s1s2["ooooov"] += -(1./6.) * contract("ni,jA,klaB,ABmn->jklima",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs2s1s2["ooooov"] += -(1./12.) * contract("Ab,iA,jkaB,bBlm->ijklma",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs2s1s2["ooooov"] +=  (1./12.) * contract("Ab,iB,jkaA,bBlm->ijklma",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs2s1s2["ooooov"] += -(1./12.) * contract("BA,iB,jkaC,AClm->ijklma",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs2s1s2["ooooov"] +=  (1./12.) * contract("BA,iC,jkaB,AClm->ijklma",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs2s1s2["ooooov"] = asym_term(fs2s1s2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s1s2["oovooo"] += -(1./6.) * contract("in,Aj,knAB,aBlm->ikajlm",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s1s2["oovooo"] += -(1./12.) * contract("bA,Ai,jkbB,aBlm->jkailm",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs2s1s2["oovooo"] +=  (1./12.) * contract("bA,Bi,jkbB,aAlm->jkailm",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs2s1s2["oovooo"] += -(1./12.) * contract("BA,Ai,jkBC,aClm->jkailm",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s1s2["oovooo"] +=  (1./12.) * contract("BA,Ci,jkBC,aAlm->jkailm",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s1s2["oovooo"] = asym_term(fs2s1s2["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s1s2["oooovv"] +=  (1./6.) * contract("Ai,Bm,jkaB,lmbA->jkliab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"])
		fs2s1s2["oooovv"] += -(1./6.) * contract("Am,Bi,jmaB,klbA->jkliab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"])
		fs2s1s2["oooovv"] +=  (1./6.) * contract("ma,iA,jkbB,ABlm->ijklab",f["ov"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs2s1s2["oooovv"] = asym_term(fs2s1s2["oooovv"],"oooovv")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s1s2["oovoov"] += -(1./18.) * contract("Am,iA,jmaB,bBkl->ijbkla",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs2s1s2["oovoov"] += -(1./18.) * contract("mA,Ai,jkaB,bBlm->jkbila",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs2s1s2["oovoov"] +=  (1./36.) * contract("Am,mB,ijaA,bBkl->ijbkla",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs2s1s2["oovoov"] +=  (1./36.) * contract("mA,Bm,ijaB,bAkl->ijbkla",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs2s1s2["oovoov"] = asym_term(fs2s1s2["oovoov"],"oovoov")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s1s2["ovvooo"] +=  (1./6.) * contract("am,Ai,jmAB,bBkl->jabikl",f["vo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s1s2["ovvooo"] +=  (1./6.) * contract("iA,mB,aBjk,bAlm->iabjkl",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"])
		fs2s1s2["ovvooo"] +=  (1./6.) * contract("mA,iB,aAjk,bBlm->iabjkl",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"])
		fs2s1s2["ovvooo"] = asym_term(fs2s1s2["ovvooo"],"ovvooo")

		# ooovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		fs2s1s2["ooovvv"] +=  0.500 * contract("Aa,Bl,ijbB,klcA->ijkabc",f["Vv"],t1["Vo"],t2["oovV"],t2["oovV"])
		fs2s1s2["ooovvv"] = asym_term(fs2s1s2["ooovvv"],"ooovvv")

		# vvvooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s1s2["vvvooo"] +=  0.500 * contract("aA,lB,bBij,cAkl->abcijk",f["vV"],t1["oV"],t2["vVoo"],t2["vVoo"])
		fs2s1s2["vvvooo"] = asym_term(fs2s1s2["vvvooo"],"vvvooo")

	return fs2s1s2

def fn_s2_s2_s1(f,t1,t2,inc_3_body=True):
	# [[[Fn,S_2ext],S_2ext],S_1ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize
	fs2s2s1 = {
		"c": 0.0,
		"oo": np.zeros((n_occ,n_occ)),
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"vv": np.zeros((n_virt_int,n_virt_int)),
		"oooo": np.zeros((n_occ,n_occ,n_occ,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"ovov": np.zeros((n_occ,n_virt_int,n_occ,n_virt_int)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ)),
		"ovvv": np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int)),
		"vvov": np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	} 
	# Populate [[[Fn,S_2ext],S_2ext],S_1ext]
	fs2s2s1["c"] += -0.500 * contract("ai,iA,jkaB,ABjk->",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["c"] +=  1.000 * contract("ai,jA,ikaB,ABjk->",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["c"] += -1.000 * contract("Ai,jA,ikaB,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["c"] += -0.500 * contract("Ai,jA,ikBC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["c"] += -0.500 * contract("Ai,iB,jkaA,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["c"] += -0.500 * contract("Ai,iB,jkAC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["c"] +=  1.000 * contract("Ai,jB,ikaA,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["c"] +=  1.000 * contract("Ai,jB,ikAC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["c"] += -0.500 * contract("ia,Ai,jkAB,aBjk->",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["c"] +=  1.000 * contract("ia,Aj,jkAB,aBik->",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["c"] += -0.500 * contract("iA,Bi,jkaB,aAjk->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["c"] += -0.500 * contract("iA,Bi,jkBC,ACjk->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["c"] += -1.000 * contract("iA,Aj,jkaB,aBik->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["c"] += -0.500 * contract("iA,Aj,jkBC,BCik->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["c"] +=  1.000 * contract("iA,Bj,jkaB,aAik->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["c"] +=  1.000 * contract("iA,Bj,jkBC,ACik->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])

	fs2s2s1["oo"] += -0.500 * contract("ai,jA,klaB,ABkl->ji",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("ai,kA,jlaB,ABkl->ji",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["oo"] += -0.500 * contract("Ai,jB,klaA,aBkl->ji",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] += -0.500 * contract("Ai,jB,klAC,BCkl->ji",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("Ai,kB,jlaA,aBkl->ji",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("Ai,kB,jlAC,BCkl->ji",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("ak,iA,klaB,ABjl->ij",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["oo"] += -1.000 * contract("ak,kA,ilaB,ABjl->ij",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("ak,lA,ikaB,ABjl->ij",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["oo"] += -1.000 * contract("Ak,iA,klaB,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] += -0.500 * contract("Ak,iA,klBC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] += -1.000 * contract("Ak,lA,ikaB,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] += -0.500 * contract("Ak,lA,ikBC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("Ak,iB,klaA,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("Ak,iB,klAC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] += -1.000 * contract("Ak,kB,ilaA,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] += -1.000 * contract("Ak,kB,ilAC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("Ak,lB,ikaA,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("Ak,lB,ikAC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] += -0.500 * contract("ia,Aj,klAB,aBkl->ij",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("ia,Ak,klAB,aBjl->ij",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("ka,Ai,jlAB,aBkl->ji",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["oo"] += -1.000 * contract("ka,Ak,ilAB,aBjl->ij",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("ka,Al,ilAB,aBjk->ij",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["oo"] += -0.500 * contract("iA,Bj,klaB,aAkl->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] += -0.500 * contract("iA,Bj,klBC,ACkl->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("iA,Bk,klaB,aAjl->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("iA,Bk,klBC,ACjl->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] += -1.000 * contract("kA,Ai,jlaB,aBkl->ji",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] += -0.500 * contract("kA,Ai,jlBC,BCkl->ji",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("kA,Bi,jlaB,aAkl->ji",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("kA,Bi,jlBC,ACkl->ji",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] += -1.000 * contract("kA,Bk,ilaB,aAjl->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] += -1.000 * contract("kA,Bk,ilBC,ACjl->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] += -1.000 * contract("kA,Al,ilaB,aBjk->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] += -0.500 * contract("kA,Al,ilBC,BCjk->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("kA,Bl,ilaB,aAjk->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oo"] +=  1.000 * contract("kA,Bl,ilBC,ACjk->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])

	fs2s2s1["ov"] += -1.000 * contract("ij,kA,jlaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ov"] +=  2.000 * contract("kj,iA,jlaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ov"] += -1.000 * contract("kj,jA,ilaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ov"] +=  2.000 * contract("kj,lA,ijaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ov"] += -0.500 * contract("ba,iA,jkbB,ABjk->ia",f["vv"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ov"] +=  1.000 * contract("ba,jA,ikbB,ABjk->ia",f["vv"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ov"] += -0.500 * contract("Aa,iB,jkbA,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ov"] += -0.500 * contract("Aa,iB,jkAC,BCjk->ia",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["ov"] +=  1.000 * contract("Aa,jB,ikbA,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ov"] +=  1.000 * contract("Aa,jB,ikAC,BCjk->ia",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["ov"] += -0.500 * contract("Ab,iA,jkaB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ov"] +=  1.000 * contract("Ab,jA,ikaB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ov"] +=  1.000 * contract("Ab,iB,jkaA,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ov"] += -2.000 * contract("Ab,jB,ikaA,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ov"] += -0.500 * contract("BA,iB,jkaC,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ov"] +=  1.000 * contract("BA,jB,ikaC,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ov"] +=  1.000 * contract("BA,iC,jkaB,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ov"] += -2.000 * contract("BA,jC,ikaB,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])

	fs2s2s1["vo"] += -1.000 * contract("ji,Ak,klAB,aBjl->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["vo"] +=  2.000 * contract("kj,Ai,jlAB,aBkl->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["vo"] += -1.000 * contract("kj,Ak,jlAB,aBil->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["vo"] +=  2.000 * contract("kj,Al,jlAB,aBik->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["vo"] += -0.500 * contract("ab,Ai,jkAB,bBjk->ai",f["vv"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["vo"] +=  1.000 * contract("ab,Aj,jkAB,bBik->ai",f["vv"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["vo"] += -0.500 * contract("aA,Bi,jkbB,bAjk->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vo"] += -0.500 * contract("aA,Bi,jkBC,ACjk->ai",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["vo"] +=  1.000 * contract("aA,Bj,jkbB,bAik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vo"] +=  1.000 * contract("aA,Bj,jkBC,ACik->ai",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["vo"] += -0.500 * contract("bA,Ai,jkbB,aBjk->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vo"] +=  1.000 * contract("bA,Bi,jkbB,aAjk->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vo"] +=  1.000 * contract("bA,Aj,jkbB,aBik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vo"] += -2.000 * contract("bA,Bj,jkbB,aAik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vo"] += -0.500 * contract("BA,Ai,jkBC,aCjk->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["vo"] +=  1.000 * contract("BA,Ci,jkBC,aAjk->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["vo"] +=  1.000 * contract("BA,Aj,jkBC,aCik->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["vo"] += -2.000 * contract("BA,Cj,jkBC,aAik->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])

	fs2s2s1["vv"] += -1.000 * contract("ai,jA,ikbB,ABjk->ab",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["vv"] +=  1.000 * contract("Ai,jA,ikaB,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vv"] +=  0.500 * contract("Ai,iB,jkaA,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vv"] += -1.000 * contract("Ai,jB,ikaA,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vv"] += -1.000 * contract("ia,Aj,jkAB,bBik->ba",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["vv"] +=  0.500 * contract("iA,Bi,jkaB,bAjk->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vv"] +=  1.000 * contract("iA,Aj,jkaB,bBik->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vv"] += -1.000 * contract("iA,Bj,jkaB,bAik->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs2s2s1["oooo"] += -1.000 * contract("ai,jA,kmaB,ABlm->jkil",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["oooo"] += -1.000 * contract("Ai,jB,kmaA,aBlm->jkil",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oooo"] += -1.000 * contract("Ai,jB,kmAC,BClm->jkil",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oooo"] += -1.000 * contract("ia,Aj,kmAB,aBlm->ikjl",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["oooo"] += -1.000 * contract("iA,Bj,kmaB,aAlm->ikjl",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oooo"] += -1.000 * contract("iA,Bj,kmBC,AClm->ikjl",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oooo"] += -0.500 * contract("ai,mA,jkaB,ABlm->jkil",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["oooo"] += -0.500 * contract("Ai,mB,jkaA,aBlm->jkil",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oooo"] += -0.500 * contract("Ai,mB,jkAC,BClm->jkil",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oooo"] += -0.500 * contract("am,iA,jmaB,ABkl->ijkl",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["oooo"] +=  0.500 * contract("Am,iA,jmaB,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oooo"] += -0.500 * contract("Am,iB,jmaA,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oooo"] += -0.500 * contract("Am,iB,jmAC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oooo"] += -0.500 * contract("ia,Am,jmAB,aBkl->ijkl",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["oooo"] += -0.500 * contract("ma,Ai,jkAB,aBlm->jkil",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["oooo"] += -0.500 * contract("iA,Bm,jmaB,aAkl->ijkl",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oooo"] += -0.500 * contract("iA,Bm,jmBC,ACkl->ijkl",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oooo"] +=  0.500 * contract("mA,Ai,jkaB,aBlm->jkil",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oooo"] += -0.500 * contract("mA,Bi,jkaB,aAlm->jkil",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oooo"] += -0.500 * contract("mA,Bi,jkBC,AClm->jkil",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oooo"] += -0.250 * contract("am,mA,ijaB,ABkl->ijkl",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["oooo"] +=  0.250 * contract("Am,iA,jmBC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oooo"] += -0.250 * contract("Am,mB,ijaA,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oooo"] += -0.250 * contract("Am,mB,ijAC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oooo"] += -0.250 * contract("ma,Am,ijAB,aBkl->ijkl",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["oooo"] +=  0.250 * contract("mA,Ai,jkBC,BClm->jkil",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oooo"] += -0.250 * contract("mA,Bm,ijaB,aAkl->ijkl",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["oooo"] += -0.250 * contract("mA,Bm,ijBC,ACkl->ijkl",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["oooo"] = asym_term(fs2s2s1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2s2s1["ooov"] += -1.000 * contract("ml,iA,jlaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ooov"] += -1.000 * contract("Ab,iB,jlaA,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ooov"] += -1.000 * contract("BA,iC,jlaB,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ooov"] += -0.500 * contract("li,jA,kmaB,ABlm->jkia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ooov"] +=  0.500 * contract("il,jA,lmaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ooov"] +=  0.500 * contract("il,mA,jlaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ooov"] +=  0.500 * contract("ba,iA,jlbB,ABkl->ijka",f["vv"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ooov"] +=  0.500 * contract("Aa,iB,jlbA,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ooov"] +=  0.500 * contract("Aa,iB,jlAC,BCkl->ijka",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["ooov"] +=  0.500 * contract("Ab,iA,jlaB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ooov"] += -0.500 * contract("Ab,lB,ijaA,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ooov"] +=  0.500 * contract("BA,iB,jlaC,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ooov"] += -0.500 * contract("BA,lC,ijaB,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ooov"] += -0.250 * contract("li,mA,jkaB,ABlm->jkia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ooov"] += -0.250 * contract("ml,lA,ijaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ooov"] +=  0.250 * contract("ba,lA,ijbB,ABkl->ijka",f["vv"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ooov"] +=  0.250 * contract("Aa,lB,ijbA,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ooov"] +=  0.250 * contract("Aa,lB,ijAC,BCkl->ijka",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["ooov"] +=  0.250 * contract("Ab,lA,ijaB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ooov"] +=  0.250 * contract("BA,lB,ijaC,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ooov"] = asym_term(fs2s2s1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2s2s1["ovoo"] += -1.000 * contract("ml,Ai,jlAB,aBkm->jaik",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovoo"] += -1.000 * contract("bA,Bi,jlbB,aAkl->jaik",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovoo"] += -1.000 * contract("BA,Ci,jlBC,aAkl->jaik",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovoo"] +=  0.500 * contract("li,Aj,kmAB,aBlm->kaij",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovoo"] +=  0.500 * contract("li,Am,jmAB,aBkl->jaik",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovoo"] += -0.500 * contract("il,Aj,lmAB,aBkm->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovoo"] +=  0.500 * contract("ab,Ai,jlAB,bBkl->jaik",f["vv"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovoo"] +=  0.500 * contract("aA,Bi,jlbB,bAkl->jaik",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovoo"] +=  0.500 * contract("aA,Bi,jlBC,ACkl->jaik",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["ovoo"] +=  0.500 * contract("bA,Ai,jlbB,aBkl->jaik",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovoo"] += -0.500 * contract("bA,Bl,ilbB,aAjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovoo"] +=  0.500 * contract("BA,Ai,jlBC,aCkl->jaik",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovoo"] += -0.500 * contract("BA,Cl,ilBC,aAjk->iajk",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovoo"] += -0.250 * contract("il,Am,lmAB,aBjk->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovoo"] += -0.250 * contract("ml,Am,ilAB,aBjk->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovoo"] +=  0.250 * contract("ab,Al,ilAB,bBjk->iajk",f["vv"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovoo"] +=  0.250 * contract("aA,Bl,ilbB,bAjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovoo"] +=  0.250 * contract("aA,Bl,ilBC,ACjk->iajk",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
	fs2s2s1["ovoo"] +=  0.250 * contract("bA,Al,ilbB,aBjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovoo"] +=  0.250 * contract("BA,Al,ilBC,aCjk->iajk",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovoo"] = asym_term(fs2s2s1["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	fs2s2s1["oovv"] += -2.000 * contract("Ak,Bl,ikaB,jlbA->ijab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"])
	fs2s2s1["oovv"] +=  1.000 * contract("Ak,Bl,klaB,ijbA->ijab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"])
	fs2s2s1["oovv"] += -1.000 * contract("ka,iA,jlbB,ABkl->ijab",f["ov"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["oovv"] += -0.500 * contract("ka,lA,ijbB,ABkl->ijab",f["ov"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["oovv"] = asym_term(fs2s2s1["oovv"],"oovv")

	fs2s2s1["ovov"] += -0.250 * contract("Ai,kB,jlaA,bBkl->jbia",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovov"] += -0.250 * contract("ak,iA,klbB,ABjl->iajb",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ovov"] += -0.250 * contract("ak,lA,ikbB,ABjl->iajb",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
	fs2s2s1["ovov"] +=  0.250 * contract("Ak,iA,klaB,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovov"] +=  0.250 * contract("Ak,lA,ikaB,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovov"] += -0.250 * contract("Ak,iB,klaA,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovov"] +=  0.250 * contract("Ak,kB,ilaA,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovov"] += -0.250 * contract("ka,Ai,jlAB,bBkl->jbia",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovov"] += -0.250 * contract("ka,Al,ilAB,bBjk->ibja",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["ovov"] += -0.250 * contract("iA,Bk,klaB,bAjl->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovov"] +=  0.250 * contract("kA,Ai,jlaB,bBkl->jbia",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovov"] += -0.250 * contract("kA,Bi,jlaB,bAkl->jbia",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovov"] +=  0.250 * contract("kA,Bk,ilaB,bAjl->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovov"] +=  0.250 * contract("kA,Al,ilaB,bBjk->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovov"] +=  0.125 * contract("Ai,jB,klaA,bBkl->jbia",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovov"] +=  0.125 * contract("iA,Bj,klaB,bAkl->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	fs2s2s1["vvoo"] += -2.000 * contract("kA,lB,aAil,bBjk->abij",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"])
	fs2s2s1["vvoo"] += -1.000 * contract("ak,Ai,klAB,bBjl->abij",f["vo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["vvoo"] += -1.000 * contract("kA,lB,aAij,bBkl->abij",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"])
	fs2s2s1["vvoo"] += -0.500 * contract("ak,Al,klAB,bBij->abij",f["vo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
	fs2s2s1["vvoo"] = asym_term(fs2s2s1["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	fs2s2s1["ovvv"] += -0.500 * contract("Aa,jB,ikbA,cBjk->icab",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovvv"] +=  0.250 * contract("Aa,iB,jkbA,cBjk->icab",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
	fs2s2s1["ovvv"] = asym_term(fs2s2s1["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	fs2s2s1["vvov"] += -0.500 * contract("aA,Bj,jkbB,cAik->acib",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vvov"] +=  0.250 * contract("aA,Bi,jkbB,cAjk->acib",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
	fs2s2s1["vvov"] = asym_term(fs2s2s1["vvov"],"vvov")

	if(inc_3_body):
		fs2s2s1["oooooo"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ))
		fs2s2s1["ooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2s1["oovooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s1["oooovv"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s2s1["oovoov"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2s1["ovvooo"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s1["oovovv"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs2s2s1["ovvoov"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))

		# oooooo = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ))
		fs2s2s1["oooooo"] += -0.250 * contract("ai,jA,klaB,ABmn->jklimn",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs2s2s1["oooooo"] += -0.250 * contract("Ai,jB,klaA,aBmn->jklimn",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs2s2s1["oooooo"] += -0.250 * contract("Ai,jB,klAC,BCmn->jklimn",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"])
		fs2s2s1["oooooo"] += -0.250 * contract("ia,Aj,klAB,aBmn->ikljmn",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s2s1["oooooo"] += -0.250 * contract("iA,Bj,klaB,aAmn->ikljmn",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs2s2s1["oooooo"] += -0.250 * contract("iA,Bj,klBC,ACmn->ikljmn",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
		fs2s2s1["oooooo"] = asym_term(fs2s2s1["oooooo"],"oooooo")

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2s1["ooooov"] += -(1./6.) * contract("ni,jA,klaB,ABmn->jklima",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs2s2s1["ooooov"] +=  (1./6.) * contract("in,jA,knaB,ABlm->ijklma",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs2s2s1["ooooov"] +=  (1./6.) * contract("Ab,iB,jkaA,bBlm->ijklma",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs2s2s1["ooooov"] +=  (1./6.) * contract("BA,iC,jkaB,AClm->ijklma",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs2s2s1["ooooov"] += -(1./12.) * contract("ba,iA,jkbB,ABlm->ijklma",f["vv"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs2s2s1["ooooov"] += -(1./12.) * contract("Aa,iB,jkbA,bBlm->ijklma",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs2s2s1["ooooov"] += -(1./12.) * contract("Aa,iB,jkAC,BClm->ijklma",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"])
		fs2s2s1["ooooov"] += -(1./12.) * contract("Ab,iA,jkaB,bBlm->ijklma",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs2s2s1["ooooov"] += -(1./12.) * contract("BA,iB,jkaC,AClm->ijklma",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs2s2s1["ooooov"] = asym_term(fs2s2s1["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s1["oovooo"] +=  (1./6.) * contract("ni,Aj,klAB,aBmn->klaijm",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s2s1["oovooo"] += -(1./6.) * contract("in,Aj,knAB,aBlm->ikajlm",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s2s1["oovooo"] +=  (1./6.) * contract("bA,Bi,jkbB,aAlm->jkailm",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs2s2s1["oovooo"] +=  (1./6.) * contract("BA,Ci,jkBC,aAlm->jkailm",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s2s1["oovooo"] += -(1./12.) * contract("ab,Ai,jkAB,bBlm->jkailm",f["vv"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s2s1["oovooo"] += -(1./12.) * contract("aA,Bi,jkbB,bAlm->jkailm",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs2s2s1["oovooo"] += -(1./12.) * contract("aA,Bi,jkBC,AClm->jkailm",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"])
		fs2s2s1["oovooo"] += -(1./12.) * contract("bA,Ai,jkbB,aBlm->jkailm",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs2s2s1["oovooo"] += -(1./12.) * contract("BA,Ai,jkBC,aClm->jkailm",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s2s1["oovooo"] = asym_term(fs2s2s1["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s2s1["oooovv"] += -(1./3.) * contract("Am,Bi,jmaB,klbA->jkliab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"])
		fs2s2s1["oooovv"] +=  (1./6.) * contract("ma,iA,jkbB,ABlm->ijklab",f["ov"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs2s2s1["oooovv"] = asym_term(fs2s2s1["oooovv"],"oooovv")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2s1["oovoov"] +=  (1./9.) * contract("Ai,jB,kmaA,bBlm->jkbila",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs2s2s1["oovoov"] +=  (1./9.) * contract("iA,Bj,kmaB,bAlm->ikbjla",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs2s2s1["oovoov"] +=  (1./18.) * contract("am,iA,jmbB,ABkl->ijaklb",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"])
		fs2s2s1["oovoov"] += -(1./18.) * contract("Am,iA,jmaB,bBkl->ijbkla",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs2s2s1["oovoov"] +=  (1./18.) * contract("ma,Ai,jkAB,bBlm->jkbila",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s2s1["oovoov"] += -(1./18.) * contract("mA,Ai,jkaB,bBlm->jkbila",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs2s2s1["oovoov"] = asym_term(fs2s2s1["oovoov"],"oovoov")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s1["ovvooo"] +=  (1./3.) * contract("mA,iB,aAjk,bBlm->iabjkl",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"])
		fs2s2s1["ovvooo"] +=  (1./6.) * contract("am,Ai,jmAB,bBkl->jabikl",f["vo"],t1["Vo"],t2["ooVV"],t2["vVoo"])
		fs2s2s1["ovvooo"] = asym_term(fs2s2s1["ovvooo"],"ovvooo")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs2s2s1["oovovv"] += -(1./9.) * contract("Aa,iB,jlbA,cBkl->ijckab",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"])
		fs2s2s1["oovovv"] = asym_term(fs2s2s1["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2s1["ovvoov"] += -(1./9.) * contract("aA,Bi,jlbB,cAkl->jacikb",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"])
		fs2s2s1["ovvoov"] = asym_term(fs2s2s1["ovvoov"],"ovvoov")

	return fs2s2s1 

def fn_s2_s2_s2(f,t2,inc_3_body=True,inc_4_body=True):
	# [[[Fn,S_2ext],S_2ext],S_2ext]
	# for sizing arrays
	n_occ = f["oo"].shape[0]
	n_virt_int = f["vv"].shape[0]
	# initialize
	fs2s2s2 = {
		"ov": np.zeros((n_occ,n_virt_int)),
		"vo": np.zeros((n_virt_int,n_occ)),
		"ooov": np.zeros((n_occ,n_occ,n_occ,n_virt_int)),
		"ovoo": np.zeros((n_occ,n_virt_int,n_occ,n_occ)),
		"oovv": np.zeros((n_occ,n_occ,n_virt_int,n_virt_int)),
		"vvoo": np.zeros((n_virt_int,n_virt_int,n_occ,n_occ)),
		"ovvv": np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int)),
		"vvov": np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	} 
	# Populate [[[Fn,S_2ext],S_2ext],S_2ext]
	fs2s2s2["ov"] += -4.000 * contract("bj,jkaA,ilbB,ABkl->ia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ov"] += -4.000 * contract("Aj,jkaB,ilbA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ov"] +=  4.000 * contract("Aj,jkbB,ilaA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ov"] += -4.000 * contract("Aj,ikAB,jlaC,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ov"] += -2.000 * contract("Aj,ijaB,klbA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ov"] +=  2.000 * contract("Aj,ijbB,klaA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ov"] +=  2.000 * contract("Aj,klAB,ijaC,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ov"] +=  2.000 * contract("Aj,jkBC,ilaA,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ov"] += -2.000 * contract("bj,ijaA,klbB,ABkl->ia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ov"] +=  1.000 * contract("bj,ikaA,jlbB,ABkl->ia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ov"] +=  1.000 * contract("Aj,ikaB,jlbA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ov"] += -1.000 * contract("Aj,ikbB,jlaA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ov"] +=  1.000 * contract("Aj,jkAB,ilaC,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ov"] +=  1.000 * contract("Aj,ijBC,klaA,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ov"] += -0.500 * contract("bj,klaA,ijbB,ABkl->ia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ov"] += -0.500 * contract("Aj,klaB,ijbA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ov"] +=  0.500 * contract("Aj,ijAB,klaC,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ov"] += -0.500 * contract("Aj,ikBC,jlaA,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])

	fs2s2s2["vo"] += -4.000 * contract("jb,klAB,bBil,aAjk->ai",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vo"] += -4.000 * contract("jA,klbB,bAil,aBjk->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vo"] +=  4.000 * contract("jA,klbB,aAil,bBjk->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vo"] += -4.000 * contract("jA,klBC,ABik,aCjl->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["vo"] += -2.000 * contract("jb,klAB,bBkl,aAij->ai",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vo"] +=  2.000 * contract("jA,klbB,aAkl,bBij->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vo"] += -2.000 * contract("jA,klbB,bAkl,aBij->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vo"] +=  2.000 * contract("jA,klBC,BCjk,aAil->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["vo"] +=  2.000 * contract("jA,klBC,ABkl,aCij->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["vo"] +=  1.000 * contract("jb,klAB,aBil,bAjk->ai",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vo"] += -1.000 * contract("jA,klbB,aBil,bAjk->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vo"] +=  1.000 * contract("jA,klbB,bBil,aAjk->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vo"] +=  1.000 * contract("jA,klBC,BCij,aAkl->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["vo"] +=  1.000 * contract("jA,klBC,ABjk,aCil->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["vo"] +=  0.500 * contract("jb,klAB,aBkl,bAij->ai",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vo"] += -0.500 * contract("jA,klbB,aBkl,bAij->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vo"] +=  0.500 * contract("jA,klBC,ABij,aCkl->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["vo"] += -0.500 * contract("jA,klBC,BCik,aAjl->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2s2s2["ooov"] +=  2.000 * contract("bl,ilaA,jmbB,ABkm->ijka",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] +=  2.000 * contract("Al,ilaB,jmbA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ooov"] += -2.000 * contract("Al,ilbB,jmaA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ooov"] +=  2.000 * contract("Al,imAB,jlaC,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] += -1.000 * contract("bl,lmaA,ijbB,ABkm->ijka",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] += -1.000 * contract("Al,lmaB,ijbA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ooov"] +=  1.000 * contract("Al,ijAB,lmaC,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] += -1.000 * contract("Al,ilBC,jmaA,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] +=  0.750 * contract("Al,lmbB,ijaA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ooov"] += -0.500 * contract("bi,jlaA,kmbB,ABlm->jkia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] += -0.500 * contract("Ai,jlaB,kmbA,bBlm->jkia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ooov"] +=  0.500 * contract("Ai,jlbB,kmaA,bBlm->jkia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ooov"] +=  0.500 * contract("Ai,jlAB,kmaC,BClm->jkia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] += -0.500 * contract("bl,imaA,jlbB,ABkm->ijka",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] += -0.500 * contract("Al,imaB,jlbA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ooov"] += -0.500 * contract("Al,ilAB,jmaC,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] +=  0.375 * contract("Al,lmBC,ijaA,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] +=  0.250 * contract("Ai,jlBC,kmaA,BClm->jkia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] += -0.250 * contract("bl,ijaA,lmbB,ABkm->ijka",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] += -0.250 * contract("Al,ijaB,lmbA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ooov"] +=  0.250 * contract("Al,ijbB,lmaA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ooov"] +=  0.250 * contract("Al,lmAB,ijaC,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] +=  0.125 * contract("bi,jkaA,lmbB,ABlm->jkia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] +=  0.125 * contract("bi,lmaA,jkbB,ABlm->jkia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] +=  0.125 * contract("Ai,jkaB,lmbA,bBlm->jkia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ooov"] +=  0.125 * contract("Ai,lmaB,jkbA,bBlm->jkia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ooov"] += -0.125 * contract("Ai,jkbB,lmaA,bBlm->jkia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ooov"] += -0.125 * contract("Ai,jkAB,lmaC,BClm->jkia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] += -0.125 * contract("Ai,lmAB,jkaC,BClm->jkia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] +=  0.125 * contract("Al,ijBC,lmaA,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] += -0.0625 * contract("Ai,jkBC,lmaA,BClm->jkia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ooov"] = asym_term(fs2s2s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2s2s2["ovoo"] += -2.000 * contract("lb,imAB,bBjm,aAkl->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  2.000 * contract("lA,imbB,aAjm,bBkl->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -2.000 * contract("lA,imbB,bAjm,aBkl->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  2.000 * contract("lA,imBC,ABjm,aCkl->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -1.000 * contract("lb,imAB,bBjk,aAlm->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -1.000 * contract("lA,imbB,bAjk,aBlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  1.000 * contract("lA,imBC,ABjk,aClm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -1.000 * contract("lA,imBC,BCjl,aAkm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  0.750 * contract("lA,imbB,aAjk,bBlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -0.500 * contract("ib,lmAB,aBjm,bAkl->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  0.500 * contract("lb,imAB,aBjm,bAkl->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  0.500 * contract("iA,lmbB,aBjm,bAkl->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -0.500 * contract("iA,lmbB,bBjm,aAkl->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  0.500 * contract("iA,lmBC,ABjl,aCkm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -0.500 * contract("lA,imbB,aBjm,bAkl->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -0.500 * contract("lA,imBC,ABjl,aCkm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  0.375 * contract("lA,imBC,BClm,aAjk->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  0.250 * contract("lb,imAB,aBjk,bAlm->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  0.250 * contract("iA,lmBC,BCjl,aAkm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -0.250 * contract("lA,imbB,aBjk,bAlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  0.250 * contract("lA,imbB,bBjk,aAlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  0.250 * contract("lA,imBC,ABlm,aCjk->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -0.125 * contract("ib,lmAB,aBjk,bAlm->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -0.125 * contract("ib,lmAB,bAjk,aBlm->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  0.125 * contract("iA,lmbB,aBjk,bAlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  0.125 * contract("iA,lmbB,bAjk,aBlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -0.125 * contract("iA,lmbB,bBjk,aAlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -0.125 * contract("iA,lmBC,ABjk,aClm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -0.125 * contract("iA,lmBC,ABlm,aCjk->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] +=  0.125 * contract("lA,imBC,BCjk,aAlm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] += -0.0625 * contract("iA,lmBC,BCjk,aAlm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["ovoo"] = asym_term(fs2s2s2["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	fs2s2s2["oovv"] +=  3.000 * contract("lk,imaA,jkbB,ABlm->ijab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] += -3.000 * contract("Ac,ikaB,jlbA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["oovv"] += -3.000 * contract("BA,ikaC,jlbB,ACkl->ijab",f["VV"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] +=  1.500 * contract("lk,ijaA,kmbB,ABlm->ijab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] +=  1.000 * contract("ik,jlaA,kmbB,ABlm->ijab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] += -1.000 * contract("ca,ikbA,jlcB,ABkl->ijab",f["vv"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] += -1.000 * contract("Aa,ikbB,jlcA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["oovv"] +=  1.000 * contract("Aa,ikcB,jlbA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["oovv"] +=  1.000 * contract("Aa,ikAB,jlbC,BCkl->ijab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] +=  0.750 * contract("Ac,ijaB,klbA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["oovv"] +=  0.750 * contract("Ac,klaB,ijbA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["oovv"] +=  0.750 * contract("BA,ijaC,klbB,ACkl->ijab",f["VV"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] +=  0.750 * contract("BA,klaC,ijbB,ACkl->ijab",f["VV"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] += -0.500 * contract("ik,lmaA,jkbB,ABlm->ijab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] +=  0.500 * contract("Aa,ikBC,jlbA,BCkl->ijab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] +=  0.250 * contract("ca,ijbA,klcB,ABkl->ijab",f["vv"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] +=  0.250 * contract("ca,klbA,ijcB,ABkl->ijab",f["vv"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] +=  0.250 * contract("Aa,ijbB,klcA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["oovv"] +=  0.250 * contract("Aa,klbB,ijcA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["oovv"] += -0.250 * contract("Aa,ijcB,klbA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["oovv"] += -0.250 * contract("Aa,ijAB,klbC,BCkl->ijab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] += -0.250 * contract("Aa,klAB,ijbC,BCkl->ijab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] += -0.125 * contract("Aa,ijBC,klbA,BCkl->ijab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["oovv"] = asym_term(fs2s2s2["oovv"],"oovv")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	fs2s2s2["vvoo"] += -3.000 * contract("lk,kmAB,aBim,bAjl->abij",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] +=  3.000 * contract("cA,klcB,aBil,bAjk->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] +=  3.000 * contract("BA,klBC,aCil,bAjk->abij",f["VV"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] += -1.500 * contract("lk,kmAB,aBij,bAlm->abij",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] +=  1.000 * contract("ki,lmAB,aBjm,bAkl->abij",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] += -1.000 * contract("ac,klAB,bBil,cAjk->abij",f["vv"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] +=  1.000 * contract("aA,klcB,bBil,cAjk->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] += -1.000 * contract("aA,klcB,cBil,bAjk->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] +=  1.000 * contract("aA,klBC,ABik,bCjl->abij",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] += -0.750 * contract("cA,klcB,aAij,bBkl->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] +=  0.750 * contract("cA,klcB,aBij,bAkl->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] += -0.750 * contract("BA,klBC,aAij,bCkl->abij",f["VV"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] +=  0.750 * contract("BA,klBC,aCij,bAkl->abij",f["VV"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] +=  0.500 * contract("ki,lmAB,aBlm,bAjk->abij",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] +=  0.500 * contract("aA,klBC,BCik,bAjl->abij",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] += -0.250 * contract("ac,klAB,bBij,cAkl->abij",f["vv"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] += -0.250 * contract("ac,klAB,cAij,bBkl->abij",f["vv"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] +=  0.250 * contract("aA,klcB,bBij,cAkl->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] +=  0.250 * contract("aA,klcB,cAij,bBkl->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] += -0.250 * contract("aA,klcB,cBij,bAkl->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] += -0.250 * contract("aA,klBC,ABij,bCkl->abij",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] += -0.250 * contract("aA,klBC,ABkl,bCij->abij",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] += -0.125 * contract("aA,klBC,BCij,bAkl->abij",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
	fs2s2s2["vvoo"] = asym_term(fs2s2s2["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	fs2s2s2["ovvv"] +=  2.000 * contract("Aj,jkaB,ilbA,cBkl->icab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ovvv"] +=  1.000 * contract("Aj,ijaB,klbA,cBkl->icab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ovvv"] += -0.500 * contract("aj,ikbA,jlcB,ABkl->iabc",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ovvv"] += -0.500 * contract("Aj,ikaB,jlbA,cBkl->icab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
	fs2s2s2["ovvv"] +=  0.250 * contract("aj,klbA,ijcB,ABkl->iabc",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
	fs2s2s2["ovvv"] = asym_term(fs2s2s2["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	fs2s2s2["vvov"] += -2.000 * contract("jA,klaB,bAil,cBjk->bcia",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvov"] += -1.000 * contract("jA,klaB,bAkl,cBij->bcia",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvov"] += -0.500 * contract("ja,klAB,bBil,cAjk->bcia",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvov"] +=  0.500 * contract("jA,klaB,bBil,cAjk->bcia",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvov"] += -0.250 * contract("ja,klAB,bBkl,cAij->bcia",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
	fs2s2s2["vvov"] = asym_term(fs2s2s2["vvov"],"vvov")

	if(inc_3_body):
		fs2s2s2["ooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2s2["oovooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s2["oooovv"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s2s2["ovvooo"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s2["ooovvv"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		fs2s2s2["oovovv"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs2s2s2["ovvoov"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2s2["vvvooo"] = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s2["oovvvv"] = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		fs2s2s2["vvvoov"] = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2s2["ooooov"] += -(1./3.) * contract("bn,inaA,jkbB,ABlm->ijklma",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooov"] += -(1./3.) * contract("An,inaB,jkbA,bBlm->ijklma",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["ooooov"] +=  (1./3.) * contract("An,ijAB,knaC,BClm->ijklma",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooov"] +=  0.250 * contract("An,inbB,jkaA,bBlm->ijklma",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["ooooov"] += -(1./6.) * contract("bi,jkaA,lnbB,ABmn->jklima",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooov"] += -(1./6.) * contract("bi,jnaA,klbB,ABmn->jklima",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooov"] += -(1./6.) * contract("Ai,jkaB,lnbA,bBmn->jklima",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["ooooov"] += -(1./6.) * contract("Ai,jnaB,klbA,bBmn->jklima",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["ooooov"] +=  (1./6.) * contract("Ai,jkbB,lnaA,bBmn->jklima",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["ooooov"] +=  (1./6.) * contract("Ai,jkAB,lnaC,BCmn->jklima",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooov"] +=  (1./6.) * contract("Ai,jnAB,klaC,BCmn->jklima",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooov"] +=  0.125 * contract("An,inBC,jkaA,BClm->ijklma",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooov"] +=  (1./12.) * contract("Ai,jkBC,lnaA,BCmn->jklima",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooov"] += -(1./12.) * contract("bn,ijaA,knbB,ABlm->ijklma",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooov"] += -(1./12.) * contract("An,ijaB,knbA,bBlm->ijklma",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["ooooov"] +=  (1./12.) * contract("An,inAB,jkaC,BClm->ijklma",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooov"] = asym_term(fs2s2s2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s2["oovooo"] += -(1./3.) * contract("nb,ijAB,bBkl,aAmn->ijaklm",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] += -(1./3.) * contract("nA,ijbB,bAkl,aBmn->ijaklm",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] +=  (1./3.) * contract("nA,ijBC,ABkl,aCmn->ijaklm",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] +=  0.250 * contract("nA,ijbB,aAkl,bBmn->ijaklm",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] +=  (1./6.) * contract("ib,jnAB,aBkl,bAmn->ijaklm",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] +=  (1./6.) * contract("ib,jnAB,bAkl,aBmn->ijaklm",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] += -(1./6.) * contract("iA,jnbB,aBkl,bAmn->ijaklm",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] += -(1./6.) * contract("iA,jnbB,bAkl,aBmn->ijaklm",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] +=  (1./6.) * contract("iA,jnbB,bBkl,aAmn->ijaklm",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] +=  (1./6.) * contract("iA,jnBC,ABkl,aCmn->ijaklm",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] +=  (1./6.) * contract("iA,jnBC,ABkn,aClm->ijaklm",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] +=  0.125 * contract("nA,ijBC,BCkn,aAlm->ijaklm",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] +=  (1./12.) * contract("nb,ijAB,aBkl,bAmn->ijaklm",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] +=  (1./12.) * contract("iA,jnBC,BCkl,aAmn->ijaklm",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] += -(1./12.) * contract("nA,ijbB,aBkl,bAmn->ijaklm",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] +=  (1./12.) * contract("nA,ijBC,ABkn,aClm->ijaklm",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
		fs2s2s2["oovooo"] = asym_term(fs2s2s2["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s2s2["oooovv"] +=  0.500 * contract("nm,ijaA,kmbB,ABln->ijklab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooovv"] +=  0.500 * contract("Ac,ijaB,kmbA,cBlm->ijklab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oooovv"] +=  0.500 * contract("Ac,imaB,jkbA,cBlm->ijklab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oooovv"] +=  0.500 * contract("BA,ijaC,kmbB,AClm->ijklab",f["VV"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooovv"] +=  0.500 * contract("BA,imaC,jkbB,AClm->ijklab",f["VV"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooovv"] +=  (1./3.) * contract("mi,jkaA,lnbB,ABmn->jkliab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooovv"] +=  (1./3.) * contract("im,jnaA,kmbB,ABln->ijklab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooovv"] +=  (1./6.) * contract("im,jkaA,mnbB,ABln->ijklab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooovv"] +=  (1./6.) * contract("ca,ijbA,kmcB,ABlm->ijklab",f["vv"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooovv"] +=  (1./6.) * contract("ca,imbA,jkcB,ABlm->ijklab",f["vv"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooovv"] +=  (1./6.) * contract("Aa,ijbB,kmcA,cBlm->ijklab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oooovv"] +=  (1./6.) * contract("Aa,imbB,jkcA,cBlm->ijklab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oooovv"] += -(1./6.) * contract("Aa,ijcB,kmbA,cBlm->ijklab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oooovv"] += -(1./6.) * contract("Aa,ijAB,kmbC,BClm->ijklab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooovv"] += -(1./6.) * contract("Aa,imAB,jkbC,BClm->ijklab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooovv"] += -(1./12.) * contract("Aa,ijBC,kmbA,BClm->ijklab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooovv"] = asym_term(fs2s2s2["oooovv"],"oooovv")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s2["ovvooo"] += -0.500 * contract("nm,imAB,aBjk,bAln->iabjkl",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] += -0.500 * contract("cA,imcB,aAjk,bBlm->iabjkl",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] +=  0.500 * contract("cA,imcB,aBjk,bAlm->iabjkl",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] += -0.500 * contract("BA,imBC,aAjk,bClm->iabjkl",f["VV"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] +=  0.500 * contract("BA,imBC,aCjk,bAlm->iabjkl",f["VV"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] += -(1./3.) * contract("mi,jnAB,aBkn,bAlm->jabikl",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] += -(1./3.) * contract("im,mnAB,aBjk,bAln->iabjkl",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] += -(1./6.) * contract("mi,jnAB,aBkl,bAmn->jabikl",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] += -(1./6.) * contract("ac,imAB,bBjk,cAlm->iabjkl",f["vv"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] += -(1./6.) * contract("ac,imAB,cAjk,bBlm->iabjkl",f["vv"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] +=  (1./6.) * contract("aA,imcB,bBjk,cAlm->iabjkl",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] +=  (1./6.) * contract("aA,imcB,cAjk,bBlm->iabjkl",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] += -(1./6.) * contract("aA,imcB,cBjk,bAlm->iabjkl",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] += -(1./6.) * contract("aA,imBC,ABjk,bClm->iabjkl",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] += -(1./6.) * contract("aA,imBC,ABjm,bCkl->iabjkl",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] += -(1./12.) * contract("aA,imBC,BCjk,bAlm->iabjkl",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
		fs2s2s2["ovvooo"] = asym_term(fs2s2s2["ovvooo"],"ovvooo")

		# ooovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		fs2s2s2["ooovvv"] += 1.000 * contract("la,ijbA,kmcB,ABlm->ijkabc",f["ov"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooovvv"] = asym_term(fs2s2s2["ooovvv"],"ooovvv")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs2s2s2["oovovv"] += -(4./9.) * contract("Al,ilaB,jmbA,cBkm->ijckab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oovovv"] +=  (1./6.) * contract("Al,lmaB,ijbA,cBkm->ijckab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oovovv"] +=  (1./9.) * contract("Ai,jlaB,kmbA,cBlm->jkciab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oovovv"] +=  (1./9.) * contract("al,imbA,jlcB,ABkm->ijakbc",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oovovv"] +=  (1./18.) * contract("al,ijbA,lmcB,ABkm->ijakbc",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oovovv"] +=  (1./18.) * contract("Al,ijaB,lmbA,cBkm->ijckab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oovovv"] += -(1./36.) * contract("Ai,jkaB,lmbA,cBlm->jkciab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oovovv"] = asym_term(fs2s2s2["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2s2["ovvoov"] += -(4./9.) * contract("lA,imaB,bAjm,cBkl->ibcjka",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvoov"] += -(1./6.) * contract("lA,imaB,bAjk,cBlm->ibcjka",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvoov"] += -(1./9.) * contract("la,imAB,bBjm,cAkl->ibcjka",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvoov"] += -(1./9.) * contract("iA,lmaB,bBjm,cAkl->ibcjka",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvoov"] += -(1./18.) * contract("la,imAB,bBjk,cAlm->ibcjka",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvoov"] +=  (1./18.) * contract("lA,imaB,bBjk,cAlm->ibcjka",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvoov"] += -(1./36.) * contract("iA,lmaB,bBjk,cAlm->ibcjka",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvoov"] = asym_term(fs2s2s2["ovvoov"],"ovvoov")

		# vvvooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s2["vvvooo"] += -1.000 * contract("al,lmAB,bBij,cAkm->abcijk",f["vo"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["vvvooo"] = asym_term(fs2s2s2["vvvooo"],"vvvooo")

		# oovvvv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		fs2s2s2["oovvvv"] +=  (1./3.) * contract("Aa,ikbB,jlcA,dBkl->ijdabc",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oovvvv"] += -(1./12.) * contract("Aa,ijbB,klcA,dBkl->ijdabc",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oovvvv"] = asym_term(fs2s2s2["oovvvv"],"oovvvv")

		# vvvoov = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2s2["vvvoov"] += -(1./3.) * contract("aA,klbB,cBil,dAjk->acdijb",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["vvvoov"] += -(1./12.) * contract("aA,klbB,cBij,dAkl->acdijb",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["vvvoov"] = asym_term(fs2s2s2["vvvoov"],"vvvoov")

	if(inc_4_body):
		fs2s2s2["ooooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2s2["ooovoooo"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		fs2s2s2["oooooovv"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s2s2["oovvoooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		fs2s2s2["ooooovvv"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		fs2s2s2["ooovoovv"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s2s2["oovvooov"] = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2s2["ovvvoooo"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		fs2s2s2["ooovovvv"] = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int,n_virt_int))
		fs2s2s2["ovvvooov"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))

		# ooooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2s2["ooooooov"] +=  0.03125 * contract("bi,jkaA,lmbB,ABnp->jklminpa",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooooov"] +=  0.03125 * contract("Ai,jkaB,lmbA,bBnp->jklminpa",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["ooooooov"] += -0.03125 * contract("Ai,jkAB,lmaC,BCnp->jklminpa",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooooov"] = asym_term(fs2s2s2["ooooooov"],"ooooooov")

		# ooovoooo = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		fs2s2s2["ooovoooo"] += -0.03125 * contract("ib,jkAB,aBlm,bAnp->ijkalmnp",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ooovoooo"] +=  0.03125 * contract("iA,jkbB,aBlm,bAnp->ijkalmnp",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ooovoooo"] += -0.03125 * contract("iA,jkBC,ABlm,aCnp->ijkalmnp",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
		fs2s2s2["ooovoooo"] = asym_term(fs2s2s2["ooovoooo"],"ooovoooo")

		# oooooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s2s2["oooooovv"] +=  0.0625 * contract("Ac,ijaB,klbA,cBmn->ijklmnab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oooooovv"] +=  0.0625 * contract("BA,ijaC,klbB,ACmn->ijklmnab",f["VV"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooooovv"] += -(1./24.) * contract("pi,jkaA,lmbB,ABnp->jklminab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooooovv"] += -(1./24.) * contract("ip,jkaA,lpbB,ABmn->ijklmnab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooooovv"] +=  (1./48.) * contract("ca,ijbA,klcB,ABmn->ijklmnab",f["vv"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooooovv"] +=  (1./48.) * contract("Aa,ijbB,klcA,cBmn->ijklmnab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["oooooovv"] += -(1./48.) * contract("Aa,ijAB,klbC,BCmn->ijklmnab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["oooooovv"] = asym_term(fs2s2s2["oooooovv"],"oooooovv")

		# oovvoooo = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		fs2s2s2["oovvoooo"] +=  0.0625 * contract("cA,ijcB,aBkl,bAmn->ijabklmn",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovvoooo"] +=  0.0625 * contract("BA,ijBC,aCkl,bAmn->ijabklmn",f["VV"],t2["ooVV"],t2["vVoo"],t2["vVoo"])	
		fs2s2s2["oovvoooo"] +=  (1./24.) * contract("pi,jkAB,aBlm,bAnp->jkabilmn",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovvoooo"] += -(1./24.) * contract("ip,jpAB,aAkl,bBmn->ijabklmn",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovvoooo"] += -(1./48.) * contract("ac,ijAB,bBkl,cAmn->ijabklmn",f["vv"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovvoooo"] +=  (1./48.) * contract("aA,ijcB,bBkl,cAmn->ijabklmn",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovvoooo"] += -(1./48.) * contract("aA,ijBC,ABkl,bCmn->ijabklmn",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"])
		fs2s2s2["oovvoooo"] = asym_term(fs2s2s2["oovvoooo"],"oovvoooo")

		# ooooovvv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		fs2s2s2["ooooovvv"] +=  0.0625 * contract("na,ijbA,klcB,ABmn->ijklmabc",f["ov"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooooovvv"] = asym_term(fs2s2s2["ooooovvv"],"ooooovvv")

		# ooovoovv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s2s2["ooovoovv"] +=  0.03125 * contract("An,inaB,jkbA,cBlm->ijkclmab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["ooovoovv"] +=  (1./48.) * contract("Ai,jkaB,lnbA,cBmn->jklcimab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["ooovoovv"] +=  (1./96.) * contract("an,ijbA,kncB,ABlm->ijkalmbc",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"])
		fs2s2s2["ooovoovv"] = asym_term(fs2s2s2["ooovoovv"],"ooovoovv")

		# oovvooov = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2s2["oovvooov"] += -0.03125 * contract("nA,ijaB,bAkl,cBmn->ijbcklma",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovvooov"] +=  (1./48.) * contract("iA,jnaB,bBkl,cAmn->ijbcklma",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovvooov"] += -(1./96.) * contract("na,ijAB,bBkl,cAmn->ijbcklma",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["oovvooov"] = asym_term(fs2s2s2["oovvooov"],"oovvooov")

		# ovvvoooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		fs2s2s2["ovvvoooo"] +=  0.0625 * contract("an,inAB,bAjk,cBlm->iabcjklm",f["vo"],t2["ooVV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvvoooo"] = asym_term(fs2s2s2["ovvvoooo"],"ovvvoooo")

		# ooovovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int,n_virt_int))
		fs2s2s2["ooovovvv"] += -0.03125 * contract("Aa,ijbB,kmcA,dBlm->ijkdlabc",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"])
		fs2s2s2["ooovovvv"] = asym_term(fs2s2s2["ooovovvv"],"ooovovvv")

		# ovvvooov = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2s2["ovvvooov"] += -0.03125 * contract("aA,imbB,cBjk,dAlm->iacdjklb",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"])
		fs2s2s2["ovvvooov"] = asym_term(fs2s2s2["ovvvooov"],"ovvvooov")

	return fs2s2s2 


def eff_ham_a1(fmat,vten,n_a,n_b,n_act):
    # For sizing
    n_occ = n_a + n_b
    n_orb = int(fmat.shape[0]/2)

    # Initialize return values
    constant = 0.0
    one_body_as = cp.deepcopy(fmat[0:2*n_act,0:2*n_act])
    two_body_as = cp.deepcopy(vten[0:2*n_act,0:2*n_act,0:2*n_act,0:2*n_act])

    return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act)

def eff_ham_a2(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=False):
    # For sizing
    n_occ = n_a + n_b
    n_orb = int(fmat.shape[0]/2)


    # Initialize return values
    constant = 0.0
    one_body_as = cp.deepcopy(fmat[0:2*n_act,0:2*n_act])
    two_body_as = cp.deepcopy(vten[0:2*n_act,0:2*n_act,0:2*n_act,0:2*n_act])
    if(three_body):
        three_body_as = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))

    # Initialize dictionaries
    fdic = one_body_mat2dic(fmat,n_occ,n_act,n_orb)
    vdic = two_body_ten2dic(vten,n_occ,n_act,n_orb)
    t1dic = t1_mat2dic(t1_to_ext(t1_amps,n_act),n_act)
    t2dic = t2_ten2dic(t2_to_ext(t2_amps,n_act),n_act)

    # Begin computing DUCC terms
    # [Fn,S1]
    fn_s1_dic = fn_s1(fdic,t1dic)
    constant += fn_s1_dic["c"]
    one_body_as += one_body_dic2mat(fn_s1_dic,n_occ,n_act,n_act)

    # [Fn,S2]
    fn_s2_dic = fn_s2(fdic,t2dic)
    one_body_as += one_body_dic2mat(fn_s2_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(fn_s2_dic,n_occ,n_act,n_act)

    # [Wn,S2]
    wn_s2_dic = wn_s2(vdic,t2dic,inc_3_body=three_body)
    constant += wn_s2_dic["c"]
    one_body_as += one_body_dic2mat(wn_s2_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(wn_s2_dic,n_occ,n_act,n_act)
    if(three_body):	
        three_body_as += three_body_dic2ten(wn_s2_dic,n_occ,n_act)

    # [[Fn,S2,S2]]
    fn_s2_s2_dic = fn_s2_s2(fdic,t2dic)
    constant += 0.5 * fn_s2_s2_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(fn_s2_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s2_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += 0.5 * three_body_dic2ten(fn_s2_s2_dic,n_occ,n_act)

    # return hamiltonian
    if(three_body):
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act, w=three_body_as)
    else:
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act)

def eff_ham_a3(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=False):
    # For sizing
    n_occ = n_a + n_b
    n_orb = int(fmat.shape[0]/2)

    # Initialize return values
    constant = 0.0
    one_body_as = cp.deepcopy(fmat[0:2*n_act,0:2*n_act])
    two_body_as = cp.deepcopy(vten[0:2*n_act,0:2*n_act,0:2*n_act,0:2*n_act])
    if(three_body):
        three_body_as = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))

    # Initialize dictionaries
    fdic = one_body_mat2dic(fmat,n_occ,n_act,n_orb)
    vdic = two_body_ten2dic(vten,n_occ,n_act,n_orb)
    t1dic = t1_mat2dic(t1_to_ext(t1_amps,n_act),n_act)
    t2dic = t2_ten2dic(t2_to_ext(t2_amps,n_act),n_act)

    # Begin computing DUCC terms
    # [Fn,S1]
    fn_s1_dic = fn_s1(fdic,t1dic)
    constant += fn_s1_dic["c"]
    one_body_as += one_body_dic2mat(fn_s1_dic,n_occ,n_act,n_act)

    # [Fn,S2]
    fn_s2_dic = fn_s2(fdic,t2dic)
    one_body_as += one_body_dic2mat(fn_s2_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(fn_s2_dic,n_occ,n_act,n_act)

    # [Wn,S1]
    wn_s1_dic = wn_s1(vdic,t1dic)
    one_body_as += one_body_dic2mat(wn_s1_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(wn_s1_dic,n_occ,n_act,n_act)

    # [Wn,S2]
    wn_s2_dic = wn_s2(vdic,t2dic,inc_3_body=three_body)
    constant += wn_s2_dic["c"]
    one_body_as += one_body_dic2mat(wn_s2_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(wn_s2_dic,n_occ,n_act,n_act)
    if(three_body):	
        three_body_as += three_body_dic2ten(wn_s2_dic,n_occ,n_act)

    # return hamiltonian
    if(three_body):
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act, w=three_body_as)
    else:
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act)

def eff_ham_a4(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=False):
    # For sizing
    n_occ = n_a + n_b
    n_orb = int(fmat.shape[0]/2)

    # Initialize return values
    constant = 0.0
    one_body_as = cp.deepcopy(fmat[0:2*n_act,0:2*n_act])
    two_body_as = cp.deepcopy(vten[0:2*n_act,0:2*n_act,0:2*n_act,0:2*n_act])
    if(three_body):
        three_body_as = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))

    # Initialize dictionaries
    fdic = one_body_mat2dic(fmat,n_occ,n_act,n_orb)
    vdic = two_body_ten2dic(vten,n_occ,n_act,n_orb)
    t1dic = t1_mat2dic(t1_to_ext(t1_amps,n_act),n_act)
    t2dic = t2_ten2dic(t2_to_ext(t2_amps,n_act),n_act)

    # Begin computing DUCC terms
    # [Fn,S1]
    fn_s1_dic = fn_s1(fdic,t1dic)
    constant += fn_s1_dic["c"]
    one_body_as += one_body_dic2mat(fn_s1_dic,n_occ,n_act,n_act)

    # [Fn,S2]
    fn_s2_dic = fn_s2(fdic,t2dic)
    one_body_as += one_body_dic2mat(fn_s2_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(fn_s2_dic,n_occ,n_act,n_act)

    # [Wn,S1]
    wn_s1_dic = wn_s1(vdic,t1dic)
    one_body_as += one_body_dic2mat(wn_s1_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(wn_s1_dic,n_occ,n_act,n_act)

    # [Wn,S2]
    wn_s2_dic = wn_s2(vdic,t2dic,inc_3_body=three_body)
    constant += wn_s2_dic["c"]
    one_body_as += one_body_dic2mat(wn_s2_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(wn_s2_dic,n_occ,n_act,n_act)
    if(three_body):	
        three_body_as += three_body_dic2ten(wn_s2_dic,n_occ,n_act)

    # [[Fn,S1,S1]]
    fn_s1_s1_dic = fn_s1_s1(fdic,t1dic)
    constant += 0.5 * fn_s1_s1_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(fn_s1_s1_dic,n_occ,n_act,n_act)

    # [[Fn,S1,S2]]
    fn_s1_s2_dic = fn_s1_s2(fdic,t1dic,t2dic)
    one_body_as += 0.5 * one_body_dic2mat(fn_s1_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s1_s2_dic,n_occ,n_act,n_act)

    # [[Fn,S2,S1]]
    fn_s2_s1_dic = fn_s2_s1(fdic,t1dic,t2dic)
    constant += 0.5 * fn_s2_s1_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(fn_s2_s1_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s2_s1_dic,n_occ,n_act,n_act)

    # [[Fn,S2,S2]]
    fn_s2_s2_dic = fn_s2_s2(fdic,t2dic)
    constant += 0.5 * fn_s2_s2_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(fn_s2_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s2_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += 0.5 * three_body_dic2ten(fn_s2_s2_dic,n_occ,n_act)

    # return hamiltonian
    if(three_body):
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act, w=three_body_as)
    else:
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act)

def eff_ham_a5(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=False,four_body=False):
    # For sizing
    n_occ = n_a + n_b
    n_orb = int(fmat.shape[0]/2)

    # Initialize return values
    constant = 0.0
    one_body_as = cp.deepcopy(fmat[0:2*n_act,0:2*n_act])
    two_body_as = cp.deepcopy(vten[0:2*n_act,0:2*n_act,0:2*n_act,0:2*n_act])
    if(three_body):
        three_body_as = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))
    if(four_body):
        four_body_as = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))

    # Initialize dictionaries
    fdic = one_body_mat2dic(fmat,n_occ,n_act,n_orb)
    vdic = two_body_ten2dic(vten,n_occ,n_act,n_orb)
    t1dic = t1_mat2dic(t1_to_ext(t1_amps,n_act),n_act)
    t2dic = t2_ten2dic(t2_to_ext(t2_amps,n_act),n_act)

    # Begin computing DUCC terms
    # [Fn,S1]
    fn_s1_dic = fn_s1(fdic,t1dic)
    constant += fn_s1_dic["c"]
    one_body_as += one_body_dic2mat(fn_s1_dic,n_occ,n_act,n_act)

    # [Fn,S2]
    fn_s2_dic = fn_s2(fdic,t2dic)
    one_body_as += one_body_dic2mat(fn_s2_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(fn_s2_dic,n_occ,n_act,n_act)

    # [Wn,S1]
    wn_s1_dic = wn_s1(vdic,t1dic)
    one_body_as += one_body_dic2mat(wn_s1_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(wn_s1_dic,n_occ,n_act,n_act)

    # [Wn,S2]
    wn_s2_dic = wn_s2(vdic,t2dic,inc_3_body=three_body)
    constant += wn_s2_dic["c"]
    one_body_as += one_body_dic2mat(wn_s2_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(wn_s2_dic,n_occ,n_act,n_act)
    if(three_body):	
        three_body_as += three_body_dic2ten(wn_s2_dic,n_occ,n_act)

    # [[Fn,S1],S2]
    fn_s1_s2_dic = fn_s1_s2(fdic,t1dic,t2dic)
    one_body_as += 0.5 * one_body_dic2mat(fn_s1_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s1_s2_dic,n_occ,n_act,n_act)

    # [[Fn,S2],S1]
    fn_s2_s1_dic = fn_s2_s1(fdic,t1dic,t2dic)
    constant += 0.5 * fn_s2_s1_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(fn_s2_s1_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s2_s1_dic,n_occ,n_act,n_act)

    # [[Fn,S2],S2]
    fn_s2_s2_dic = fn_s2_s2(fdic,t2dic)
    constant += 0.5 * fn_s2_s2_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(fn_s2_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s2_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += 0.5 * three_body_dic2ten(fn_s2_s2_dic,n_occ,n_act)

    # [[Wn,S2],S2]
    wn_s2_s2_dic = wn_s2_s2(vdic,t2dic)
    constant += 0.5 * wn_s2_s2_dic['c']
    one_body_as += 0.5 * one_body_dic2mat(wn_s2_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(wn_s2_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += 0.5 * three_body_dic2ten(wn_s2_s2_dic,n_occ,n_act)
    if(four_body):
        four_body_as += 0.5 * four_body_dic2ten(wn_s2_s2_dic,n_occ,n_act)

    # [[[Fn,S2],S2],S2]
    fn_s2_s2_s2_dic = fn_s2_s2_s2(fdic,t2dic)
    one_body_as += (1./6.) * one_body_dic2mat(fn_s2_s2_s2_dic,n_occ,n_act,n_act)
    two_body_as += (1./6.) * two_body_dic2ten(fn_s2_s2_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += (1./6.) * three_body_dic2ten(fn_s2_s2_s2_dic,n_occ,n_act)
    if(four_body):
        four_body_as += (1./6.) * four_body_dic2ten(fn_s2_s2_s2_dic,n_occ,n_act)

    # return hamiltonian
    if(three_body and four_body):
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act, w=three_body_as, x=four_body_as)
    elif(three_body):
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act, w=three_body_as)
    else:
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act)

def eff_ham_a6(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=False,four_body=False):
    # For sizing
    n_occ = n_a + n_b
    n_orb = int(fmat.shape[0]/2)

    # Initialize return values
    constant = 0.0
    one_body_as = cp.deepcopy(fmat[0:2*n_act,0:2*n_act])
    two_body_as = cp.deepcopy(vten[0:2*n_act,0:2*n_act,0:2*n_act,0:2*n_act])
    if(three_body):
        three_body_as = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))
    if(four_body):
        four_body_as = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))

    # Initialize dictionaries
    fdic = one_body_mat2dic(fmat,n_occ,n_act,n_orb)
    vdic = two_body_ten2dic(vten,n_occ,n_act,n_orb)
    t1dic = t1_mat2dic(t1_to_ext(t1_amps,n_act),n_act)
    t2dic = t2_ten2dic(t2_to_ext(t2_amps,n_act),n_act)

    # Begin computing DUCC terms
    # [Fn,S1]
    fn_s1_dic = fn_s1(fdic,t1dic)
    constant += fn_s1_dic["c"]
    one_body_as += one_body_dic2mat(fn_s1_dic,n_occ,n_act,n_act)

    # [Fn,S2]
    fn_s2_dic = fn_s2(fdic,t2dic)
    one_body_as += one_body_dic2mat(fn_s2_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(fn_s2_dic,n_occ,n_act,n_act)

    # [Wn,S1]
    wn_s1_dic = wn_s1(vdic,t1dic)
    one_body_as += one_body_dic2mat(wn_s1_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(wn_s1_dic,n_occ,n_act,n_act)

    # [Wn,S2]
    wn_s2_dic = wn_s2(vdic,t2dic,inc_3_body=three_body)
    constant += wn_s2_dic["c"]
    one_body_as += one_body_dic2mat(wn_s2_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(wn_s2_dic,n_occ,n_act,n_act)
    if(three_body):	
        three_body_as += three_body_dic2ten(wn_s2_dic,n_occ,n_act)

    # [[Fn,S1],S1]
    fn_s1_s1_dic = fn_s1_s1(fdic,t1dic)
    constant += 0.5 * fn_s1_s1_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(fn_s1_s1_dic,n_occ,n_act,n_act)

    # [[Fn,S1],S2]
    fn_s1_s2_dic = fn_s1_s2(fdic,t1dic,t2dic)
    one_body_as += 0.5 * one_body_dic2mat(fn_s1_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s1_s2_dic,n_occ,n_act,n_act)

    # [[Fn,S2],S1]
    fn_s2_s1_dic = fn_s2_s1(fdic,t1dic,t2dic)
    constant += 0.5 * fn_s2_s1_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(fn_s2_s1_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s2_s1_dic,n_occ,n_act,n_act)

    # [[Fn,S2],S2]
    fn_s2_s2_dic = fn_s2_s2(fdic,t2dic)
    constant += 0.5 * fn_s2_s2_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(fn_s2_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s2_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += 0.5 * three_body_dic2ten(fn_s2_s2_dic,n_occ,n_act)

    # [[Wn,S1],S1]
    wn_s1_s1_dic = wn_s1_s1(vdic,t1dic)
    constant += 0.5 * wn_s1_s1_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(wn_s1_s1_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(wn_s1_s1_dic,n_occ,n_act,n_act)

    # [[Wn,S1],S2]
    wn_s1_s2_dic = wn_s1_s2(vdic,t1dic,t2dic)
    constant += 0.5 * wn_s1_s2_dic['c']
    one_body_as += 0.5 * one_body_dic2mat(wn_s1_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(wn_s1_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += 0.5 * three_body_dic2ten(wn_s1_s2_dic,n_occ,n_act)

    # [[Wn,S2],S1]
    wn_s2_s1_dic = wn_s2_s1(vdic,t1dic,t2dic)
    constant += 0.5 * wn_s2_s1_dic['c']
    one_body_as += 0.5 * one_body_dic2mat(wn_s2_s1_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(wn_s2_s1_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += 0.5 * three_body_dic2ten(wn_s2_s1_dic,n_occ,n_act)

    # [[Wn,S2],S2]
    wn_s2_s2_dic = wn_s2_s2(vdic,t2dic)
    constant += 0.5 * wn_s2_s2_dic['c']
    one_body_as += 0.5 * one_body_dic2mat(wn_s2_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(wn_s2_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += 0.5 * three_body_dic2ten(wn_s2_s2_dic,n_occ,n_act)
    if(four_body):
        four_body_as += 0.5 * four_body_dic2ten(wn_s2_s2_dic,n_occ,n_act)

    # return hamiltonian
    if(three_body and four_body):
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act, w=three_body_as, x=four_body_as)
    elif(three_body):
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act, w=three_body_as)
    else:
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act)

def eff_ham_a7(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=False,four_body=False):
    # For sizing
    n_occ = n_a + n_b
    n_orb = int(fmat.shape[0]/2)

    # Initialize return values
    constant = 0.0
    one_body_as = cp.deepcopy(fmat[0:2*n_act,0:2*n_act])
    two_body_as = cp.deepcopy(vten[0:2*n_act,0:2*n_act,0:2*n_act,0:2*n_act])
    if(three_body):
        three_body_as = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))
    if(four_body):
        four_body_as = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))

    # Initialize dictionaries
    fdic = one_body_mat2dic(fmat,n_occ,n_act,n_orb)
    vdic = two_body_ten2dic(vten,n_occ,n_act,n_orb)
    t1dic = t1_mat2dic(t1_to_ext(t1_amps,n_act),n_act)
    t2dic = t2_ten2dic(t2_to_ext(t2_amps,n_act),n_act)

    # Begin computing DUCC terms
    # [Fn,S1]
    fn_s1_dic = fn_s1(fdic,t1dic)
    constant += fn_s1_dic["c"]
    one_body_as += one_body_dic2mat(fn_s1_dic,n_occ,n_act,n_act)

    # [Fn,S2]
    fn_s2_dic = fn_s2(fdic,t2dic)
    one_body_as += one_body_dic2mat(fn_s2_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(fn_s2_dic,n_occ,n_act,n_act)

    # [Wn,S1]
    wn_s1_dic = wn_s1(vdic,t1dic)
    one_body_as += one_body_dic2mat(wn_s1_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(wn_s1_dic,n_occ,n_act,n_act)

    # [Wn,S2]
    wn_s2_dic = wn_s2(vdic,t2dic,inc_3_body=three_body)
    constant += wn_s2_dic["c"]
    one_body_as += one_body_dic2mat(wn_s2_dic,n_occ,n_act,n_act)
    two_body_as += two_body_dic2ten(wn_s2_dic,n_occ,n_act,n_act)
    if(three_body):	
        three_body_as += three_body_dic2ten(wn_s2_dic,n_occ,n_act)

    # [[Fn,S1],S1]
    fn_s1_s1_dic = fn_s1_s1(fdic,t1dic)
    constant += 0.5 * fn_s1_s1_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(fn_s1_s1_dic,n_occ,n_act,n_act)

    # [[Fn,S1],S2]
    fn_s1_s2_dic = fn_s1_s2(fdic,t1dic,t2dic)
    one_body_as += 0.5 * one_body_dic2mat(fn_s1_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s1_s2_dic,n_occ,n_act,n_act)

    # [[Fn,S2],S1]
    fn_s2_s1_dic = fn_s2_s1(fdic,t1dic,t2dic)
    constant += 0.5 * fn_s2_s1_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(fn_s2_s1_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s2_s1_dic,n_occ,n_act,n_act)

    # [[Fn,S2],S2]
    fn_s2_s2_dic = fn_s2_s2(fdic,t2dic,inc_3_body=three_body)
    constant += 0.5 * fn_s2_s2_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(fn_s2_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(fn_s2_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += 0.5 * three_body_dic2ten(fn_s2_s2_dic,n_occ,n_act)

    # [[Wn,S1],S1]
    wn_s1_s1_dic = wn_s1_s1(vdic,t1dic)
    constant += 0.5 * wn_s1_s1_dic["c"]
    one_body_as += 0.5 * one_body_dic2mat(wn_s1_s1_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(wn_s1_s1_dic,n_occ,n_act,n_act)

    # [[Wn,S1],S2]
    wn_s1_s2_dic = wn_s1_s2(vdic,t1dic,t2dic,inc_3_body=three_body)
    constant += 0.5 * wn_s1_s2_dic['c']
    one_body_as += 0.5 * one_body_dic2mat(wn_s1_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(wn_s1_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += 0.5 * three_body_dic2ten(wn_s1_s2_dic,n_occ,n_act)

    # [[Wn,S2],S1]
    wn_s2_s1_dic = wn_s2_s1(vdic,t1dic,t2dic,inc_3_body=three_body)
    constant += 0.5 * wn_s2_s1_dic['c']
    one_body_as += 0.5 * one_body_dic2mat(wn_s2_s1_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(wn_s2_s1_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += 0.5 * three_body_dic2ten(wn_s2_s1_dic,n_occ,n_act)

    # [[Wn,S2],S2]
    wn_s2_s2_dic = wn_s2_s2(vdic,t2dic,inc_3_body=three_body,inc_4_body=four_body)
    constant += 0.5 * wn_s2_s2_dic['c']
    one_body_as += 0.5 * one_body_dic2mat(wn_s2_s2_dic,n_occ,n_act,n_act)
    two_body_as += 0.5 * two_body_dic2ten(wn_s2_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += 0.5 * three_body_dic2ten(wn_s2_s2_dic,n_occ,n_act)
    if(four_body):
        four_body_as += 0.5 * four_body_dic2ten(wn_s2_s2_dic,n_occ,n_act)

    # [[[Fn,S1],S1],S1]
    fn_s1_s1_s1_dic = fn_s1_s1_s1(fdic,t1dic)
    constant += (1./6.) * fn_s1_s1_s1_dic['c']
    one_body_as += (1./6.) * one_body_dic2mat(fn_s1_s1_s1_dic,n_occ,n_act,n_act)

    # [[[Fn,S1],S1],S2]
    fn_s1_s1_s2_dic = fn_s1_s1_s2(fdic,t1dic,t2dic)
    one_body_as += (1./6.) * one_body_dic2mat(fn_s1_s1_s2_dic,n_occ,n_act,n_act)
    two_body_as += (1./6.) * two_body_dic2ten(fn_s1_s1_s2_dic,n_occ,n_act,n_act)

    # [[[Fn,S1],S2],S1]
    fn_s1_s2_s1_dic = fn_s1_s2_s1(fdic,t1dic,t2dic)
    constant += (1./6.) * fn_s1_s2_s1_dic['c']
    one_body_as += (1./6.) * one_body_dic2mat(fn_s1_s2_s1_dic,n_occ,n_act,n_act)
    two_body_as += (1./6.) * two_body_dic2ten(fn_s1_s2_s1_dic,n_occ,n_act,n_act)

    # [[[Fn,S1],S2],S2]
    fn_s1_s2_s2_dic = fn_s1_s2_s2(fdic,t1dic,t2dic,inc_3_body=three_body)
    constant += (1./6.) * fn_s1_s2_s2_dic["c"]
    one_body_as += (1./6.) * one_body_dic2mat(fn_s1_s2_s2_dic,n_occ,n_act,n_act)
    two_body_as += (1./6.) * two_body_dic2ten(fn_s1_s2_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += (1./6.) * three_body_dic2ten(fn_s1_s2_s2_dic,n_occ,n_act)

    # [[[Fn,S2],S1],S1]
    fn_s2_s1_s1_dic = fn_s2_s1_s1(fdic,t1dic,t2dic)
    constant += (1./6.) * fn_s2_s1_s1_dic["c"]
    one_body_as += (1./6.) * one_body_dic2mat(fn_s2_s1_s1_dic,n_occ,n_act,n_act)
    two_body_as += (1./6.) * two_body_dic2ten(fn_s2_s1_s1_dic,n_occ,n_act,n_act)

    # [[[Fn,S2],S1],S2]
    fn_s2_s1_s2_dic = fn_s2_s1_s2(fdic,t1dic,t2dic,inc_3_body=three_body)
    constant += (1./6.) * fn_s2_s1_s2_dic["c"]
    one_body_as += (1./6.) * one_body_dic2mat(fn_s2_s1_s2_dic,n_occ,n_act,n_act)
    two_body_as += (1./6.) * two_body_dic2ten(fn_s2_s1_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += (1./6.) * three_body_dic2ten(fn_s2_s1_s2_dic,n_occ,n_act)

    # [[[Fn,S2],S2],S1]
    fn_s2_s2_s1_dic = fn_s2_s2_s1(fdic,t1dic,t2dic,inc_3_body=three_body)
    constant += (1./6.) * fn_s2_s2_s1_dic["c"]
    one_body_as += (1./6.) * one_body_dic2mat(fn_s2_s2_s1_dic,n_occ,n_act,n_act)
    two_body_as += (1./6.) * two_body_dic2ten(fn_s2_s2_s1_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += (1./6.) * three_body_dic2ten(fn_s2_s2_s1_dic,n_occ,n_act)

    # [[[Fn,S2],S2],S2]
    fn_s2_s2_s2_dic = fn_s2_s2_s2(fdic,t2dic,inc_3_body=three_body,inc_4_body=four_body)
    one_body_as += (1./6.) * one_body_dic2mat(fn_s2_s2_s2_dic,n_occ,n_act,n_act)
    two_body_as += (1./6.) * two_body_dic2ten(fn_s2_s2_s2_dic,n_occ,n_act,n_act)
    if(three_body):
        three_body_as += (1./6.) * three_body_dic2ten(fn_s2_s2_s2_dic,n_occ,n_act)
    if(four_body):
        four_body_as += (1./6.) * four_body_dic2ten(fn_s2_s2_s2_dic,n_occ,n_act)

    # return hamiltonian
    if(three_body and four_body):
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act, w=three_body_as, x=four_body_as)
    elif(three_body):
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act, w=three_body_as)
    else:
        return Hamiltonian(one_body_as, two_body_as, n_a, n_b, n_orb, constant, n_act=n_act)


def calc_ducc(system, H, n_act: int, approximation: str="a7", *, three_body: bool = False, four_body: bool  = False,):
    mccsd = cc.UCCSD(system.meanfield)
    # mccsd.conv_tol = 1e-12
    # mccsd.conv_tol_normt = 1e-10
    mccsd.max_cycle = 1000
    mccsd.verbose = 0
    mccsd.kernel()

    t1_amps = np.array(mccsd.t1)
    t2_amps = np.array(mccsd.t2)

    fmat, vten = H._f, H._v 
    ccsd_energy = calc_ccsd(fmat, vten, t1_amps, t2_amps, verbose=0)
    assert np.isclose(ccsd_energy+system.meanfield.e_tot, mccsd.e_tot, rtol=0.0, atol=1e-8)
    ccsd_summary(mccsd.e_tot, ccsd_energy)

    print("\n   DUCC Calculation Summary")
    print("   -------------------------------------")
    print("Size of the active space                       :%10i" %(n_act))
    n_a = system.n_a
    n_b = system.n_b 
    t0 = time.perf_counter()
    key = approximation.strip().lower()
    if key == "a1":
        ham = eff_ham_a1(fmat,vten,n_a,n_b,n_act)    
    elif key == "a2":
        ham = eff_ham_a2(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=three_body)
    elif key == "a3":
        ham = eff_ham_a3(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=three_body)
    elif key == "a4":
        ham = eff_ham_a4(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=three_body)
    elif key == "a5":
        ham = eff_ham_a5(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=three_body,four_body=four_body)
    elif key == "a6":
        ham = eff_ham_a6(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=three_body,four_body=four_body)
    elif key == "a7":
        ham = eff_ham_a7(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=three_body,four_body=four_body)        
    else:
        raise ValueError(f"Unsupported DUCC method {key!r}; choose between 'A1' to 'A7'.")

    print(f"DUCC {key.upper()} hamiltonian constructed!")
    dt = time.perf_counter() - t0
    m, s = divmod(dt, 60)
    print("DUCC wall time                                 :%8.2f m  %3.2f s" % (m, s))
    
    return ham 


