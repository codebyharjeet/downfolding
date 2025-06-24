import scipy
import numpy as np 
import copy as cp 
import pyscf
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, cc, lib
from pyscf.cc import ccsd
from downfolding.hamiltonian import HamFormat, Hamiltonian
from downfolding.ccsd import calc_ccsd
from downfolding.helper import asym_term, one_body_mat2dic, one_body_dic2mat, two_body_ten2dic, two_body_dic2ten, three_body_ten2dic, three_body_dic2ten, four_body_ten2dic, four_body_dic2ten, t1_mat2dic, t2_ten2dic, get_many_body_terms, as_proj, t1_to_op, t2_to_op, t1_to_ext, t2_to_ext


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
	fs1["c"]  += 1.000 * np.einsum("Ai,iA->",f["Vo"],t1["oV"],optimize="optimal") # o*V
	fs1["c"]  += 1.000 * np.einsum("iA,Ai->",f["oV"],t1["Vo"],optimize="optimal") # o*V

	fs1["oo"] += 1.000 * np.einsum("Ai,jA->ji",f["Vo"],t1["oV"],optimize="optimal") # o*o*V
	fs1["oo"] += 1.000 * np.einsum("iA,Aj->ij",f["oV"],t1["Vo"],optimize="optimal") # o*o*V

	fs1["ov"] += 1.000 * np.einsum("Aa,iA->ia",f["Vv"],t1["oV"],optimize="optimal") # o*v*V

	fs1["vo"] += 1.000 * np.einsum("aA,Ai->ai",f["vV"],t1["Vo"],optimize="optimal") # o*v*V

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
	fs2["ov"] +=  1.000 * np.einsum("Aj,ijaA->ia",f["Vo"],t2["oovV"],optimize="optimal") # o*o*v*V

	fs2["vo"] +=  1.000 * np.einsum("jA,aAij->ai",f["oV"],t2["vVoo"],optimize="optimal") # o*o*v*V

	# ooov += np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2["ooov"] += -0.250 * np.einsum("Ai,jkaA->jkia",f["Vo"],t2["oovV"],optimize="optimal")
	fs2["ooov"] = asym_term(fs2["ooov"],"ooov")

	# ovoo += np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2["ovoo"] += -0.250 * np.einsum("iA,aAjk->iajk",f["oV"],t2["vVoo"],optimize="optimal")
	fs2["ovoo"] = asym_term(fs2["ovoo"],"ovoo")

	# oovv += np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	fs2["oovv"] += -0.500 * np.einsum("Aa,ijbA->ijab",f["Vv"],t2["oovV"],optimize="optimal")
	fs2["oovv"] = asym_term(fs2["oovv"],"oovv")

	# vvoo += np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	fs2["vvoo"] += -0.500 * np.einsum("aA,bAij->abij",f["vV"],t2["vVoo"],optimize="optimal")
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
	ws1["oo"] += 4.000 * np.einsum("Ak,jkiA->ji",t1["Vo"],v["oooV"],optimize="optimal")
	ws1["oo"] += 4.000 * np.einsum("kA,jAik->ji",t1["oV"],v["oVoo"],optimize="optimal")

	ws1["ov"] +=  4.000 * np.einsum("Aj,ijaA->ia",t1["Vo"],v["oovV"],optimize="optimal")
	ws1["ov"] += -4.000 * np.einsum("jA,iAja->ia",t1["oV"],v["oVov"],optimize="optimal")

	ws1["vo"] += -4.000 * np.einsum("Aj,jaiA->ai",t1["Vo"],v["ovoV"],optimize="optimal")
	ws1["vo"] +=  4.000 * np.einsum("jA,aAij->ai",t1["oV"],v["vVoo"],optimize="optimal")

	ws1["vv"] += -4.000 * np.einsum("Ai,ibaA->ba",t1["Vo"],v["ovvV"],optimize="optimal")
	ws1["vv"] += -4.000 * np.einsum("iA,bAia->ba",t1["oV"],v["vVov"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	ws1["oooo"] += -2.000 * np.einsum("Ai,kljA->klij",t1["Vo"],v["oooV"],optimize="optimal")
	ws1["oooo"] += -2.000 * np.einsum("iA,lAjk->iljk",t1["oV"],v["oVoo"],optimize="optimal")
	ws1["oooo"] = asym_term(ws1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	ws1["ooov"] += -2.000 * np.einsum("iA,kAja->ikja",t1["oV"],v["oVov"],optimize="optimal")
	ws1["ooov"] += -1.000 * np.einsum("Ai,jkaA->jkia",t1["Vo"],v["oovV"],optimize="optimal")
	ws1["ooov"] = asym_term(ws1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	ws1["ovoo"] += -2.000 * np.einsum("Ai,kajA->kaij",t1["Vo"],v["ovoV"],optimize="optimal")
	ws1["ovoo"] += -1.000 * np.einsum("iA,aAjk->iajk",t1["oV"],v["vVoo"],optimize="optimal")
	ws1["ovoo"] = asym_term(ws1["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	ws1["oovv"] += -2.000 * np.einsum("iA,jAab->ijab",t1["oV"],v["oVvv"],optimize="optimal")
	ws1["oovv"] = asym_term(ws1["oovv"],"oovv")

	ws1["ovov"] += -1.000 * np.einsum("Ai,jbaA->jbia",t1["Vo"],v["ovvV"],optimize="optimal")
	ws1["ovov"] += -1.000 * np.einsum("iA,bAja->ibja",t1["oV"],v["vVov"],optimize="optimal")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	ws1["vvoo"] += -2.000 * np.einsum("Ai,abjA->abij",t1["Vo"],v["vvoV"],optimize="optimal")
	ws1["vvoo"] = asym_term(ws1["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	ws1["ovvv"] += -1.000 * np.einsum("iA,cAab->icab",t1["oV"],v["vVvv"],optimize="optimal")
	ws1["ovvv"] = asym_term(ws1["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	ws1["vvov"] += -1.000 * np.einsum("Ai,bcaA->bcia",t1["Vo"],v["vvvV"],optimize="optimal")
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
	ws2["c"] += 2.000 * np.einsum("aAij,ijaA->",t2["vVoo"],v["oovV"],optimize="optimal")
	ws2["c"] += 1.000 * np.einsum("ABij,ijAB->",t2["VVoo"],v["ooVV"],optimize="optimal")
	ws2["c"] += 2.000 * np.einsum("ijaA,aAij->",t2["oovV"],v["vVoo"],optimize="optimal")
	ws2["c"] += 1.000 * np.einsum("ijAB,ABij->",t2["ooVV"],v["VVoo"],optimize="optimal")

	ws2["oo"] += 4.000 * np.einsum("aAik,jkaA->ji",t2["vVoo"],v["oovV"],optimize="optimal")
	ws2["oo"] += 2.000 * np.einsum("ABik,jkAB->ji",t2["VVoo"],v["ooVV"],optimize="optimal")
	ws2["oo"] += 4.000 * np.einsum("ikaA,aAjk->ij",t2["oovV"],v["vVoo"],optimize="optimal")
	ws2["oo"] += 2.000 * np.einsum("ikAB,ABjk->ij",t2["ooVV"],v["VVoo"],optimize="optimal")

	ws2["ov"] += -2.000 * np.einsum("jkaA,iAjk->ia",t2["oovV"],v["oVoo"],optimize="optimal")
	ws2["ov"] += -4.000 * np.einsum("ijbA,bAja->ia",t2["oovV"],v["vVov"],optimize="optimal")
	ws2["ov"] += -2.000 * np.einsum("ijAB,ABja->ia",t2["ooVV"],v["VVov"],optimize="optimal")

	ws2["vo"] += -4.000 * np.einsum("bAij,jabA->ai",t2["vVoo"],v["ovvV"],optimize="optimal")
	ws2["vo"] += -2.000 * np.einsum("ABij,jaAB->ai",t2["VVoo"],v["ovVV"],optimize="optimal")
	ws2["vo"] += -2.000 * np.einsum("aAjk,jkiA->ai",t2["vVoo"],v["oooV"],optimize="optimal")

	ws2["vv"] += -2.000 * np.einsum("aAij,ijbA->ab",t2["vVoo"],v["oovV"],optimize="optimal")
	ws2["vv"] += -2.000 * np.einsum("ijaA,bAij->ba",t2["oovV"],v["vVoo"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	ws2["oooo"] +=  1.000 * np.einsum("aAij,klaA->klij",t2["vVoo"],v["oovV"],optimize="optimal")
	ws2["oooo"] +=  1.000 * np.einsum("ijaA,aAkl->ijkl",t2["oovV"],v["vVoo"],optimize="optimal")
	ws2["oooo"] +=  0.500 * np.einsum("ABij,klAB->klij",t2["VVoo"],v["ooVV"],optimize="optimal")
	ws2["oooo"] +=  0.500 * np.einsum("ijAB,ABkl->ijkl",t2["ooVV"],v["VVoo"],optimize="optimal")
	ws2["oooo"] = asym_term(ws2["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	ws2["ooov"] += -2.000 * np.einsum("ilaA,kAjl->ikja",t2["oovV"],v["oVoo"],optimize="optimal")
	ws2["ooov"] +=  1.000 * np.einsum("ijbA,bAka->ijka",t2["oovV"],v["vVov"],optimize="optimal")
	ws2["ooov"] +=  0.500 * np.einsum("ijAB,ABka->ijka",t2["ooVV"],v["VVov"],optimize="optimal")
	ws2["ooov"] = asym_term(ws2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	ws2["ovoo"] += -2.000 * np.einsum("aAil,kljA->kaij",t2["vVoo"],v["oooV"],optimize="optimal")
	ws2["ovoo"] +=  1.000 * np.einsum("bAij,kabA->kaij",t2["vVoo"],v["ovvV"],optimize="optimal")
	ws2["ovoo"] +=  0.500 * np.einsum("ABij,kaAB->kaij",t2["VVoo"],v["ovVV"],optimize="optimal")
	ws2["ovoo"] = asym_term(ws2["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	ws2["oovv"] += -4.000 * np.einsum("ikaA,jAkb->ijab",t2["oovV"],v["oVov"],optimize="optimal")
	ws2["oovv"] +=  1.000 * np.einsum("ijcA,cAab->ijab",t2["oovV"],v["vVvv"],optimize="optimal")
	ws2["oovv"] +=  0.500 * np.einsum("ijAB,ABab->ijab",t2["ooVV"],v["VVvv"],optimize="optimal")
	ws2["oovv"] = asym_term(ws2["oovv"],"oovv")

	ws2["ovov"] += -1.000 * np.einsum("aAik,jkbA->jaib",t2["vVoo"],v["oovV"],optimize="optimal")
	ws2["ovov"] += -1.000 * np.einsum("ikaA,bAjk->ibja",t2["oovV"],v["vVoo"],optimize="optimal")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	ws2["vvoo"] += -4.000 * np.einsum("aAik,kbjA->abij",t2["vVoo"],v["ovoV"],optimize="optimal")
	ws2["vvoo"] +=  1.000 * np.einsum("cAij,abcA->abij",t2["vVoo"],v["vvvV"],optimize="optimal")
	ws2["vvoo"] +=  0.500 * np.einsum("ABij,abAB->abij",t2["VVoo"],v["vvVV"],optimize="optimal")
	ws2["vvoo"] = asym_term(ws2["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	ws2["ovvv"] += -2.000 * np.einsum("ijaA,cAjb->icab",t2["oovV"],v["vVov"],optimize="optimal")
	ws2["ovvv"] = asym_term(ws2["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	ws2["vvov"] += -2.000 * np.einsum("aAij,jcbA->acib",t2["vVoo"],v["ovvV"],optimize="optimal")
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
		ws2["ooooov"] += -(1./3.) * np.einsum("ijaA,mAkl->ijmkla",t2["oovV"],v["oVoo"],optimize="optimal")
		ws2["ooooov"] = asym_term(ws2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		ws2["oovooo"] += -(1./3.) * np.einsum("aAij,lmkA->lmaijk",t2["vVoo"],v["oooV"],optimize="optimal")
		ws2["oovooo"] = asym_term(ws2["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2["oooovv"] +=  (2./3.) * np.einsum("ijaA,lAkb->ijlkab",t2["oovV"],v["oVov"],optimize="optimal")
		ws2["oooovv"] = asym_term(ws2["oooovv"],"oooovv")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2["oovoov"] += -(1./9.) * np.einsum("aAij,klbA->klaijb",t2["vVoo"],v["oovV"],optimize="optimal")
		ws2["oovoov"] += -(1./9.) * np.einsum("ijaA,bAkl->ijbkla",t2["oovV"],v["vVoo"],optimize="optimal")
		ws2["oovoov"] = asym_term(ws2["oovoov"],"oovoov")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2["ovvooo"] +=  (2./3.) * np.einsum("aAij,lbkA->labijk",t2["vVoo"],v["ovoV"],optimize="optimal")
		ws2["ovvooo"] = asym_term(ws2["ovvooo"],"ovvooo")

		# ooovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2["ooovvv"] += -1.000 * np.einsum("ijaA,kAbc->ijkabc",t2["oovV"],v["oVvv"],optimize="optimal")
		ws2["ooovvv"] = asym_term(ws2["ooovvv"],"ooovvv")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2["oovovv"] +=  (2./9.) * np.einsum("ijaA,cAkb->ijckab",t2["oovV"],v["vVov"],optimize="optimal")
		ws2["oovovv"] = asym_term(ws2["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2["ovvoov"] +=  (2./9.) * np.einsum("aAij,kcbA->kacijb",t2["vVoo"],v["ovvV"],optimize="optimal")
		ws2["ovvoov"] = asym_term(ws2["ovvoov"],"ovvoov")

		# vvvooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2["vvvooo"] += -1.000 * np.einsum("aAij,bckA->abcijk",t2["vVoo"],v["vvoV"],optimize="optimal")
		ws2["vvvooo"] = asym_term(ws2["vvvooo"],"vvvooo")

		# oovvvv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws2["oovvvv"] += -(1./3.) * np.einsum("ijaA,dAbc->ijdabc",t2["oovV"],v["vVvv"],optimize="optimal")
		ws2["oovvvv"] = asym_term(ws2["oovvvv"],"oovvvv")

		# vvvoov = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2["vvvoov"] += -(1./3.) * np.einsum("aAij,cdbA->acdijb",t2["vVoo"],v["vvvV"],optimize="optimal")
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
	fs1s1["c"] += -2.000 * np.einsum("ji,iA,Aj->",f["oo"],t1["oV"],t1["Vo"],optimize="optimal") # o * o * Vext
	fs1s1["c"] +=  2.000 * np.einsum("BA,iB,Ai->",f["VV"],t1["oV"],t1["Vo"],optimize="optimal") # o * Vext * Vext

	fs1s1["oo"] +=  2.000 * np.einsum("BA,iB,Aj->ij",f["VV"],t1["oV"],t1["Vo"],optimize="optimal")
	fs1s1["oo"] += -1.000 * np.einsum("ki,jA,Ak->ji",f["oo"],t1["oV"],t1["Vo"],optimize="optimal")
	fs1s1["oo"] += -1.000 * np.einsum("ik,kA,Aj->ij",f["oo"],t1["oV"],t1["Vo"],optimize="optimal")

	fs1s1["ov"] += -1.000 * np.einsum("ja,iA,Aj->ia",f["ov"],t1["oV"],t1["Vo"],optimize="optimal")

	fs1s1["vo"] += -1.000 * np.einsum("aj,jA,Ai->ai",f["vo"],t1["oV"],t1["Vo"],optimize="optimal")

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
	fs1s2["ov"] += -1.000 * np.einsum("kj,Ak,ijaA->ia",f["oo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2["ov"] +=  1.000 * np.einsum("BA,Aj,ijaB->ia",f["VV"],t1["Vo"],t2["oovV"],optimize="optimal")

	fs1s2["vo"] += -1.000 * np.einsum("kj,jA,aAik->ai",f["oo"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2["vo"] +=  1.000 * np.einsum("BA,jB,aAij->ai",f["VV"],t1["oV"],t2["vVoo"],optimize="optimal")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs1s2["ooov"] +=  0.250 * np.einsum("li,Al,jkaA->jkia",f["oo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2["ooov"] += -0.250 * np.einsum("BA,Ai,jkaB->jkia",f["VV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2["ooov"] = asym_term(fs1s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs1s2["ovoo"] +=  0.250 * np.einsum("il,lA,aAjk->iajk",f["oo"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2["ovoo"] += -0.250 * np.einsum("BA,iB,aAjk->iajk",f["VV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2["ovoo"] = asym_term(fs1s2["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	fs1s2["oovv"] +=  0.500 * np.einsum("ka,Ak,ijbA->ijab",f["ov"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2["oovv"] = asym_term(fs1s2["oovv"],"oovv")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	fs1s2["vvoo"] +=  0.500 * np.einsum("ak,kA,bAij->abij",f["vo"],t1["oV"],t2["vVoo"],optimize="optimal")
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
	fs2s1["c"] += 1.000 * np.einsum("ai,Aj,ijaA->",f["vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1["c"] += 1.000 * np.einsum("Ai,Bj,ijAB->",f["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1["c"] += 1.000 * np.einsum("ia,jA,aAij->",f["ov"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1["c"] += 1.000 * np.einsum("iA,jB,ABij->",f["oV"],t1["oV"],t2["VVoo"],optimize="optimal")

	fs2s1["oo"] +=  1.000 * np.einsum("ai,Ak,jkaA->ji",f["vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1["oo"] +=  1.000 * np.einsum("Ai,Bk,jkAB->ji",f["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1["oo"] += -1.000 * np.einsum("ak,Ai,jkaA->ji",f["vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1["oo"] += -1.000 * np.einsum("Ak,Bi,jkAB->ji",f["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1["oo"] +=  1.000 * np.einsum("ia,kA,aAjk->ij",f["ov"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1["oo"] += -1.000 * np.einsum("ka,iA,aAjk->ij",f["ov"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1["oo"] +=  1.000 * np.einsum("iA,kB,ABjk->ij",f["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs2s1["oo"] += -1.000 * np.einsum("kA,iB,ABjk->ij",f["oV"],t1["oV"],t2["VVoo"],optimize="optimal")

	fs2s1["ov"] += -1.000 * np.einsum("ij,Ak,jkaA->ia",f["oo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1["ov"] += -1.000 * np.einsum("kj,Ak,ijaA->ia",f["oo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1["ov"] +=  1.000 * np.einsum("ba,Aj,ijbA->ia",f["vv"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1["ov"] +=  1.000 * np.einsum("Aa,Bj,ijAB->ia",f["Vv"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1["ov"] +=  1.000 * np.einsum("BA,Aj,ijaB->ia",f["VV"],t1["Vo"],t2["oovV"],optimize="optimal")

	fs2s1["vo"] += -1.000 * np.einsum("ji,kA,aAjk->ai",f["oo"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1["vo"] += -1.000 * np.einsum("kj,jA,aAik->ai",f["oo"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1["vo"] +=  1.000 * np.einsum("ab,jA,bAij->ai",f["vv"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1["vo"] +=  1.000 * np.einsum("aA,jB,ABij->ai",f["vV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs2s1["vo"] +=  1.000 * np.einsum("BA,jB,aAij->ai",f["VV"],t1["oV"],t2["vVoo"],optimize="optimal")

	fs2s1["vv"] += -1.000 * np.einsum("ai,Aj,ijbA->ab",f["vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1["vv"] += -1.000 * np.einsum("ia,jA,bAij->ba",f["ov"],t1["oV"],t2["vVoo"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs2s1["oooo"] +=  0.500 * np.einsum("ai,Aj,klaA->klij",f["vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1["oooo"] +=  0.500 * np.einsum("Ai,Bj,klAB->klij",f["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1["oooo"] +=  0.500 * np.einsum("ia,jA,aAkl->ijkl",f["ov"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1["oooo"] +=  0.500 * np.einsum("iA,jB,ABkl->ijkl",f["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs2s1["oooo"] = asym_term(fs2s1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2s1["ooov"] += -0.500 * np.einsum("il,Aj,klaA->ikja",f["oo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1["ooov"] += -0.250 * np.einsum("ba,Ai,jkbA->jkia",f["vv"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1["ooov"] += -0.250 * np.einsum("Aa,Bi,jkAB->jkia",f["Vv"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1["ooov"] += -0.250 * np.einsum("BA,Ai,jkaB->jkia",f["VV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1["ooov"] = asym_term(fs2s1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2s1["ovoo"] += -0.500 * np.einsum("li,jA,aAkl->jaik",f["oo"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1["ovoo"] += -0.250 * np.einsum("ab,iA,bAjk->iajk",f["vv"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1["ovoo"] += -0.250 * np.einsum("aA,iB,ABjk->iajk",f["vV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs2s1["ovoo"] += -0.250 * np.einsum("BA,iB,aAjk->iajk",f["VV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1["ovoo"] = asym_term(fs2s1["ovoo"],"ovoo")

	fs2s1["ovov"] +=  0.250 * np.einsum("ak,Ai,jkbA->jaib",f["vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1["ovov"] +=  0.250 * np.einsum("ka,iA,bAjk->ibja",f["ov"],t1["oV"],t2["vVoo"],optimize="optimal")

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
	fs2s2["c"] += -2.000 * np.einsum("ji,ikaA,aAjk->",f["oo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["c"] += -1.000 * np.einsum("ji,ikAB,ABjk->",f["oo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2["c"] +=  1.000 * np.einsum("ba,ijbA,aAij->",f["vv"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["c"] +=  1.000 * np.einsum("Aa,ijAB,aBij->",f["Vv"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2["c"] +=  1.000 * np.einsum("aA,ijaB,ABij->",f["vV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2["c"] +=  1.000 * np.einsum("BA,ijaB,aAij->",f["VV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["c"] +=  1.000 * np.einsum("BA,ijBC,ACij->",f["VV"],t2["ooVV"],t2["VVoo"],optimize="optimal") # V^3 o^2

	fs2s2["oo"] +=  2.000 * np.einsum("ba,ikbA,aAjk->ij",f["vv"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["oo"] +=  2.000 * np.einsum("Aa,ikAB,aBjk->ij",f["Vv"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2["oo"] +=  2.000 * np.einsum("aA,ikaB,ABjk->ij",f["vV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2["oo"] +=  2.000 * np.einsum("BA,ikaB,aAjk->ij",f["VV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["oo"] +=  2.000 * np.einsum("BA,ikBC,ACjk->ij",f["VV"],t2["ooVV"],t2["VVoo"],optimize="optimal") # V^3 o^4
	fs2s2["oo"] += -2.000 * np.einsum("lk,ikaA,aAjl->ij",f["oo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["oo"] += -1.000 * np.einsum("ki,jlaA,aAkl->ji",f["oo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["oo"] += -1.000 * np.einsum("ik,klaA,aAjl->ij",f["oo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["oo"] += -1.000 * np.einsum("lk,ikAB,ABjl->ij",f["oo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2["oo"] += -0.500 * np.einsum("ki,jlAB,ABkl->ji",f["oo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2["oo"] += -0.500 * np.einsum("ik,klAB,ABjl->ij",f["oo"],t2["ooVV"],t2["VVoo"],optimize="optimal")

	fs2s2["ov"] +=  1.000 * np.einsum("jb,ikaA,bAjk->ia",f["ov"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ov"] +=  1.000 * np.einsum("jA,ikaB,ABjk->ia",f["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2["ov"] += -1.000 * np.einsum("ja,ikbA,bAjk->ia",f["ov"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ov"] += -0.500 * np.einsum("ja,ikAB,ABjk->ia",f["ov"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2["ov"] += -0.500 * np.einsum("ib,jkaA,bAjk->ia",f["ov"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ov"] += -0.500 * np.einsum("iA,jkaB,ABjk->ia",f["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")

	fs2s2["vo"] +=  1.000 * np.einsum("bj,jkbA,aAik->ai",f["vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["vo"] +=  1.000 * np.einsum("Aj,jkAB,aBik->ai",f["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2["vo"] += -1.000 * np.einsum("aj,jkbA,bAik->ai",f["vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["vo"] += -0.500 * np.einsum("bi,jkbA,aAjk->ai",f["vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["vo"] += -0.500 * np.einsum("Ai,jkAB,aBjk->ai",f["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2["vo"] += -0.500 * np.einsum("aj,jkAB,ABik->ai",f["vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")

	fs2s2["vv"] +=  2.000 * np.einsum("ji,ikaA,bAjk->ba",f["oo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["vv"] += -1.000 * np.einsum("BA,ijaB,bAij->ba",f["VV"],t2["oovV"],t2["vVoo"],optimize="optimal") # v^2 V^2 o^2
	fs2s2["vv"] += -0.500 * np.einsum("ca,ijcA,bAij->ba",f["vv"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["vv"] += -0.500 * np.einsum("Aa,ijAB,bBij->ba",f["Vv"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2["vv"] += -0.500 * np.einsum("ac,ijbA,cAij->ab",f["vv"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["vv"] += -0.500 * np.einsum("aA,ijbB,ABij->ab",f["vV"],t2["oovV"],t2["VVoo"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs2s2["oooo"] +=  0.500 * np.einsum("mi,jkaA,aAlm->jkil",f["oo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["oooo"] +=  0.500 * np.einsum("im,jmaA,aAkl->ijkl",f["oo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["oooo"] +=  0.500 * np.einsum("ba,ijbA,aAkl->ijkl",f["vv"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["oooo"] +=  0.500 * np.einsum("Aa,ijAB,aBkl->ijkl",f["Vv"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2["oooo"] +=  0.500 * np.einsum("aA,ijaB,ABkl->ijkl",f["vV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2["oooo"] +=  0.500 * np.einsum("BA,ijaB,aAkl->ijkl",f["VV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["oooo"] +=  0.500 * np.einsum("BA,ijBC,ACkl->ijkl",f["VV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2["oooo"] +=  0.250 * np.einsum("mi,jkAB,ABlm->jkil",f["oo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2["oooo"] +=  0.250 * np.einsum("im,jmAB,ABkl->ijkl",f["oo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2["oooo"] = asym_term(fs2s2["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2s2["ooov"] +=  0.500 * np.einsum("ib,jlaA,bAkl->ijka",f["ov"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ooov"] +=  0.500 * np.einsum("iA,jlaB,ABkl->ijka",f["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2["ooov"] += -0.250 * np.einsum("la,ijbA,bAkl->ijka",f["ov"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ooov"] +=  0.250 * np.einsum("lb,ijaA,bAkl->ijka",f["ov"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ooov"] +=  0.250 * np.einsum("lA,ijaB,ABkl->ijka",f["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2["ooov"] += -0.125 * np.einsum("la,ijAB,ABkl->ijka",f["ov"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2["ooov"] = asym_term(fs2s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2s2["ovoo"] +=  0.500 * np.einsum("bi,jlbA,aAkl->jaik",f["vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovoo"] +=  0.500 * np.einsum("Ai,jlAB,aBkl->jaik",f["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovoo"] += -0.250 * np.einsum("al,ilbA,bAjk->iajk",f["vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovoo"] +=  0.250 * np.einsum("bl,ilbA,aAjk->iajk",f["vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovoo"] +=  0.250 * np.einsum("Al,ilAB,aBjk->iajk",f["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovoo"] += -0.125 * np.einsum("al,ilAB,ABjk->iajk",f["vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2["ovoo"] = asym_term(fs2s2["ovoo"],"ovoo")

	fs2s2["ovov"] +=  0.500 * np.einsum("lk,ikaA,bAjl->ibja",f["oo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovov"] += -0.500 * np.einsum("BA,ikaB,bAjk->ibja",f["VV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovov"] +=  0.250 * np.einsum("ki,jlaA,bAkl->jbia",f["oo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovov"] +=  0.250 * np.einsum("ik,klaA,bAjl->ibja",f["oo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovov"] += -0.250 * np.einsum("ca,ikcA,bAjk->ibja",f["vv"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovov"] += -0.250 * np.einsum("Aa,ikAB,bBjk->ibja",f["Vv"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovov"] += -0.250 * np.einsum("ac,ikbA,cAjk->iajb",f["vv"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovov"] += -0.250 * np.einsum("aA,ikbB,ABjk->iajb",f["vV"],t2["oovV"],t2["VVoo"],optimize="optimal")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	fs2s2["ovvv"] +=  0.500 * np.einsum("ja,ikbA,cAjk->icab",f["ov"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["ovvv"] = asym_term(fs2s2["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	fs2s2["vvov"] +=  0.500 * np.einsum("aj,jkbA,cAik->acib",f["vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2["vvov"] = asym_term(fs2s2["vvov"],"vvov")

	if(inc_3_body):
		fs2s2["ooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2["oovooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2["oovoov"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2["oovovv"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs2s2["ovvoov"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2["ooooov"] += -(1./12.) * np.einsum("ib,jkaA,bAlm->ijklma",f["ov"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2["ooooov"] += -(1./12.) * np.einsum("iA,jkaB,ABlm->ijklma",f["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2["ooooov"] = asym_term(fs2s2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2["oovooo"] += -(1./12.) * np.einsum("bi,jkbA,aAlm->jkailm",f["vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2["oovooo"] += -(1./12.) * np.einsum("Ai,jkAB,aBlm->jkailm",f["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s2["oovooo"] = asym_term(fs2s2["oovooo"],"oovooo")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2["oovoov"] += -(1./18.) * np.einsum("BA,ijaB,bAkl->ijbkla",f["VV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2["oovoov"] += -(1./18.) * np.einsum("mi,jkaA,bAlm->jkbila",f["oo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2["oovoov"] += -(1./18.) * np.einsum("im,jmaA,bAkl->ijbkla",f["oo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2["oovoov"] += -(1./36.) * np.einsum("ca,ijcA,bAkl->ijbkla",f["vv"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2["oovoov"] += -(1./36.) * np.einsum("Aa,ijAB,bBkl->ijbkla",f["Vv"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s2["oovoov"] += -(1./36.) * np.einsum("ac,ijbA,cAkl->ijaklb",f["vv"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2["oovoov"] += -(1./36.) * np.einsum("aA,ijbB,ABkl->ijaklb",f["vV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2["oovoov"] = asym_term(fs2s2["oovoov"],"oovoov")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs2s2["oovovv"] +=  (1./18.) * np.einsum("la,ijbA,cAkl->ijckab",f["ov"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2["oovovv"] = asym_term(fs2s2["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2["ovvoov"] +=  (1./18.) * np.einsum("al,ilbA,cAjk->iacjkb",f["vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
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
	ws1s1["c"] +=  4.000 * np.einsum("Ai,Bj,ijAB->",t1["Vo"],t1["Vo"],v["ooVV"],optimize="optimal")
	ws1s1["c"] += -8.000 * np.einsum("iA,Bj,jAiB->",t1["oV"],t1["Vo"],v["oVoV"],optimize="optimal")
	ws1s1["c"] +=  4.000 * np.einsum("iA,jB,ABij->",t1["oV"],t1["oV"],v["VVoo"],optimize="optimal")

	ws1s1["oo"] +=  8.000 * np.einsum("Ai,Bk,jkAB->ji",t1["Vo"],t1["Vo"],v["ooVV"],optimize="optimal")
	ws1s1["oo"] += -8.000 * np.einsum("iA,Bk,kAjB->ij",t1["oV"],t1["Vo"],v["oVoV"],optimize="optimal")
	ws1s1["oo"] +=  8.000 * np.einsum("iA,kB,ABjk->ij",t1["oV"],t1["oV"],v["VVoo"],optimize="optimal")
	ws1s1["oo"] += -8.000 * np.einsum("kA,Bi,jAkB->ji",t1["oV"],t1["Vo"],v["oVoV"],optimize="optimal")
	ws1s1["oo"] +=  8.000 * np.einsum("kA,Bk,jAiB->ji",t1["oV"],t1["Vo"],v["oVoV"],optimize="optimal")
	ws1s1["oo"] += -8.000 * np.einsum("kA,Al,jlik->ji",t1["oV"],t1["Vo"],v["oooo"],optimize="optimal")

	ws1s1["ov"] += -8.000 * np.einsum("iA,Bj,jAaB->ia",t1["oV"],t1["Vo"],v["oVvV"],optimize="optimal")
	ws1s1["ov"] += -8.000 * np.einsum("iA,jB,ABja->ia",t1["oV"],t1["oV"],v["VVov"],optimize="optimal")
	ws1s1["ov"] +=  8.000 * np.einsum("jA,Bj,iAaB->ia",t1["oV"],t1["Vo"],v["oVvV"],optimize="optimal")
	ws1s1["ov"] +=  8.000 * np.einsum("jA,Ak,ikja->ia",t1["oV"],t1["Vo"],v["ooov"],optimize="optimal")

	ws1s1["vo"] += -8.000 * np.einsum("Ai,Bj,jaAB->ai",t1["Vo"],t1["Vo"],v["ovVV"],optimize="optimal")
	ws1s1["vo"] += -8.000 * np.einsum("jA,Bi,aAjB->ai",t1["oV"],t1["Vo"],v["vVoV"],optimize="optimal")
	ws1s1["vo"] +=  8.000 * np.einsum("jA,Bj,aAiB->ai",t1["oV"],t1["Vo"],v["vVoV"],optimize="optimal")
	ws1s1["vo"] +=  8.000 * np.einsum("jA,Ak,kaij->ai",t1["oV"],t1["Vo"],v["ovoo"],optimize="optimal")

	ws1s1["vv"] +=  8.000 * np.einsum("iA,Bi,bAaB->ba",t1["oV"],t1["Vo"],v["vVvV"],optimize="optimal")
	ws1s1["vv"] += -8.000 * np.einsum("iA,Aj,jbia->ba",t1["oV"],t1["Vo"],v["ovov"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	ws1s1["oooo"] +=  8.000 * np.einsum("iA,Bj,lAkB->iljk",t1["oV"],t1["Vo"],v["oVoV"],optimize="optimal")
	ws1s1["oooo"] +=  2.000 * np.einsum("Ai,Bj,klAB->klij",t1["Vo"],t1["Vo"],v["ooVV"],optimize="optimal")
	ws1s1["oooo"] +=  2.000 * np.einsum("iA,Am,lmjk->iljk",t1["oV"],t1["Vo"],v["oooo"],optimize="optimal")
	ws1s1["oooo"] +=  2.000 * np.einsum("iA,jB,ABkl->ijkl",t1["oV"],t1["oV"],v["VVoo"],optimize="optimal")
	ws1s1["oooo"] +=  2.000 * np.einsum("mA,Ai,kljm->klij",t1["oV"],t1["Vo"],v["oooo"],optimize="optimal")
	ws1s1["oooo"] = asym_term(ws1s1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	ws1s1["ooov"] +=  4.000 * np.einsum("iA,Bj,kAaB->ikja",t1["oV"],t1["Vo"],v["oVvV"],optimize="optimal")
	ws1s1["ooov"] +=  2.000 * np.einsum("iA,Al,klja->ikja",t1["oV"],t1["Vo"],v["ooov"],optimize="optimal")
	ws1s1["ooov"] +=  2.000 * np.einsum("iA,jB,ABka->ijka",t1["oV"],t1["oV"],v["VVov"],optimize="optimal")
	ws1s1["ooov"] += -1.000 * np.einsum("lA,Ai,jkla->jkia",t1["oV"],t1["Vo"],v["ooov"],optimize="optimal")
	ws1s1["ooov"] = asym_term(ws1s1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	ws1s1["ovoo"] +=  4.000 * np.einsum("iA,Bj,aAkB->iajk",t1["oV"],t1["Vo"],v["vVoV"],optimize="optimal")
	ws1s1["ovoo"] +=  2.000 * np.einsum("lA,Ai,kajl->kaij",t1["oV"],t1["Vo"],v["ovoo"],optimize="optimal")
	ws1s1["ovoo"] +=  2.000 * np.einsum("Ai,Bj,kaAB->kaij",t1["Vo"],t1["Vo"],v["ovVV"],optimize="optimal")
	ws1s1["ovoo"] += -1.000 * np.einsum("iA,Al,lajk->iajk",t1["oV"],t1["Vo"],v["ovoo"],optimize="optimal")
	ws1s1["ovoo"] = asym_term(ws1s1["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	ws1s1["oovv"] +=  2.000 * np.einsum("iA,Ak,jkab->ijab",t1["oV"],t1["Vo"],v["oovv"],optimize="optimal")
	ws1s1["oovv"] +=  2.000 * np.einsum("iA,jB,ABab->ijab",t1["oV"],t1["oV"],v["VVvv"],optimize="optimal")
	ws1s1["oovv"] = asym_term(ws1s1["oovv"],"oovv")

	ws1s1["ovov"] +=  2.000 * np.einsum("iA,Bj,bAaB->ibja",t1["oV"],t1["Vo"],v["vVvV"],optimize="optimal")
	ws1s1["ovov"] += -1.000 * np.einsum("iA,Ak,kbja->ibja",t1["oV"],t1["Vo"],v["ovov"],optimize="optimal")
	ws1s1["ovov"] += -1.000 * np.einsum("kA,Ai,jbka->jbia",t1["oV"],t1["Vo"],v["ovov"],optimize="optimal")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	ws1s1["vvoo"] += 2.000 * np.einsum("Ai,Bj,abAB->abij",t1["Vo"],t1["Vo"],v["vvVV"],optimize="optimal")
	ws1s1["vvoo"] += 2.000 * np.einsum("kA,Ai,abjk->abij",t1["oV"],t1["Vo"],v["vvoo"],optimize="optimal")
	ws1s1["vvoo"] = asym_term(ws1s1["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	ws1s1["ovvv"] += -1.000 * np.einsum("iA,Aj,jcab->icab",t1["oV"],t1["Vo"],v["ovvv"],optimize="optimal")
	ws1s1["ovvv"] = asym_term(ws1s1["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	ws1s1["vvov"] += -1.000 * np.einsum("jA,Ai,bcja->bcia",t1["oV"],t1["Vo"],v["vvov"],optimize="optimal")
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
	ws1s2["c"] +=  2.000 * np.einsum("Ai,jkaA,iajk->",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws1s2["c"] += -4.000 * np.einsum("Ai,ijaB,aBjA->",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws1s2["c"] += -2.000 * np.einsum("Ai,jkAB,iBjk->",t1["Vo"],t2["ooVV"],v["oVoo"],optimize="optimal")
	ws1s2["c"] += -2.000 * np.einsum("Ai,ijBC,BCjA->",t1["Vo"],t2["ooVV"],v["VVoV"],optimize="optimal")
	ws1s2["c"] += -4.000 * np.einsum("iA,aBij,jAaB->",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws1s2["c"] += -2.000 * np.einsum("iA,BCij,jABC->",t1["oV"],t2["VVoo"],v["oVVV"],optimize="optimal")
	ws1s2["c"] +=  2.000 * np.einsum("iA,aAjk,jkia->",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws1s2["c"] += -2.000 * np.einsum("iA,ABjk,jkiB->",t1["oV"],t2["VVoo"],v["oooV"],optimize="optimal")

	ws1s2["oo"] += -4.000 * np.einsum("Ai,jkaB,aBkA->ji",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws1s2["oo"] += -2.000 * np.einsum("Ai,jkBC,BCkA->ji",t1["Vo"],t2["ooVV"],v["VVoV"],optimize="optimal")
	ws1s2["oo"] +=  4.000 * np.einsum("Ak,ilaA,kajl->ij",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws1s2["oo"] +=  4.000 * np.einsum("Ak,ikaB,aBjA->ij",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws1s2["oo"] += -4.000 * np.einsum("Ak,ilAB,kBjl->ij",t1["Vo"],t2["ooVV"],v["oVoo"],optimize="optimal")
	ws1s2["oo"] +=  2.000 * np.einsum("Ak,ikBC,BCjA->ij",t1["Vo"],t2["ooVV"],v["VVoV"],optimize="optimal")
	ws1s2["oo"] += -4.000 * np.einsum("iA,aBjk,kAaB->ij",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws1s2["oo"] += -2.000 * np.einsum("iA,BCjk,kABC->ij",t1["oV"],t2["VVoo"],v["oVVV"],optimize="optimal")
	ws1s2["oo"] +=  4.000 * np.einsum("kA,aBik,jAaB->ji",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws1s2["oo"] +=  2.000 * np.einsum("kA,BCik,jABC->ji",t1["oV"],t2["VVoo"],v["oVVV"],optimize="optimal")
	ws1s2["oo"] +=  4.000 * np.einsum("kA,aAil,jlka->ji",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws1s2["oo"] += -4.000 * np.einsum("kA,ABil,jlkB->ji",t1["oV"],t2["VVoo"],v["oooV"],optimize="optimal")

	ws1s2["ov"] +=  2.000 * np.einsum("Aj,klaA,ijkl->ia",t1["Vo"],t2["oovV"],v["oooo"],optimize="optimal")
	ws1s2["ov"] += -4.000 * np.einsum("Aj,ikaB,jBkA->ia",t1["Vo"],t2["oovV"],v["oVoV"],optimize="optimal")
	ws1s2["ov"] +=  4.000 * np.einsum("Aj,jkaB,iBkA->ia",t1["Vo"],t2["oovV"],v["oVoV"],optimize="optimal")
	ws1s2["ov"] += -4.000 * np.einsum("Aj,ikbA,jbka->ia",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
	ws1s2["ov"] +=  4.000 * np.einsum("Aj,ijbB,bBaA->ia",t1["Vo"],t2["oovV"],v["vVvV"],optimize="optimal")
	ws1s2["ov"] +=  4.000 * np.einsum("Aj,ikAB,jBka->ia",t1["Vo"],t2["ooVV"],v["oVov"],optimize="optimal")
	ws1s2["ov"] +=  2.000 * np.einsum("Aj,ijBC,BCaA->ia",t1["Vo"],t2["ooVV"],v["VVvV"],optimize="optimal")
	ws1s2["ov"] += -2.000 * np.einsum("iA,jkaB,ABjk->ia",t1["oV"],t2["oovV"],v["VVoo"],optimize="optimal")
	ws1s2["ov"] +=  4.000 * np.einsum("jA,ikaB,ABjk->ia",t1["oV"],t2["oovV"],v["VVoo"],optimize="optimal")

	ws1s2["vo"] += -2.000 * np.einsum("Ai,aBjk,jkAB->ai",t1["Vo"],t2["vVoo"],v["ooVV"],optimize="optimal")
	ws1s2["vo"] +=  4.000 * np.einsum("Aj,aBik,jkAB->ai",t1["Vo"],t2["vVoo"],v["ooVV"],optimize="optimal")
	ws1s2["vo"] +=  4.000 * np.einsum("jA,bBij,aAbB->ai",t1["oV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws1s2["vo"] +=  2.000 * np.einsum("jA,BCij,aABC->ai",t1["oV"],t2["VVoo"],v["vVVV"],optimize="optimal")
	ws1s2["vo"] += -4.000 * np.einsum("jA,aBik,kAjB->ai",t1["oV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws1s2["vo"] += -4.000 * np.einsum("jA,bAik,kajb->ai",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws1s2["vo"] +=  4.000 * np.einsum("jA,ABik,kajB->ai",t1["oV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws1s2["vo"] +=  4.000 * np.einsum("jA,aBjk,kAiB->ai",t1["oV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws1s2["vo"] +=  2.000 * np.einsum("jA,aAkl,klij->ai",t1["oV"],t2["vVoo"],v["oooo"],optimize="optimal")

	ws1s2["vv"] += -2.000 * np.einsum("Ai,jkaA,ibjk->ba",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws1s2["vv"] +=  4.000 * np.einsum("Ai,ijaB,bBjA->ba",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws1s2["vv"] +=  4.000 * np.einsum("iA,aBij,jAbB->ab",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws1s2["vv"] += -2.000 * np.einsum("iA,aAjk,jkib->ab",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	ws1s2["oooo"] += -2.000 * np.einsum("Ai,jkaB,aBlA->jkil",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws1s2["oooo"] += -2.000 * np.einsum("iA,aBjk,lAaB->iljk",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws1s2["oooo"] += -1.000 * np.einsum("Ai,jkBC,BClA->jkil",t1["Vo"],t2["ooVV"],v["VVoV"],optimize="optimal")
	ws1s2["oooo"] +=  1.000 * np.einsum("Am,ijaA,makl->ijkl",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws1s2["oooo"] += -1.000 * np.einsum("Am,ijAB,mBkl->ijkl",t1["Vo"],t2["ooVV"],v["oVoo"],optimize="optimal")
	ws1s2["oooo"] += -1.000 * np.einsum("iA,BCjk,lABC->iljk",t1["oV"],t2["VVoo"],v["oVVV"],optimize="optimal")
	ws1s2["oooo"] +=  1.000 * np.einsum("mA,aAij,klma->klij",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws1s2["oooo"] += -1.000 * np.einsum("mA,ABij,klmB->klij",t1["oV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws1s2["oooo"] = asym_term(ws1s2["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	ws1s2["ooov"] +=  2.000 * np.einsum("Ai,jlaB,kBlA->jkia",t1["Vo"],t2["oovV"],v["oVoV"],optimize="optimal")
	ws1s2["ooov"] +=  2.000 * np.einsum("Al,imaA,kljm->ikja",t1["Vo"],t2["oovV"],v["oooo"],optimize="optimal")
	ws1s2["ooov"] += -2.000 * np.einsum("Al,ilaB,kBjA->ikja",t1["Vo"],t2["oovV"],v["oVoV"],optimize="optimal")
	ws1s2["ooov"] +=  2.000 * np.einsum("iA,jlaB,ABkl->ijka",t1["oV"],t2["oovV"],v["VVoo"],optimize="optimal")
	ws1s2["ooov"] += -1.000 * np.einsum("Ai,jkbB,bBaA->jkia",t1["Vo"],t2["oovV"],v["vVvV"],optimize="optimal")
	ws1s2["ooov"] +=  1.000 * np.einsum("Al,ijaB,lBkA->ijka",t1["Vo"],t2["oovV"],v["oVoV"],optimize="optimal")
	ws1s2["ooov"] +=  1.000 * np.einsum("Al,ijbA,lbka->ijka",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
	ws1s2["ooov"] += -1.000 * np.einsum("Al,ijAB,lBka->ijka",t1["Vo"],t2["ooVV"],v["oVov"],optimize="optimal")
	ws1s2["ooov"] +=  1.000 * np.einsum("lA,ijaB,ABkl->ijka",t1["oV"],t2["oovV"],v["VVoo"],optimize="optimal")
	ws1s2["ooov"] += -0.500 * np.einsum("Ai,jkBC,BCaA->jkia",t1["Vo"],t2["ooVV"],v["VVvV"],optimize="optimal")
	ws1s2["ooov"] = asym_term(ws1s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	ws1s2["ovoo"] +=  2.000 * np.einsum("Ai,aBjl,klAB->kaij",t1["Vo"],t2["vVoo"],v["ooVV"],optimize="optimal")
	ws1s2["ovoo"] +=  2.000 * np.einsum("iA,aBjl,lAkB->iajk",t1["oV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws1s2["ovoo"] += -2.000 * np.einsum("lA,aBil,kAjB->kaij",t1["oV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws1s2["ovoo"] +=  2.000 * np.einsum("lA,aAim,kmjl->kaij",t1["oV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws1s2["ovoo"] +=  1.000 * np.einsum("Al,aBij,klAB->kaij",t1["Vo"],t2["vVoo"],v["ooVV"],optimize="optimal")
	ws1s2["ovoo"] += -1.000 * np.einsum("iA,bBjk,aAbB->iajk",t1["oV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws1s2["ovoo"] +=  1.000 * np.einsum("lA,aBij,kAlB->kaij",t1["oV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws1s2["ovoo"] +=  1.000 * np.einsum("lA,bAij,kalb->kaij",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws1s2["ovoo"] += -1.000 * np.einsum("lA,ABij,kalB->kaij",t1["oV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws1s2["ovoo"] += -0.500 * np.einsum("iA,BCjk,aABC->iajk",t1["oV"],t2["VVoo"],v["vVVV"],optimize="optimal")
	ws1s2["ovoo"] = asym_term(ws1s2["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	ws1s2["oovv"] +=  4.000 * np.einsum("Ak,ilaA,jklb->ijab",t1["Vo"],t2["oovV"],v["ooov"],optimize="optimal")
	ws1s2["oovv"] +=  4.000 * np.einsum("Ak,ikaB,jBbA->ijab",t1["Vo"],t2["oovV"],v["oVvV"],optimize="optimal")
	ws1s2["oovv"] +=  4.000 * np.einsum("iA,jkaB,ABkb->ijab",t1["oV"],t2["oovV"],v["VVov"],optimize="optimal")
	ws1s2["oovv"] +=  2.000 * np.einsum("kA,ijaB,ABkb->ijab",t1["oV"],t2["oovV"],v["VVov"],optimize="optimal")
	ws1s2["oovv"] += -2.000 * np.einsum("Ak,ijaB,kBbA->ijab",t1["Vo"],t2["oovV"],v["oVvV"],optimize="optimal")
	ws1s2["oovv"] +=  1.000 * np.einsum("Ak,ijcA,kcab->ijab",t1["Vo"],t2["oovV"],v["ovvv"],optimize="optimal")
	ws1s2["oovv"] += -1.000 * np.einsum("Ak,ijAB,kBab->ijab",t1["Vo"],t2["ooVV"],v["oVvv"],optimize="optimal")
	ws1s2["oovv"] = asym_term(ws1s2["oovv"],"oovv")

	ws1s2["ovov"] +=  1.000 * np.einsum("Ai,jkaB,bBkA->jbia",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws1s2["ovov"] += -1.000 * np.einsum("Ak,ilaA,kbjl->ibja",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws1s2["ovov"] += -1.000 * np.einsum("Ak,ikaB,bBjA->ibja",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws1s2["ovov"] +=  1.000 * np.einsum("iA,aBjk,kAbB->iajb",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws1s2["ovov"] += -1.000 * np.einsum("kA,aBik,jAbB->jaib",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws1s2["ovov"] += -1.000 * np.einsum("kA,aAil,jlkb->jaib",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	ws1s2["vvoo"] +=  4.000 * np.einsum("Ai,aBjk,kbAB->abij",t1["Vo"],t2["vVoo"],v["ovVV"],optimize="optimal")
	ws1s2["vvoo"] +=  4.000 * np.einsum("kA,aBik,bAjB->abij",t1["oV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws1s2["vvoo"] +=  4.000 * np.einsum("kA,aAil,lbjk->abij",t1["oV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws1s2["vvoo"] +=  2.000 * np.einsum("Ak,aBij,kbAB->abij",t1["Vo"],t2["vVoo"],v["ovVV"],optimize="optimal")
	ws1s2["vvoo"] += -2.000 * np.einsum("kA,aBij,bAkB->abij",t1["oV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws1s2["vvoo"] +=  1.000 * np.einsum("kA,cAij,abkc->abij",t1["oV"],t2["vVoo"],v["vvov"],optimize="optimal")
	ws1s2["vvoo"] += -1.000 * np.einsum("kA,ABij,abkB->abij",t1["oV"],t2["VVoo"],v["vvoV"],optimize="optimal")
	ws1s2["vvoo"] = asym_term(ws1s2["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	ws1s2["ovvv"] += -2.000 * np.einsum("Aj,ikaA,jckb->icab",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
	ws1s2["ovvv"] +=  2.000 * np.einsum("Aj,ijaB,cBbA->icab",t1["Vo"],t2["oovV"],v["vVvV"],optimize="optimal")
	ws1s2["ovvv"] = asym_term(ws1s2["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	ws1s2["vvov"] +=  2.000 * np.einsum("jA,aBij,cAbB->acib",t1["oV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws1s2["vvov"] += -2.000 * np.einsum("jA,aAik,kcjb->acib",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
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
		ws1s2["ooooov"] +=  (2./3.) * np.einsum("Ai,jkaB,mBlA->jkmila",t1["Vo"],t2["oovV"],v["oVoV"],optimize="optimal")
		ws1s2["ooooov"] +=  (1./3.) * np.einsum("An,ijaA,mnkl->ijmkla",t1["Vo"],t2["oovV"],v["oooo"],optimize="optimal")
		ws1s2["ooooov"] += -(1./3.) * np.einsum("iA,jkaB,ABlm->ijklma",t1["oV"],t2["oovV"],v["VVoo"],optimize="optimal")
		ws1s2["ooooov"] = asym_term(ws1s2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		ws1s2["oovooo"] +=  (2./3.) * np.einsum("iA,aBjk,mAlB->imajkl",t1["oV"],t2["vVoo"],v["oVoV"],optimize="optimal")
		ws1s2["oovooo"] += -(1./3.) * np.einsum("Ai,aBjk,lmAB->lmaijk",t1["Vo"],t2["vVoo"],v["ooVV"],optimize="optimal")
		ws1s2["oovooo"] +=  (1./3.) * np.einsum("nA,aAij,lmkn->lmaijk",t1["oV"],t2["vVoo"],v["oooo"],optimize="optimal")
		ws1s2["oovooo"] = asym_term(ws1s2["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws1s2["oooovv"] += -(2./3.) * np.einsum("Ai,jkaB,lBbA->jkliab",t1["Vo"],t2["oovV"],v["oVvV"],optimize="optimal")
		ws1s2["oooovv"] += -(2./3.) * np.einsum("Am,ijaA,lmkb->ijlkab",t1["Vo"],t2["oovV"],v["ooov"],optimize="optimal")
		ws1s2["oooovv"] +=  (2./3.) * np.einsum("iA,jkaB,ABlb->ijklab",t1["oV"],t2["oovV"],v["VVov"],optimize="optimal")
		ws1s2["oooovv"] = asym_term(ws1s2["oooovv"],"oooovv")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		ws1s2["oovoov"] +=  (2./9.) * np.einsum("Ai,jkaB,bBlA->jkbila",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
		ws1s2["oovoov"] +=  (2./9.) * np.einsum("iA,aBjk,lAbB->ilajkb",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
		ws1s2["oovoov"] += -(1./9.) * np.einsum("Am,ijaA,mbkl->ijbkla",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
		ws1s2["oovoov"] += -(1./9.) * np.einsum("mA,aAij,klmb->klaijb",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws1s2["oovoov"] = asym_term(ws1s2["oovoov"],"oovoov")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws1s2["ovvooo"] +=  (2./3.) * np.einsum("Ai,aBjk,lbAB->labijk",t1["Vo"],t2["vVoo"],v["ovVV"],optimize="optimal")
		ws1s2["ovvooo"] += -(2./3.) * np.einsum("iA,aBjk,bAlB->iabjkl",t1["oV"],t2["vVoo"],v["vVoV"],optimize="optimal")
		ws1s2["ovvooo"] += -(2./3.) * np.einsum("mA,aAij,lbkm->labijk",t1["oV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws1s2["ovvooo"] = asym_term(ws1s2["ovvooo"],"ovvooo")

		# ooovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws1s2["ooovvv"] +=  1.000 * np.einsum("Al,ijaA,klbc->ijkabc",t1["Vo"],t2["oovV"],v["oovv"],optimize="optimal")
		ws1s2["ooovvv"] += -1.000 * np.einsum("iA,jkaB,ABbc->ijkabc",t1["oV"],t2["oovV"],v["VVvv"],optimize="optimal")
		ws1s2["ooovvv"] = asym_term(ws1s2["ooovvv"],"ooovvv")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws1s2["oovovv"] += -(2./9.) * np.einsum("Ai,jkaB,cBbA->jkciab",t1["Vo"],t2["oovV"],v["vVvV"],optimize="optimal")
		ws1s2["oovovv"] +=  (2./9.) * np.einsum("Al,ijaA,lckb->ijckab",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
		ws1s2["oovovv"] = asym_term(ws1s2["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws1s2["ovvoov"] += -(2./9.) * np.einsum("iA,aBjk,cAbB->iacjkb",t1["oV"],t2["vVoo"],v["vVvV"],optimize="optimal")
		ws1s2["ovvoov"] +=  (2./9.) * np.einsum("lA,aAij,kclb->kacijb",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws1s2["ovvoov"] = asym_term(ws1s2["ovvoov"],"ovvoov")

		# vvvooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws1s2["vvvooo"] += -1.000 * np.einsum("Ai,aBjk,bcAB->abcijk",t1["Vo"],t2["vVoo"],v["vvVV"],optimize="optimal")
		ws1s2["vvvooo"] +=  1.000 * np.einsum("lA,aAij,bckl->abcijk",t1["oV"],t2["vVoo"],v["vvoo"],optimize="optimal")
		ws1s2["vvvooo"] = asym_term(ws1s2["vvvooo"],"vvvooo")

		# oovvvv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws1s2["oovvvv"] += -(1./3.) * np.einsum("Ak,ijaA,kdbc->ijdabc",t1["Vo"],t2["oovV"],v["ovvv"],optimize="optimal")
		ws1s2["oovvvv"] = asym_term(ws1s2["oovvvv"],"oovvvv")

		# vvvoov = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws1s2["vvvoov"] += -(1./3.) * np.einsum("kA,aAij,cdkb->acdijb",t1["oV"],t2["vVoo"],v["vvov"],optimize="optimal")
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
	ws2s1["c"] +=  2.000 * np.einsum("Ai,jkaA,iajk->",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws2s1["c"] += -4.000 * np.einsum("Ai,ijaB,aBjA->",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws2s1["c"] += -2.000 * np.einsum("Ai,jkAB,iBjk->",t1["Vo"],t2["ooVV"],v["oVoo"],optimize="optimal")
	ws2s1["c"] += -2.000 * np.einsum("Ai,ijBC,BCjA->",t1["Vo"],t2["ooVV"],v["VVoV"],optimize="optimal")
	ws2s1["c"] += -4.000 * np.einsum("iA,aBij,jAaB->",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s1["c"] += -2.000 * np.einsum("iA,BCij,jABC->",t1["oV"],t2["VVoo"],v["oVVV"],optimize="optimal")
	ws2s1["c"] +=  2.000 * np.einsum("iA,aAjk,jkia->",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s1["c"] += -2.000 * np.einsum("iA,ABjk,jkiB->",t1["oV"],t2["VVoo"],v["oooV"],optimize="optimal")

	ws2s1["oo"] +=  2.000 * np.einsum("Ai,klaA,jakl->ji",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws2s1["oo"] += -4.000 * np.einsum("Ai,jkaB,aBkA->ji",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws2s1["oo"] += -2.000 * np.einsum("Ai,klAB,jBkl->ji",t1["Vo"],t2["ooVV"],v["oVoo"],optimize="optimal")
	ws2s1["oo"] += -2.000 * np.einsum("Ai,jkBC,BCkA->ji",t1["Vo"],t2["ooVV"],v["VVoV"],optimize="optimal")
	ws2s1["oo"] +=  4.000 * np.einsum("Ak,ilaA,kajl->ij",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws2s1["oo"] += -4.000 * np.einsum("Ak,klaA,jail->ji",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws2s1["oo"] +=  4.000 * np.einsum("Ak,ikaB,aBjA->ij",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws2s1["oo"] += -4.000 * np.einsum("Ak,ilAB,kBjl->ij",t1["Vo"],t2["ooVV"],v["oVoo"],optimize="optimal")
	ws2s1["oo"] +=  4.000 * np.einsum("Ak,klAB,jBil->ji",t1["Vo"],t2["ooVV"],v["oVoo"],optimize="optimal")
	ws2s1["oo"] +=  2.000 * np.einsum("Ak,ikBC,BCjA->ij",t1["Vo"],t2["ooVV"],v["VVoV"],optimize="optimal")
	ws2s1["oo"] += -4.000 * np.einsum("iA,aBjk,kAaB->ij",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s1["oo"] += -2.000 * np.einsum("iA,BCjk,kABC->ij",t1["oV"],t2["VVoo"],v["oVVV"],optimize="optimal")
	ws2s1["oo"] +=  2.000 * np.einsum("iA,aAkl,klja->ij",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s1["oo"] += -2.000 * np.einsum("iA,ABkl,kljB->ij",t1["oV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws2s1["oo"] +=  4.000 * np.einsum("kA,aBik,jAaB->ji",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s1["oo"] +=  2.000 * np.einsum("kA,BCik,jABC->ji",t1["oV"],t2["VVoo"],v["oVVV"],optimize="optimal")
	ws2s1["oo"] +=  4.000 * np.einsum("kA,aAil,jlka->ji",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s1["oo"] += -4.000 * np.einsum("kA,ABil,jlkB->ji",t1["oV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws2s1["oo"] += -4.000 * np.einsum("kA,aAkl,jlia->ji",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s1["oo"] +=  4.000 * np.einsum("kA,ABkl,jliB->ji",t1["oV"],t2["VVoo"],v["oooV"],optimize="optimal")

	ws2s1["ov"] +=  2.000 * np.einsum("Aj,klaA,ijkl->ia",t1["Vo"],t2["oovV"],v["oooo"],optimize="optimal")
	ws2s1["ov"] += -4.000 * np.einsum("Aj,ikaB,jBkA->ia",t1["Vo"],t2["oovV"],v["oVoV"],optimize="optimal")
	ws2s1["ov"] +=  4.000 * np.einsum("Aj,jkaB,iBkA->ia",t1["Vo"],t2["oovV"],v["oVoV"],optimize="optimal")
	ws2s1["ov"] += -4.000 * np.einsum("Aj,ikbA,jbka->ia",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
	ws2s1["ov"] +=  4.000 * np.einsum("Aj,jkbA,ibka->ia",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
	ws2s1["ov"] +=  4.000 * np.einsum("Aj,ijbB,bBaA->ia",t1["Vo"],t2["oovV"],v["vVvV"],optimize="optimal")
	ws2s1["ov"] +=  4.000 * np.einsum("Aj,ikAB,jBka->ia",t1["Vo"],t2["ooVV"],v["oVov"],optimize="optimal")
	ws2s1["ov"] += -4.000 * np.einsum("Aj,jkAB,iBka->ia",t1["Vo"],t2["ooVV"],v["oVov"],optimize="optimal")
	ws2s1["ov"] +=  2.000 * np.einsum("Aj,ijBC,BCaA->ia",t1["Vo"],t2["ooVV"],v["VVvV"],optimize="optimal")
	ws2s1["ov"] +=  2.000 * np.einsum("iA,bAjk,jkab->ia",t1["oV"],t2["vVoo"],v["oovv"],optimize="optimal")
	ws2s1["ov"] += -2.000 * np.einsum("iA,ABjk,jkaB->ia",t1["oV"],t2["VVoo"],v["oovV"],optimize="optimal")
	ws2s1["ov"] += -2.000 * np.einsum("iA,jkaB,ABjk->ia",t1["oV"],t2["oovV"],v["VVoo"],optimize="optimal")
	ws2s1["ov"] += -4.000 * np.einsum("jA,bAjk,ikab->ia",t1["oV"],t2["vVoo"],v["oovv"],optimize="optimal")
	ws2s1["ov"] +=  4.000 * np.einsum("jA,ABjk,ikaB->ia",t1["oV"],t2["VVoo"],v["oovV"],optimize="optimal")
	ws2s1["ov"] +=  4.000 * np.einsum("jA,ikaB,ABjk->ia",t1["oV"],t2["oovV"],v["VVoo"],optimize="optimal")

	ws2s1["vo"] += -2.000 * np.einsum("Ai,aBjk,jkAB->ai",t1["Vo"],t2["vVoo"],v["ooVV"],optimize="optimal")
	ws2s1["vo"] +=  2.000 * np.einsum("Ai,jkbA,abjk->ai",t1["Vo"],t2["oovV"],v["vvoo"],optimize="optimal")
	ws2s1["vo"] += -2.000 * np.einsum("Ai,jkAB,aBjk->ai",t1["Vo"],t2["ooVV"],v["vVoo"],optimize="optimal")
	ws2s1["vo"] +=  4.000 * np.einsum("Aj,aBik,jkAB->ai",t1["Vo"],t2["vVoo"],v["ooVV"],optimize="optimal")
	ws2s1["vo"] += -4.000 * np.einsum("Aj,jkbA,abik->ai",t1["Vo"],t2["oovV"],v["vvoo"],optimize="optimal")
	ws2s1["vo"] +=  4.000 * np.einsum("Aj,jkAB,aBik->ai",t1["Vo"],t2["ooVV"],v["vVoo"],optimize="optimal")
	ws2s1["vo"] +=  4.000 * np.einsum("jA,bBij,aAbB->ai",t1["oV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s1["vo"] +=  2.000 * np.einsum("jA,BCij,aABC->ai",t1["oV"],t2["VVoo"],v["vVVV"],optimize="optimal")
	ws2s1["vo"] += -4.000 * np.einsum("jA,aBik,kAjB->ai",t1["oV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s1["vo"] += -4.000 * np.einsum("jA,bAik,kajb->ai",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s1["vo"] +=  4.000 * np.einsum("jA,ABik,kajB->ai",t1["oV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s1["vo"] +=  4.000 * np.einsum("jA,aBjk,kAiB->ai",t1["oV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s1["vo"] +=  4.000 * np.einsum("jA,bAjk,kaib->ai",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s1["vo"] += -4.000 * np.einsum("jA,ABjk,kaiB->ai",t1["oV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s1["vo"] +=  2.000 * np.einsum("jA,aAkl,klij->ai",t1["oV"],t2["vVoo"],v["oooo"],optimize="optimal")

	ws2s1["vv"] += -2.000 * np.einsum("Ai,jkaA,ibjk->ba",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws2s1["vv"] +=  4.000 * np.einsum("Ai,ijaB,bBjA->ba",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws2s1["vv"] +=  4.000 * np.einsum("Ai,ijcA,bcja->ba",t1["Vo"],t2["oovV"],v["vvov"],optimize="optimal")
	ws2s1["vv"] += -4.000 * np.einsum("Ai,ijAB,bBja->ba",t1["Vo"],t2["ooVV"],v["vVov"],optimize="optimal")
	ws2s1["vv"] +=  4.000 * np.einsum("iA,aBij,jAbB->ab",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s1["vv"] +=  4.000 * np.einsum("iA,cAij,jbac->ba",t1["oV"],t2["vVoo"],v["ovvv"],optimize="optimal")
	ws2s1["vv"] += -4.000 * np.einsum("iA,ABij,jbaB->ba",t1["oV"],t2["VVoo"],v["ovvV"],optimize="optimal")
	ws2s1["vv"] += -2.000 * np.einsum("iA,aAjk,jkib->ab",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	ws2s1["oooo"] += -4.000 * np.einsum("Ai,jmaA,lakm->jlik",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws2s1["oooo"] +=  4.000 * np.einsum("Ai,jmAB,lBkm->jlik",t1["Vo"],t2["ooVV"],v["oVoo"],optimize="optimal")
	ws2s1["oooo"] += -4.000 * np.einsum("iA,aAjm,lmka->iljk",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s1["oooo"] +=  4.000 * np.einsum("iA,ABjm,lmkB->iljk",t1["oV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws2s1["oooo"] += -2.000 * np.einsum("Am,imaA,lajk->iljk",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws2s1["oooo"] +=  2.000 * np.einsum("Am,imAB,lBjk->iljk",t1["Vo"],t2["ooVV"],v["oVoo"],optimize="optimal")
	ws2s1["oooo"] += -2.000 * np.einsum("iA,aBjk,lAaB->iljk",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s1["oooo"] += -2.000 * np.einsum("mA,aAim,klja->klij",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s1["oooo"] +=  2.000 * np.einsum("mA,ABim,kljB->klij",t1["oV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws2s1["oooo"] += -2.000 * np.einsum("Ai,jkaB,aBlA->jkil",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws2s1["oooo"] += -1.000 * np.einsum("Ai,jkBC,BClA->jkil",t1["Vo"],t2["ooVV"],v["VVoV"],optimize="optimal")
	ws2s1["oooo"] +=  1.000 * np.einsum("Am,ijaA,makl->ijkl",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws2s1["oooo"] += -1.000 * np.einsum("Am,ijAB,mBkl->ijkl",t1["Vo"],t2["ooVV"],v["oVoo"],optimize="optimal")
	ws2s1["oooo"] += -1.000 * np.einsum("iA,BCjk,lABC->iljk",t1["oV"],t2["VVoo"],v["oVVV"],optimize="optimal")
	ws2s1["oooo"] +=  1.000 * np.einsum("mA,aAij,klma->klij",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s1["oooo"] += -1.000 * np.einsum("mA,ABij,klmB->klij",t1["oV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws2s1["oooo"] = asym_term(ws2s1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	ws2s1["ooov"] +=  2.000 * np.einsum("Ai,jlaB,kBlA->jkia",t1["Vo"],t2["oovV"],v["oVoV"],optimize="optimal")
	ws2s1["ooov"] +=  2.000 * np.einsum("Ai,jlbA,kbla->jkia",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
	ws2s1["ooov"] += -2.000 * np.einsum("Ai,jlAB,kBla->jkia",t1["Vo"],t2["ooVV"],v["oVov"],optimize="optimal")
	ws2s1["ooov"] +=  2.000 * np.einsum("Al,imaA,kljm->ikja",t1["Vo"],t2["oovV"],v["oooo"],optimize="optimal")
	ws2s1["ooov"] += -2.000 * np.einsum("Al,ilaB,kBjA->ikja",t1["Vo"],t2["oovV"],v["oVoV"],optimize="optimal")
	ws2s1["ooov"] += -2.000 * np.einsum("Al,ilbA,kbja->ikja",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
	ws2s1["ooov"] +=  2.000 * np.einsum("Al,ilAB,kBja->ikja",t1["Vo"],t2["ooVV"],v["oVov"],optimize="optimal")
	ws2s1["ooov"] += -2.000 * np.einsum("iA,bAjl,klab->ikja",t1["oV"],t2["vVoo"],v["oovv"],optimize="optimal")
	ws2s1["ooov"] +=  2.000 * np.einsum("iA,ABjl,klaB->ikja",t1["oV"],t2["VVoo"],v["oovV"],optimize="optimal")
	ws2s1["ooov"] +=  2.000 * np.einsum("iA,jlaB,ABkl->ijka",t1["oV"],t2["oovV"],v["VVoo"],optimize="optimal")
	ws2s1["ooov"] += -1.000 * np.einsum("Ai,jkbB,bBaA->jkia",t1["Vo"],t2["oovV"],v["vVvV"],optimize="optimal")
	ws2s1["ooov"] +=  1.000 * np.einsum("Al,lmaA,jkim->jkia",t1["Vo"],t2["oovV"],v["oooo"],optimize="optimal")
	ws2s1["ooov"] +=  1.000 * np.einsum("Al,ijaB,lBkA->ijka",t1["Vo"],t2["oovV"],v["oVoV"],optimize="optimal")
	ws2s1["ooov"] +=  1.000 * np.einsum("Al,ijbA,lbka->ijka",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
	ws2s1["ooov"] += -1.000 * np.einsum("Al,ijAB,lBka->ijka",t1["Vo"],t2["ooVV"],v["oVov"],optimize="optimal")
	ws2s1["ooov"] += -1.000 * np.einsum("lA,bAil,jkab->jkia",t1["oV"],t2["vVoo"],v["oovv"],optimize="optimal")
	ws2s1["ooov"] +=  1.000 * np.einsum("lA,ABil,jkaB->jkia",t1["oV"],t2["VVoo"],v["oovV"],optimize="optimal")
	ws2s1["ooov"] +=  1.000 * np.einsum("lA,ijaB,ABkl->ijka",t1["oV"],t2["oovV"],v["VVoo"],optimize="optimal")
	ws2s1["ooov"] += -0.500 * np.einsum("Ai,lmaA,jklm->jkia",t1["Vo"],t2["oovV"],v["oooo"],optimize="optimal")
	ws2s1["ooov"] += -0.500 * np.einsum("Ai,jkBC,BCaA->jkia",t1["Vo"],t2["ooVV"],v["VVvV"],optimize="optimal")
	ws2s1["ooov"] = asym_term(ws2s1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	ws2s1["ovoo"] +=  2.000 * np.einsum("Ai,aBjl,klAB->kaij",t1["Vo"],t2["vVoo"],v["ooVV"],optimize="optimal")
	ws2s1["ovoo"] += -2.000 * np.einsum("Ai,jlbA,abkl->jaik",t1["Vo"],t2["oovV"],v["vvoo"],optimize="optimal")
	ws2s1["ovoo"] +=  2.000 * np.einsum("Ai,jlAB,aBkl->jaik",t1["Vo"],t2["ooVV"],v["vVoo"],optimize="optimal")
	ws2s1["ovoo"] +=  2.000 * np.einsum("iA,aBjl,lAkB->iajk",t1["oV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s1["ovoo"] +=  2.000 * np.einsum("iA,bAjl,lakb->iajk",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s1["ovoo"] += -2.000 * np.einsum("iA,ABjl,lakB->iajk",t1["oV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s1["ovoo"] += -2.000 * np.einsum("lA,aBil,kAjB->kaij",t1["oV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s1["ovoo"] += -2.000 * np.einsum("lA,bAil,kajb->kaij",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s1["ovoo"] +=  2.000 * np.einsum("lA,ABil,kajB->kaij",t1["oV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s1["ovoo"] +=  2.000 * np.einsum("lA,aAim,kmjl->kaij",t1["oV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s1["ovoo"] +=  1.000 * np.einsum("Al,aBij,klAB->kaij",t1["Vo"],t2["vVoo"],v["ooVV"],optimize="optimal")
	ws2s1["ovoo"] += -1.000 * np.einsum("Al,ilbA,abjk->iajk",t1["Vo"],t2["oovV"],v["vvoo"],optimize="optimal")
	ws2s1["ovoo"] +=  1.000 * np.einsum("Al,ilAB,aBjk->iajk",t1["Vo"],t2["ooVV"],v["vVoo"],optimize="optimal")
	ws2s1["ovoo"] += -1.000 * np.einsum("iA,bBjk,aAbB->iajk",t1["oV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s1["ovoo"] +=  1.000 * np.einsum("lA,aBij,kAlB->kaij",t1["oV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s1["ovoo"] +=  1.000 * np.einsum("lA,bAij,kalb->kaij",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s1["ovoo"] += -1.000 * np.einsum("lA,ABij,kalB->kaij",t1["oV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s1["ovoo"] +=  1.000 * np.einsum("lA,aAlm,kmij->kaij",t1["oV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s1["ovoo"] += -0.500 * np.einsum("iA,BCjk,aABC->iajk",t1["oV"],t2["VVoo"],v["vVVV"],optimize="optimal")
	ws2s1["ovoo"] += -0.500 * np.einsum("iA,aAlm,lmjk->iajk",t1["oV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s1["ovoo"] = asym_term(ws2s1["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	ws2s1["oovv"] +=  4.000 * np.einsum("Ak,ilaA,jklb->ijab",t1["Vo"],t2["oovV"],v["ooov"],optimize="optimal")
	ws2s1["oovv"] +=  4.000 * np.einsum("Ak,ikaB,jBbA->ijab",t1["Vo"],t2["oovV"],v["oVvV"],optimize="optimal")
	ws2s1["oovv"] +=  4.000 * np.einsum("iA,jkaB,ABkb->ijab",t1["oV"],t2["oovV"],v["VVov"],optimize="optimal")
	ws2s1["oovv"] +=  2.000 * np.einsum("Ak,klaA,ijlb->ijab",t1["Vo"],t2["oovV"],v["ooov"],optimize="optimal")
	ws2s1["oovv"] += -2.000 * np.einsum("Ak,ijaB,kBbA->ijab",t1["Vo"],t2["oovV"],v["oVvV"],optimize="optimal")
	ws2s1["oovv"] += -2.000 * np.einsum("Ak,ikcA,jcab->ijab",t1["Vo"],t2["oovV"],v["ovvv"],optimize="optimal")
	ws2s1["oovv"] +=  2.000 * np.einsum("Ak,ikAB,jBab->ijab",t1["Vo"],t2["ooVV"],v["oVvv"],optimize="optimal")
	ws2s1["oovv"] +=  2.000 * np.einsum("kA,ijaB,ABkb->ijab",t1["oV"],t2["oovV"],v["VVov"],optimize="optimal")
	ws2s1["oovv"] +=  1.000 * np.einsum("Ak,ijcA,kcab->ijab",t1["Vo"],t2["oovV"],v["ovvv"],optimize="optimal")
	ws2s1["oovv"] += -1.000 * np.einsum("Ak,ijAB,kBab->ijab",t1["Vo"],t2["ooVV"],v["oVvv"],optimize="optimal")
	ws2s1["oovv"] = asym_term(ws2s1["oovv"],"oovv")

	ws2s1["ovov"] +=  1.000 * np.einsum("Ai,jkaB,bBkA->jbia",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws2s1["ovov"] +=  1.000 * np.einsum("Ai,jkcA,bcka->jbia",t1["Vo"],t2["oovV"],v["vvov"],optimize="optimal")
	ws2s1["ovov"] += -1.000 * np.einsum("Ai,jkAB,bBka->jbia",t1["Vo"],t2["ooVV"],v["vVov"],optimize="optimal")
	ws2s1["ovov"] += -1.000 * np.einsum("Ak,ilaA,kbjl->ibja",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws2s1["ovov"] +=  1.000 * np.einsum("Ak,klaA,jbil->jbia",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws2s1["ovov"] += -1.000 * np.einsum("Ak,ikaB,bBjA->ibja",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
	ws2s1["ovov"] += -1.000 * np.einsum("Ak,ikcA,bcja->ibja",t1["Vo"],t2["oovV"],v["vvov"],optimize="optimal")
	ws2s1["ovov"] +=  1.000 * np.einsum("Ak,ikAB,bBja->ibja",t1["Vo"],t2["ooVV"],v["vVov"],optimize="optimal")
	ws2s1["ovov"] +=  1.000 * np.einsum("iA,aBjk,kAbB->iajb",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s1["ovov"] +=  1.000 * np.einsum("iA,cAjk,kbac->ibja",t1["oV"],t2["vVoo"],v["ovvv"],optimize="optimal")
	ws2s1["ovov"] += -1.000 * np.einsum("iA,ABjk,kbaB->ibja",t1["oV"],t2["VVoo"],v["ovvV"],optimize="optimal")
	ws2s1["ovov"] += -1.000 * np.einsum("kA,aBik,jAbB->jaib",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s1["ovov"] += -1.000 * np.einsum("kA,cAik,jbac->jbia",t1["oV"],t2["vVoo"],v["ovvv"],optimize="optimal")
	ws2s1["ovov"] +=  1.000 * np.einsum("kA,ABik,jbaB->jbia",t1["oV"],t2["VVoo"],v["ovvV"],optimize="optimal")
	ws2s1["ovov"] += -1.000 * np.einsum("kA,aAil,jlkb->jaib",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s1["ovov"] +=  1.000 * np.einsum("kA,aAkl,jlib->jaib",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s1["ovov"] += -0.500 * np.einsum("Ai,klaA,jbkl->jbia",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
	ws2s1["ovov"] += -0.500 * np.einsum("iA,aAkl,kljb->iajb",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	ws2s1["vvoo"] +=  4.000 * np.einsum("Ai,aBjk,kbAB->abij",t1["Vo"],t2["vVoo"],v["ovVV"],optimize="optimal")
	ws2s1["vvoo"] +=  4.000 * np.einsum("kA,aBik,bAjB->abij",t1["oV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws2s1["vvoo"] +=  4.000 * np.einsum("kA,aAil,lbjk->abij",t1["oV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s1["vvoo"] += -2.000 * np.einsum("kA,cAik,abjc->abij",t1["oV"],t2["vVoo"],v["vvov"],optimize="optimal")
	ws2s1["vvoo"] +=  2.000 * np.einsum("kA,ABik,abjB->abij",t1["oV"],t2["VVoo"],v["vvoV"],optimize="optimal")
	ws2s1["vvoo"] +=  2.000 * np.einsum("kA,aAkl,lbij->abij",t1["oV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s1["vvoo"] +=  2.000 * np.einsum("Ak,aBij,kbAB->abij",t1["Vo"],t2["vVoo"],v["ovVV"],optimize="optimal")
	ws2s1["vvoo"] += -2.000 * np.einsum("kA,aBij,bAkB->abij",t1["oV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws2s1["vvoo"] +=  1.000 * np.einsum("kA,cAij,abkc->abij",t1["oV"],t2["vVoo"],v["vvov"],optimize="optimal")
	ws2s1["vvoo"] += -1.000 * np.einsum("kA,ABij,abkB->abij",t1["oV"],t2["VVoo"],v["vvoV"],optimize="optimal")
	ws2s1["vvoo"] = asym_term(ws2s1["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	ws2s1["ovvv"] += -2.000 * np.einsum("Aj,ikaA,jckb->icab",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
	ws2s1["ovvv"] +=  2.000 * np.einsum("Aj,jkaA,ickb->icab",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
	ws2s1["ovvv"] +=  2.000 * np.einsum("Aj,ijaB,cBbA->icab",t1["Vo"],t2["oovV"],v["vVvV"],optimize="optimal")
	ws2s1["ovvv"] += -1.000 * np.einsum("Aj,ijdA,cdab->icab",t1["Vo"],t2["oovV"],v["vvvv"],optimize="optimal")
	ws2s1["ovvv"] +=  1.000 * np.einsum("Aj,ijAB,cBab->icab",t1["Vo"],t2["ooVV"],v["vVvv"],optimize="optimal")
	ws2s1["ovvv"] +=  1.000 * np.einsum("jA,aAjk,ikbc->iabc",t1["oV"],t2["vVoo"],v["oovv"],optimize="optimal")
	ws2s1["ovvv"] += -0.500 * np.einsum("iA,aAjk,jkbc->iabc",t1["oV"],t2["vVoo"],v["oovv"],optimize="optimal")
	ws2s1["ovvv"] = asym_term(ws2s1["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	ws2s1["vvov"] +=  2.000 * np.einsum("jA,aBij,cAbB->acib",t1["oV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s1["vvov"] += -2.000 * np.einsum("jA,aAik,kcjb->acib",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s1["vvov"] +=  2.000 * np.einsum("jA,aAjk,kcib->acib",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s1["vvov"] += -1.000 * np.einsum("jA,dAij,bcad->bcia",t1["oV"],t2["vVoo"],v["vvvv"],optimize="optimal")
	ws2s1["vvov"] +=  1.000 * np.einsum("jA,ABij,bcaB->bcia",t1["oV"],t2["VVoo"],v["vvvV"],optimize="optimal")
	ws2s1["vvov"] +=  1.000 * np.einsum("Aj,jkaA,bcik->bcia",t1["Vo"],t2["oovV"],v["vvoo"],optimize="optimal")
	ws2s1["vvov"] += -0.500 * np.einsum("Ai,jkaA,bcjk->bcia",t1["Vo"],t2["oovV"],v["vvoo"],optimize="optimal")
	ws2s1["vvov"] = asym_term(ws2s1["vvov"],"vvov")

	# vvvv = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_virt_int))
	ws2s1["vvvv"] +=  2.000 * np.einsum("Ai,ijaA,cdjb->cdab",t1["Vo"],t2["oovV"],v["vvov"],optimize="optimal")
	ws2s1["vvvv"] +=  2.000 * np.einsum("iA,aAij,jdbc->adbc",t1["oV"],t2["vVoo"],v["ovvv"],optimize="optimal")
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
		ws2s1["oooooo"] +=  1.000 * np.einsum("Ai,jkaA,nalm->jknilm",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
		ws2s1["oooooo"] += -1.000 * np.einsum("Ai,jkAB,nBlm->jknilm",t1["Vo"],t2["ooVV"],v["oVoo"],optimize="optimal")
		ws2s1["oooooo"] +=  1.000 * np.einsum("iA,aAjk,mnla->imnjkl",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s1["oooooo"] += -1.000 * np.einsum("iA,ABjk,mnlB->imnjkl",t1["oV"],t2["VVoo"],v["oooV"],optimize="optimal")
		ws2s1["oooooo"] = asym_term(ws2s1["oooooo"],"oooooo")

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		ws2s1["ooooov"] +=  (2./3.) * np.einsum("Ai,jnaA,lmkn->jlmika",t1["Vo"],t2["oovV"],v["oooo"],optimize="optimal")
		ws2s1["ooooov"] +=  (2./3.) * np.einsum("Ai,jkaB,mBlA->jkmila",t1["Vo"],t2["oovV"],v["oVoV"],optimize="optimal")
		ws2s1["ooooov"] +=  (2./3.) * np.einsum("Ai,jkbA,mbla->jkmila",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
		ws2s1["ooooov"] += -(2./3.) * np.einsum("Ai,jkAB,mBla->jkmila",t1["Vo"],t2["ooVV"],v["oVov"],optimize="optimal")
		ws2s1["ooooov"] +=  (1./3.) * np.einsum("iA,bAjk,lmab->ilmjka",t1["oV"],t2["vVoo"],v["oovv"],optimize="optimal")
		ws2s1["ooooov"] += -(1./3.) * np.einsum("iA,ABjk,lmaB->ilmjka",t1["oV"],t2["VVoo"],v["oovV"],optimize="optimal")
		ws2s1["ooooov"] += -(1./3.) * np.einsum("iA,jkaB,ABlm->ijklma",t1["oV"],t2["oovV"],v["VVoo"],optimize="optimal")
		ws2s1["ooooov"] = asym_term(ws2s1["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		ws2s1["oovooo"] +=  (2./3.) * np.einsum("iA,aBjk,mAlB->imajkl",t1["oV"],t2["vVoo"],v["oVoV"],optimize="optimal")
		ws2s1["oovooo"] +=  (2./3.) * np.einsum("iA,bAjk,malb->imajkl",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s1["oovooo"] += -(2./3.) * np.einsum("iA,ABjk,malB->imajkl",t1["oV"],t2["VVoo"],v["ovoV"],optimize="optimal")
		ws2s1["oovooo"] +=  (2./3.) * np.einsum("iA,aAjn,mnkl->imajkl",t1["oV"],t2["vVoo"],v["oooo"],optimize="optimal")
		ws2s1["oovooo"] += -(1./3.) * np.einsum("Ai,aBjk,lmAB->lmaijk",t1["Vo"],t2["vVoo"],v["ooVV"],optimize="optimal")
		ws2s1["oovooo"] +=  (1./3.) * np.einsum("Ai,jkbA,ablm->jkailm",t1["Vo"],t2["oovV"],v["vvoo"],optimize="optimal")
		ws2s1["oovooo"] += -(1./3.) * np.einsum("Ai,jkAB,aBlm->jkailm",t1["Vo"],t2["ooVV"],v["vVoo"],optimize="optimal")
		ws2s1["oovooo"] = asym_term(ws2s1["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s1["oooovv"] +=  (2./3.) * np.einsum("Ai,jmaA,klmb->jkliab",t1["Vo"],t2["oovV"],v["ooov"],optimize="optimal")
		ws2s1["oooovv"] += -(2./3.) * np.einsum("Ai,jkaB,lBbA->jkliab",t1["Vo"],t2["oovV"],v["oVvV"],optimize="optimal")
		ws2s1["oooovv"] +=  (2./3.) * np.einsum("iA,jkaB,ABlb->ijklab",t1["oV"],t2["oovV"],v["VVov"],optimize="optimal")
		ws2s1["oooovv"] +=  (1./3.) * np.einsum("Ai,jkcA,lcab->jkliab",t1["Vo"],t2["oovV"],v["ovvv"],optimize="optimal")
		ws2s1["oooovv"] += -(1./3.) * np.einsum("Ai,jkAB,lBab->jkliab",t1["Vo"],t2["ooVV"],v["oVvv"],optimize="optimal")
		ws2s1["oooovv"] = asym_term(ws2s1["oooovv"],"oooovv")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s1["oovoov"] +=  (4./9.) * np.einsum("Ai,jmaA,lbkm->jlbika",t1["Vo"],t2["oovV"],v["ovoo"],optimize="optimal")
		ws2s1["oovoov"] +=  (4./9.) * np.einsum("iA,aAjm,lmkb->ilajkb",t1["oV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s1["oovoov"] +=  (2./9.) * np.einsum("Ai,jkaB,bBlA->jkbila",t1["Vo"],t2["oovV"],v["vVoV"],optimize="optimal")
		ws2s1["oovoov"] +=  (2./9.) * np.einsum("Ai,jkcA,bcla->jkbila",t1["Vo"],t2["oovV"],v["vvov"],optimize="optimal")
		ws2s1["oovoov"] += -(2./9.) * np.einsum("Ai,jkAB,bBla->jkbila",t1["Vo"],t2["ooVV"],v["vVov"],optimize="optimal")
		ws2s1["oovoov"] +=  (2./9.) * np.einsum("iA,aBjk,lAbB->ilajkb",t1["oV"],t2["vVoo"],v["oVvV"],optimize="optimal")
		ws2s1["oovoov"] +=  (2./9.) * np.einsum("iA,cAjk,lbac->ilbjka",t1["oV"],t2["vVoo"],v["ovvv"],optimize="optimal")
		ws2s1["oovoov"] += -(2./9.) * np.einsum("iA,ABjk,lbaB->ilbjka",t1["oV"],t2["VVoo"],v["ovvV"],optimize="optimal")
		ws2s1["oovoov"] = asym_term(ws2s1["oovoov"],"oovoov")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2s1["ovvooo"] +=  (2./3.) * np.einsum("Ai,aBjk,lbAB->labijk",t1["Vo"],t2["vVoo"],v["ovVV"],optimize="optimal")
		ws2s1["ovvooo"] += -(2./3.) * np.einsum("iA,aBjk,bAlB->iabjkl",t1["oV"],t2["vVoo"],v["vVoV"],optimize="optimal")
		ws2s1["ovvooo"] +=  (2./3.) * np.einsum("iA,aAjm,mbkl->iabjkl",t1["oV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s1["ovvooo"] +=  (1./3.) * np.einsum("iA,cAjk,ablc->iabjkl",t1["oV"],t2["vVoo"],v["vvov"],optimize="optimal")
		ws2s1["ovvooo"] += -(1./3.) * np.einsum("iA,ABjk,ablB->iabjkl",t1["oV"],t2["VVoo"],v["vvoV"],optimize="optimal")
		ws2s1["ovvooo"] = asym_term(ws2s1["ovvooo"],"ovvooo")

		# ooovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s1["ooovvv"] += -1.000 * np.einsum("iA,jkaB,ABbc->ijkabc",t1["oV"],t2["oovV"],v["VVvv"],optimize="optimal")
		ws2s1["ooovvv"] = asym_term(ws2s1["ooovvv"],"ooovvv")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s1["oovovv"] +=  (4./9.) * np.einsum("Ai,jlaA,kclb->jkciab",t1["Vo"],t2["oovV"],v["ovov"],optimize="optimal")
		ws2s1["oovovv"] += -(2./9.) * np.einsum("Ai,jkaB,cBbA->jkciab",t1["Vo"],t2["oovV"],v["vVvV"],optimize="optimal")
		ws2s1["oovovv"] +=  (2./9.) * np.einsum("iA,aAjl,klbc->ikajbc",t1["oV"],t2["vVoo"],v["oovv"],optimize="optimal")
		ws2s1["oovovv"] +=  (1./9.) * np.einsum("Ai,jkdA,cdab->jkciab",t1["Vo"],t2["oovV"],v["vvvv"],optimize="optimal")
		ws2s1["oovovv"] += -(1./9.) * np.einsum("Ai,jkAB,cBab->jkciab",t1["Vo"],t2["ooVV"],v["vVvv"],optimize="optimal")
		ws2s1["oovovv"] = asym_term(ws2s1["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s1["ovvoov"] +=  (4./9.) * np.einsum("iA,aAjl,lckb->iacjkb",t1["oV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s1["ovvoov"] +=  (2./9.) * np.einsum("Ai,jlaA,bckl->jbcika",t1["Vo"],t2["oovV"],v["vvoo"],optimize="optimal")
		ws2s1["ovvoov"] += -(2./9.) * np.einsum("iA,aBjk,cAbB->iacjkb",t1["oV"],t2["vVoo"],v["vVvV"],optimize="optimal")
		ws2s1["ovvoov"] +=  (1./9.) * np.einsum("iA,dAjk,bcad->ibcjka",t1["oV"],t2["vVoo"],v["vvvv"],optimize="optimal")
		ws2s1["ovvoov"] += -(1./9.) * np.einsum("iA,ABjk,bcaB->ibcjka",t1["oV"],t2["VVoo"],v["vvvV"],optimize="optimal")
		ws2s1["ovvoov"] = asym_term(ws2s1["ovvoov"],"ovvoov")

		# vvvooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2s1["vvvooo"] += -1.000 * np.einsum("Ai,aBjk,bcAB->abcijk",t1["Vo"],t2["vVoo"],v["vvVV"],optimize="optimal")
		ws2s1["vvvooo"] = asym_term(ws2s1["vvvooo"],"vvvooo")

		# ovvovv = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s1["ovvovv"] +=  (2./9.) * np.einsum("Ai,jkaA,cdkb->jcdiab",t1["Vo"],t2["oovV"],v["vvov"],optimize="optimal")
		ws2s1["ovvovv"] +=  (2./9.) * np.einsum("iA,aAjk,kdbc->iadjbc",t1["oV"],t2["vVoo"],v["ovvv"],optimize="optimal")
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
	ws2s2["c"] +=  8.000 * np.einsum("ijAB,aAik,kBja->",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
	ws2s2["c"] += -8.000 * np.einsum("ijAB,ACik,kBjC->",t2["ooVV"],t2["VVoo"],v["oVoV"],optimize="optimal")
	ws2s2["c"] += -8.000 * np.einsum("ijaA,aBik,kAjB->",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["c"] += -8.000 * np.einsum("ijaA,bAik,kajb->",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["c"] +=  8.000 * np.einsum("ijaA,ABik,kajB->",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["c"] +=  4.000 * np.einsum("ijaA,bBij,aAbB->",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s2["c"] +=  2.000 * np.einsum("ijaA,aAkl,klij->",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["c"] +=  2.000 * np.einsum("ijAB,aCij,ABaC->",t2["ooVV"],t2["vVoo"],v["VVvV"],optimize="optimal")
	ws2s2["c"] +=  2.000 * np.einsum("ijaA,BCij,aABC->",t2["oovV"],t2["VVoo"],v["vVVV"],optimize="optimal")
	ws2s2["c"] +=  1.000 * np.einsum("ijAB,CDij,ABCD->",t2["ooVV"],t2["VVoo"],v["VVVV"],optimize="optimal")
	ws2s2["c"] +=  1.000 * np.einsum("ijAB,ABkl,klij->",t2["ooVV"],t2["VVoo"],v["oooo"],optimize="optimal")

	ws2s2["oo"] += -8.000 * np.einsum("klaA,aAkm,jmil->ji",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["oo"] +=  8.000 * np.einsum("klaA,bAik,jalb->ji",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["oo"] += -8.000 * np.einsum("ikaA,bAjl,lakb->ij",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["oo"] +=  8.000 * np.einsum("ikaA,bAkl,lajb->ij",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["oo"] +=  8.000 * np.einsum("ikaA,bBjk,aAbB->ij",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s2["oo"] += -8.000 * np.einsum("ikaA,aBjl,lAkB->ij",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oo"] +=  8.000 * np.einsum("ikaA,ABjl,lakB->ij",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["oo"] +=  8.000 * np.einsum("ikaA,aBkl,lAjB->ij",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oo"] += -8.000 * np.einsum("ikaA,ABkl,lajB->ij",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["oo"] +=  8.000 * np.einsum("klaA,aBik,jAlB->ji",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oo"] += -8.000 * np.einsum("klaA,ABik,jalB->ji",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["oo"] +=  8.000 * np.einsum("ikAB,aAjl,lBka->ij",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
	ws2s2["oo"] += -8.000 * np.einsum("ikAB,ACjl,lBkC->ij",t2["ooVV"],t2["VVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oo"] += -8.000 * np.einsum("ikAB,aAkl,lBja->ij",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
	ws2s2["oo"] +=  8.000 * np.einsum("ikAB,ACkl,lBjC->ij",t2["ooVV"],t2["VVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oo"] += -8.000 * np.einsum("klAB,aAik,jBla->ji",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
	ws2s2["oo"] +=  8.000 * np.einsum("klAB,ACik,jBlC->ji",t2["ooVV"],t2["VVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oo"] +=  4.000 * np.einsum("ikAB,aCjk,ABaC->ij",t2["ooVV"],t2["vVoo"],v["VVvV"],optimize="optimal")
	ws2s2["oo"] +=  4.000 * np.einsum("klaA,aAim,jmkl->ji",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["oo"] +=  4.000 * np.einsum("klaA,aBkl,jAiB->ji",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oo"] +=  4.000 * np.einsum("klaA,bAkl,jaib->ji",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["oo"] += -4.000 * np.einsum("klaA,ABkl,jaiB->ji",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["oo"] +=  4.000 * np.einsum("ikaA,aAlm,lmjk->ij",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["oo"] +=  4.000 * np.einsum("ikaA,BCjk,aABC->ij",t2["oovV"],t2["VVoo"],v["vVVV"],optimize="optimal")
	ws2s2["oo"] += -4.000 * np.einsum("klAB,aAkl,jBia->ji",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
	ws2s2["oo"] +=  4.000 * np.einsum("klAB,ACkl,jBiC->ji",t2["ooVV"],t2["VVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oo"] += -4.000 * np.einsum("klAB,ABkm,jmil->ji",t2["ooVV"],t2["VVoo"],v["oooo"],optimize="optimal")
	ws2s2["oo"] +=  2.000 * np.einsum("ikAB,ABlm,lmjk->ij",t2["ooVV"],t2["VVoo"],v["oooo"],optimize="optimal")
	ws2s2["oo"] +=  2.000 * np.einsum("klAB,ABim,jmkl->ji",t2["ooVV"],t2["VVoo"],v["oooo"],optimize="optimal")
	ws2s2["oo"] +=  2.000 * np.einsum("ikAB,CDjk,ABCD->ij",t2["ooVV"],t2["VVoo"],v["VVVV"],optimize="optimal")

	ws2s2["ov"] += -8.000 * np.einsum("ijaA,bBjk,kAbB->ia",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ov"] += -4.000 * np.einsum("ijaA,BCjk,kABC->ia",t2["oovV"],t2["VVoo"],v["oVVV"],optimize="optimal")
	ws2s2["ov"] +=  4.000 * np.einsum("ijaA,bAkl,kljb->ia",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ov"] += -4.000 * np.einsum("ijaA,ABkl,kljB->ia",t2["oovV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws2s2["ov"] += -4.000 * np.einsum("jkaA,bBjk,iAbB->ia",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ov"] += -2.000 * np.einsum("jkaA,BCjk,iABC->ia",t2["oovV"],t2["VVoo"],v["oVVV"],optimize="optimal")
	ws2s2["ov"] += -8.000 * np.einsum("jkaA,bAjl,ilkb->ia",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ov"] +=  8.000 * np.einsum("jkaA,ABjl,ilkB->ia",t2["oovV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws2s2["ov"] +=  8.000 * np.einsum("ijbA,bBjk,kAaB->ia",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ov"] +=  8.000 * np.einsum("ijbA,cAjk,kbac->ia",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
	ws2s2["ov"] += -8.000 * np.einsum("ijbA,ABjk,kbaB->ia",t2["oovV"],t2["VVoo"],v["ovvV"],optimize="optimal")
	ws2s2["ov"] += -4.000 * np.einsum("ijbA,bAkl,klja->ia",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ov"] +=  4.000 * np.einsum("jkbA,bBjk,iAaB->ia",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ov"] +=  4.000 * np.einsum("jkbA,cAjk,ibac->ia",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
	ws2s2["ov"] += -4.000 * np.einsum("jkbA,ABjk,ibaB->ia",t2["oovV"],t2["VVoo"],v["ovvV"],optimize="optimal")
	ws2s2["ov"] +=  8.000 * np.einsum("jkbA,bAjl,ilka->ia",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ov"] += -8.000 * np.einsum("ijAB,bAjk,kBab->ia",t2["ooVV"],t2["vVoo"],v["oVvv"],optimize="optimal")
	ws2s2["ov"] +=  8.000 * np.einsum("ijAB,ACjk,kBaC->ia",t2["ooVV"],t2["VVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ov"] += -2.000 * np.einsum("ijAB,ABkl,klja->ia",t2["ooVV"],t2["VVoo"],v["ooov"],optimize="optimal")
	ws2s2["ov"] += -4.000 * np.einsum("jkAB,bAjk,iBab->ia",t2["ooVV"],t2["vVoo"],v["oVvv"],optimize="optimal")
	ws2s2["ov"] +=  4.000 * np.einsum("jkAB,ACjk,iBaC->ia",t2["ooVV"],t2["VVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ov"] +=  4.000 * np.einsum("jkAB,ABjl,ilka->ia",t2["ooVV"],t2["VVoo"],v["ooov"],optimize="optimal")

	ws2s2["vo"] += -8.000 * np.einsum("jkbA,aBij,bAkB->ai",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws2s2["vo"] +=  8.000 * np.einsum("jkbA,bBij,aAkB->ai",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws2s2["vo"] +=  8.000 * np.einsum("jkbA,cAij,abkc->ai",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
	ws2s2["vo"] += -8.000 * np.einsum("jkbA,ABij,abkB->ai",t2["oovV"],t2["VVoo"],v["vvoV"],optimize="optimal")
	ws2s2["vo"] +=  4.000 * np.einsum("jkbA,aAil,lbjk->ai",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["vo"] += -4.000 * np.einsum("jkbA,bAil,lajk->ai",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["vo"] += -4.000 * np.einsum("jkbA,aBjk,bAiB->ai",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws2s2["vo"] +=  4.000 * np.einsum("jkbA,bBjk,aAiB->ai",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws2s2["vo"] +=  4.000 * np.einsum("jkbA,cAjk,abic->ai",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
	ws2s2["vo"] += -4.000 * np.einsum("jkbA,ABjk,abiB->ai",t2["oovV"],t2["VVoo"],v["vvoV"],optimize="optimal")
	ws2s2["vo"] += -8.000 * np.einsum("jkbA,aAjl,lbik->ai",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["vo"] +=  8.000 * np.einsum("jkbA,bAjl,laik->ai",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["vo"] += -4.000 * np.einsum("jkAB,aCij,ABkC->ai",t2["ooVV"],t2["vVoo"],v["VVoV"],optimize="optimal")
	ws2s2["vo"] += -8.000 * np.einsum("jkAB,bAij,aBkb->ai",t2["ooVV"],t2["vVoo"],v["vVov"],optimize="optimal")
	ws2s2["vo"] +=  8.000 * np.einsum("jkAB,ACij,aBkC->ai",t2["ooVV"],t2["VVoo"],v["vVoV"],optimize="optimal")
	ws2s2["vo"] += -4.000 * np.einsum("jkAB,aAil,lBjk->ai",t2["ooVV"],t2["vVoo"],v["oVoo"],optimize="optimal")
	ws2s2["vo"] += -2.000 * np.einsum("jkAB,ABil,lajk->ai",t2["ooVV"],t2["VVoo"],v["ovoo"],optimize="optimal")
	ws2s2["vo"] += -2.000 * np.einsum("jkAB,aCjk,ABiC->ai",t2["ooVV"],t2["vVoo"],v["VVoV"],optimize="optimal")
	ws2s2["vo"] += -4.000 * np.einsum("jkAB,bAjk,aBib->ai",t2["ooVV"],t2["vVoo"],v["vVov"],optimize="optimal")
	ws2s2["vo"] +=  4.000 * np.einsum("jkAB,ACjk,aBiC->ai",t2["ooVV"],t2["VVoo"],v["vVoV"],optimize="optimal")
	ws2s2["vo"] +=  8.000 * np.einsum("jkAB,aAjl,lBik->ai",t2["ooVV"],t2["vVoo"],v["oVoo"],optimize="optimal")
	ws2s2["vo"] +=  4.000 * np.einsum("jkAB,ABjl,laik->ai",t2["ooVV"],t2["VVoo"],v["ovoo"],optimize="optimal")

	ws2s2["vv"] += -4.000 * np.einsum("ijaA,cBij,bAcB->ba",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s2["vv"] += -2.000 * np.einsum("ijaA,BCij,bABC->ba",t2["oovV"],t2["VVoo"],v["vVVV"],optimize="optimal")
	ws2s2["vv"] +=  8.000 * np.einsum("ijaA,bBik,kAjB->ba",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["vv"] +=  8.000 * np.einsum("ijaA,cAik,kbjc->ba",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["vv"] += -8.000 * np.einsum("ijaA,ABik,kbjB->ba",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["vv"] += -2.000 * np.einsum("ijaA,bAkl,klij->ba",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["vv"] += -4.000 * np.einsum("ijcA,aBij,cAbB->ab",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s2["vv"] +=  4.000 * np.einsum("ijcA,cBij,bAaB->ba",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s2["vv"] +=  4.000 * np.einsum("ijcA,dAij,bcad->ba",t2["oovV"],t2["vVoo"],v["vvvv"],optimize="optimal")
	ws2s2["vv"] += -4.000 * np.einsum("ijcA,ABij,bcaB->ba",t2["oovV"],t2["VVoo"],v["vvvV"],optimize="optimal")
	ws2s2["vv"] +=  8.000 * np.einsum("ijcA,aAik,kcjb->ab",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["vv"] += -8.000 * np.einsum("ijcA,cAik,kbja->ba",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["vv"] += -2.000 * np.einsum("ijAB,aCij,ABbC->ab",t2["ooVV"],t2["vVoo"],v["VVvV"],optimize="optimal")
	ws2s2["vv"] += -4.000 * np.einsum("ijAB,cAij,bBac->ba",t2["ooVV"],t2["vVoo"],v["vVvv"],optimize="optimal")
	ws2s2["vv"] +=  4.000 * np.einsum("ijAB,ACij,bBaC->ba",t2["ooVV"],t2["VVoo"],v["vVvV"],optimize="optimal")
	ws2s2["vv"] += -8.000 * np.einsum("ijAB,aAik,kBjb->ab",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
	ws2s2["vv"] += -4.000 * np.einsum("ijAB,ABik,kbja->ba",t2["ooVV"],t2["VVoo"],v["ovov"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	ws2s2["oooo"] +=  8.0000 * np.einsum("imaA,aBjm,lAkB->iljk",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oooo"] +=  8.0000 * np.einsum("imaA,bAjm,lakb->iljk",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["oooo"] += -8.0000 * np.einsum("imaA,ABjm,lakB->iljk",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["oooo"] += -8.0000 * np.einsum("imaA,aAjn,lnkm->iljk",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["oooo"] += -8.0000 * np.einsum("imAB,aAjm,lBka->iljk",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
	ws2s2["oooo"] +=  8.0000 * np.einsum("imAB,ACjm,lBkC->iljk",t2["ooVV"],t2["VVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oooo"] += -4.0000 * np.einsum("ijaA,aBkm,mAlB->ijkl",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oooo"] += -4.0000 * np.einsum("ijaA,bAkm,malb->ijkl",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["oooo"] +=  4.0000 * np.einsum("ijaA,ABkm,malB->ijkl",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["oooo"] += -4.0000 * np.einsum("imaA,aBjk,lAmB->iljk",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oooo"] += -4.0000 * np.einsum("imaA,bAjk,lamb->iljk",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["oooo"] +=  4.0000 * np.einsum("imaA,ABjk,lamB->iljk",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["oooo"] +=  4.0000 * np.einsum("ijAB,aAkm,mBla->ijkl",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
	ws2s2["oooo"] += -4.0000 * np.einsum("ijAB,ACkm,mBlC->ijkl",t2["ooVV"],t2["VVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oooo"] +=  4.0000 * np.einsum("imAB,aAjk,lBma->iljk",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
	ws2s2["oooo"] += -4.0000 * np.einsum("imAB,ACjk,lBmC->iljk",t2["ooVV"],t2["VVoo"],v["oVoV"],optimize="optimal")
	ws2s2["oooo"] += -4.0000 * np.einsum("imAB,ABjn,lnkm->iljk",t2["ooVV"],t2["VVoo"],v["oooo"],optimize="optimal")
	ws2s2["oooo"] +=  2.0000 * np.einsum("ijaA,bBkl,aAbB->ijkl",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s2["oooo"] += -2.0000 * np.einsum("imaA,aAmn,lnjk->iljk",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["oooo"] += -2.0000 * np.einsum("mnaA,aAim,kljn->klij",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["oooo"] +=  1.0000 * np.einsum("ijAB,aCkl,ABaC->ijkl",t2["ooVV"],t2["vVoo"],v["VVvV"],optimize="optimal")
	ws2s2["oooo"] +=  1.0000 * np.einsum("ijaA,BCkl,aABC->ijkl",t2["oovV"],t2["VVoo"],v["vVVV"],optimize="optimal")
	ws2s2["oooo"] += -1.0000 * np.einsum("imAB,ABmn,lnjk->iljk",t2["ooVV"],t2["VVoo"],v["oooo"],optimize="optimal")
	ws2s2["oooo"] += -1.0000 * np.einsum("mnAB,ABim,kljn->klij",t2["ooVV"],t2["VVoo"],v["oooo"],optimize="optimal")
	ws2s2["oooo"] +=  0.5000 * np.einsum("mnaA,aAij,klmn->klij",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["oooo"] +=  0.5000 * np.einsum("ijAB,CDkl,ABCD->ijkl",t2["ooVV"],t2["VVoo"],v["VVVV"],optimize="optimal")
	ws2s2["oooo"] +=  0.5000 * np.einsum("ijaA,aAmn,mnkl->ijkl",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["oooo"] +=  0.2500 * np.einsum("mnAB,ABij,klmn->klij",t2["ooVV"],t2["VVoo"],v["oooo"],optimize="optimal")
	ws2s2["oooo"] +=  0.2500 * np.einsum("ijAB,ABmn,mnkl->ijkl",t2["ooVV"],t2["VVoo"],v["oooo"],optimize="optimal")
	ws2s2["oooo"] = asym_term(ws2s2["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	ws2s2["ooov"] += -4.0000 * np.einsum("ilaA,bBjl,kAbB->ikja",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ooov"] += -4.0000 * np.einsum("ilaA,bAjm,kmlb->ikja",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] +=  4.0000 * np.einsum("ilaA,ABjm,kmlB->ikja",t2["oovV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws2s2["ooov"] +=  4.0000 * np.einsum("ilbA,bBjl,kAaB->ikja",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ooov"] +=  4.0000 * np.einsum("ilbA,cAjl,kbac->ikja",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
	ws2s2["ooov"] += -4.0000 * np.einsum("ilbA,ABjl,kbaB->ikja",t2["oovV"],t2["VVoo"],v["ovvV"],optimize="optimal")
	ws2s2["ooov"] +=  4.0000 * np.einsum("ilbA,bAjm,kmla->ikja",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] += -4.0000 * np.einsum("ilAB,bAjl,kBab->ikja",t2["ooVV"],t2["vVoo"],v["oVvv"],optimize="optimal")
	ws2s2["ooov"] +=  4.0000 * np.einsum("ilAB,ACjl,kBaC->ikja",t2["ooVV"],t2["VVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ooov"] +=  2.0000 * np.einsum("ijaA,bBkl,lAbB->ijka",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ooov"] += -2.0000 * np.einsum("ilaA,BCjl,kABC->ikja",t2["oovV"],t2["VVoo"],v["oVVV"],optimize="optimal")
	ws2s2["ooov"] +=  2.0000 * np.einsum("ilaA,bAlm,kmjb->ikja",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] += -2.0000 * np.einsum("ilaA,ABlm,kmjB->ikja",t2["oovV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws2s2["ooov"] += -2.0000 * np.einsum("lmaA,bAil,jkmb->jkia",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] +=  2.0000 * np.einsum("lmaA,ABil,jkmB->jkia",t2["oovV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws2s2["ooov"] += -2.0000 * np.einsum("ijbA,bBkl,lAaB->ijka",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ooov"] += -2.0000 * np.einsum("ijbA,cAkl,lbac->ijka",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
	ws2s2["ooov"] +=  2.0000 * np.einsum("ijbA,ABkl,lbaB->ijka",t2["oovV"],t2["VVoo"],v["ovvV"],optimize="optimal")
	ws2s2["ooov"] += -2.0000 * np.einsum("ilbA,bAlm,kmja->ikja",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] +=  2.0000 * np.einsum("ijAB,bAkl,lBab->ijka",t2["ooVV"],t2["vVoo"],v["oVvv"],optimize="optimal")
	ws2s2["ooov"] += -2.0000 * np.einsum("ijAB,ACkl,lBaC->ijka",t2["ooVV"],t2["VVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ooov"] +=  2.0000 * np.einsum("ilAB,ABjm,kmla->ikja",t2["ooVV"],t2["VVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] += -1.0000 * np.einsum("ilAB,ABlm,kmja->ikja",t2["ooVV"],t2["VVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] +=  1.0000 * np.einsum("lmbA,bAil,jkma->jkia",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] +=  1.0000 * np.einsum("ijaA,BCkl,lABC->ijka",t2["oovV"],t2["VVoo"],v["oVVV"],optimize="optimal")
	ws2s2["ooov"] += -0.5000 * np.einsum("ijaA,bAlm,lmkb->ijka",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] +=  0.5000 * np.einsum("ijaA,ABlm,lmkB->ijka",t2["oovV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws2s2["ooov"] += -0.5000 * np.einsum("lmaA,bAlm,jkib->jkia",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] +=  0.5000 * np.einsum("lmaA,ABlm,jkiB->jkia",t2["oovV"],t2["VVoo"],v["oooV"],optimize="optimal")
	ws2s2["ooov"] +=  0.5000 * np.einsum("ijbA,bAlm,lmka->ijka",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] +=  0.5000 * np.einsum("lmAB,ABil,jkma->jkia",t2["ooVV"],t2["VVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] +=  0.2500 * np.einsum("ijAB,ABlm,lmka->ijka",t2["ooVV"],t2["VVoo"],v["ooov"],optimize="optimal")
	ws2s2["ooov"] = asym_term(ws2s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	ws2s2["ovoo"] += -4.0000 * np.einsum("ilbA,aBjl,bAkB->iajk",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws2s2["ovoo"] +=  4.0000 * np.einsum("ilbA,bBjl,aAkB->iajk",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws2s2["ovoo"] +=  4.0000 * np.einsum("ilbA,cAjl,abkc->iajk",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
	ws2s2["ovoo"] += -4.0000 * np.einsum("ilbA,ABjl,abkB->iajk",t2["oovV"],t2["VVoo"],v["vvoV"],optimize="optimal")
	ws2s2["ovoo"] += -4.0000 * np.einsum("ilbA,aAjm,mbkl->iajk",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] +=  4.0000 * np.einsum("ilbA,bAjm,makl->iajk",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] += -4.0000 * np.einsum("ilAB,bAjl,aBkb->iajk",t2["ooVV"],t2["vVoo"],v["vVov"],optimize="optimal")
	ws2s2["ovoo"] +=  4.0000 * np.einsum("ilAB,ACjl,aBkC->iajk",t2["ooVV"],t2["VVoo"],v["vVoV"],optimize="optimal")
	ws2s2["ovoo"] +=  4.0000 * np.einsum("ilAB,aAjm,mBkl->iajk",t2["ooVV"],t2["vVoo"],v["oVoo"],optimize="optimal")
	ws2s2["ovoo"] +=  2.0000 * np.einsum("ilbA,aBjk,bAlB->iajk",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws2s2["ovoo"] += -2.0000 * np.einsum("ilbA,bBjk,aAlB->iajk",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws2s2["ovoo"] += -2.0000 * np.einsum("ilbA,cAjk,ablc->iajk",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
	ws2s2["ovoo"] +=  2.0000 * np.einsum("ilbA,ABjk,ablB->iajk",t2["oovV"],t2["VVoo"],v["vvoV"],optimize="optimal")
	ws2s2["ovoo"] += -2.0000 * np.einsum("ilbA,aAlm,mbjk->iajk",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] +=  2.0000 * np.einsum("lmbA,aAil,kbjm->kaij",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] += -2.0000 * np.einsum("lmbA,bAil,kajm->kaij",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] +=  2.0000 * np.einsum("ilAB,bAjk,aBlb->iajk",t2["ooVV"],t2["vVoo"],v["vVov"],optimize="optimal")
	ws2s2["ovoo"] += -2.0000 * np.einsum("ilAB,ACjk,aBlC->iajk",t2["ooVV"],t2["VVoo"],v["vVoV"],optimize="optimal")
	ws2s2["ovoo"] += -2.0000 * np.einsum("ilAB,aCjl,ABkC->iajk",t2["ooVV"],t2["vVoo"],v["VVoV"],optimize="optimal")
	ws2s2["ovoo"] +=  2.0000 * np.einsum("ilAB,ABjm,makl->iajk",t2["ooVV"],t2["VVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] +=  2.0000 * np.einsum("ilAB,aAlm,mBjk->iajk",t2["ooVV"],t2["vVoo"],v["oVoo"],optimize="optimal")
	ws2s2["ovoo"] += -2.0000 * np.einsum("lmAB,aAil,kBjm->kaij",t2["ooVV"],t2["vVoo"],v["oVoo"],optimize="optimal")
	ws2s2["ovoo"] +=  1.0000 * np.einsum("ilbA,bAlm,majk->iajk",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] +=  1.0000 * np.einsum("ilAB,aCjk,ABlC->iajk",t2["ooVV"],t2["vVoo"],v["VVoV"],optimize="optimal")
	ws2s2["ovoo"] += -1.0000 * np.einsum("lmAB,ABil,kajm->kaij",t2["ooVV"],t2["VVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] += -0.5000 * np.einsum("lmbA,aAij,kblm->kaij",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] +=  0.5000 * np.einsum("lmbA,bAij,kalm->kaij",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] += -0.5000 * np.einsum("lmbA,aAlm,kbij->kaij",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] +=  0.5000 * np.einsum("ilAB,ABlm,majk->iajk",t2["ooVV"],t2["VVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] +=  0.5000 * np.einsum("lmAB,aAij,kBlm->kaij",t2["ooVV"],t2["vVoo"],v["oVoo"],optimize="optimal")
	ws2s2["ovoo"] +=  0.5000 * np.einsum("lmAB,aAlm,kBij->kaij",t2["ooVV"],t2["vVoo"],v["oVoo"],optimize="optimal")
	ws2s2["ovoo"] +=  0.2500 * np.einsum("lmAB,ABij,kalm->kaij",t2["ooVV"],t2["VVoo"],v["ovoo"],optimize="optimal")
	ws2s2["ovoo"] = asym_term(ws2s2["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	ws2s2["oovv"] += -4.0000 * np.einsum("ikaA,cAkl,jlbc->ijab",t2["oovV"],t2["vVoo"],v["oovv"],optimize="optimal")
	ws2s2["oovv"] +=  4.0000 * np.einsum("ikaA,ABkl,jlbB->ijab",t2["oovV"],t2["VVoo"],v["oovV"],optimize="optimal")
	ws2s2["oovv"] +=  4.0000 * np.einsum("ikaA,jlbB,ABkl->ijab",t2["oovV"],t2["oovV"],v["VVoo"],optimize="optimal")
	ws2s2["oovv"] += -2.0000 * np.einsum("ijaA,klbB,ABkl->ijab",t2["oovV"],t2["oovV"],v["VVoo"],optimize="optimal")
	ws2s2["oovv"] += -2.0000 * np.einsum("ikcA,cAkl,jlab->ijab",t2["oovV"],t2["vVoo"],v["oovv"],optimize="optimal")
	ws2s2["oovv"] +=  1.0000 * np.einsum("ijaA,cAkl,klbc->ijab",t2["oovV"],t2["vVoo"],v["oovv"],optimize="optimal")
	ws2s2["oovv"] += -1.0000 * np.einsum("ijaA,ABkl,klbB->ijab",t2["oovV"],t2["VVoo"],v["oovV"],optimize="optimal")
	ws2s2["oovv"] +=  1.0000 * np.einsum("klaA,cAkl,ijbc->ijab",t2["oovV"],t2["vVoo"],v["oovv"],optimize="optimal")
	ws2s2["oovv"] += -1.0000 * np.einsum("klaA,ABkl,ijbB->ijab",t2["oovV"],t2["VVoo"],v["oovV"],optimize="optimal")
	ws2s2["oovv"] += -1.0000 * np.einsum("ikAB,ABkl,jlab->ijab",t2["ooVV"],t2["VVoo"],v["oovv"],optimize="optimal")
	ws2s2["oovv"] +=  0.5000 * np.einsum("ijcA,cAkl,klab->ijab",t2["oovV"],t2["vVoo"],v["oovv"],optimize="optimal")
	ws2s2["oovv"] +=  0.2500 * np.einsum("ijAB,ABkl,klab->ijab",t2["ooVV"],t2["VVoo"],v["oovv"],optimize="optimal")
	ws2s2["oovv"] = asym_term(ws2s2["oovv"],"oovv")

	ws2s2["ovov"] += -2.0000 * np.einsum("ikaA,cBjk,bAcB->ibja",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s2["ovov"] +=  2.0000 * np.einsum("ikaA,bBjl,lAkB->ibja",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["ovov"] +=  2.0000 * np.einsum("ikaA,cAjl,lbkc->ibja",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] += -2.0000 * np.einsum("ikaA,ABjl,lbkB->ibja",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["ovov"] += -2.0000 * np.einsum("ikaA,bBkl,lAjB->ibja",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["ovov"] += -2.0000 * np.einsum("klaA,bBik,jAlB->jbia",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["ovov"] += -2.0000 * np.einsum("klaA,cAik,jblc->jbia",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] +=  2.0000 * np.einsum("klaA,ABik,jblB->jbia",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["ovov"] +=  2.0000 * np.einsum("klaA,bAkm,jmil->jbia",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["ovov"] += -2.0000 * np.einsum("ikcA,aBjk,cAbB->iajb",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s2["ovov"] +=  2.0000 * np.einsum("ikcA,cBjk,bAaB->ibja",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s2["ovov"] +=  2.0000 * np.einsum("ikcA,dAjk,bcad->ibja",t2["oovV"],t2["vVoo"],v["vvvv"],optimize="optimal")
	ws2s2["ovov"] += -2.0000 * np.einsum("ikcA,ABjk,bcaB->ibja",t2["oovV"],t2["VVoo"],v["vvvV"],optimize="optimal")
	ws2s2["ovov"] +=  2.0000 * np.einsum("ikcA,aAjl,lckb->iajb",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] += -2.0000 * np.einsum("ikcA,cAjl,lbka->ibja",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] += -2.0000 * np.einsum("ikcA,aAkl,lcjb->iajb",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] += -2.0000 * np.einsum("ikAB,cAjk,bBac->ibja",t2["ooVV"],t2["vVoo"],v["vVvv"],optimize="optimal")
	ws2s2["ovov"] +=  2.0000 * np.einsum("ikAB,ACjk,bBaC->ibja",t2["ooVV"],t2["VVoo"],v["vVvV"],optimize="optimal")
	ws2s2["ovov"] += -2.0000 * np.einsum("ikAB,aAjl,lBkb->iajb",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
	ws2s2["ovov"] +=  2.0000 * np.einsum("ikAB,aAkl,lBjb->iajb",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
	ws2s2["ovov"] += -1.0000 * np.einsum("ikaA,BCjk,bABC->ibja",t2["oovV"],t2["VVoo"],v["vVVV"],optimize="optimal")
	ws2s2["ovov"] += -1.0000 * np.einsum("ikaA,cAkl,lbjc->ibja",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] +=  1.0000 * np.einsum("ikaA,ABkl,lbjB->ibja",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["ovov"] += -1.0000 * np.einsum("ikaA,bAlm,lmjk->ibja",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["ovov"] += -1.0000 * np.einsum("klaA,bAim,jmkl->jbia",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
	ws2s2["ovov"] += -1.0000 * np.einsum("klaA,bBkl,jAiB->jbia",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
	ws2s2["ovov"] +=  1.0000 * np.einsum("ikcA,cAkl,lbja->ibja",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] += -1.0000 * np.einsum("klcA,aAik,jclb->jaib",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] +=  1.0000 * np.einsum("klcA,cAik,jbla->jbia",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] += -1.0000 * np.einsum("ikAB,aCjk,ABbC->iajb",t2["ooVV"],t2["vVoo"],v["VVvV"],optimize="optimal")
	ws2s2["ovov"] += -1.0000 * np.einsum("ikAB,ABjl,lbka->ibja",t2["ooVV"],t2["VVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] +=  1.0000 * np.einsum("klAB,aAik,jBlb->jaib",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
	ws2s2["ovov"] += -0.5000 * np.einsum("klaA,cAkl,jbic->jbia",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] +=  0.5000 * np.einsum("klaA,ABkl,jbiB->jbia",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
	ws2s2["ovov"] += -0.5000 * np.einsum("klcA,aAkl,jcib->jaib",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] +=  0.5000 * np.einsum("ikAB,ABkl,lbja->ibja",t2["ooVV"],t2["VVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] +=  0.5000 * np.einsum("klAB,ABik,jbla->jbia",t2["ooVV"],t2["VVoo"],v["ovov"],optimize="optimal")
	ws2s2["ovov"] +=  0.5000 * np.einsum("klAB,aAkl,jBib->jaib",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	ws2s2["vvoo"] +=  4.0000 * np.einsum("aAik,bBjl,klAB->abij",t2["vVoo"],t2["vVoo"],v["ooVV"],optimize="optimal")
	ws2s2["vvoo"] += -4.0000 * np.einsum("klcA,aAik,bcjl->abij",t2["oovV"],t2["vVoo"],v["vvoo"],optimize="optimal")
	ws2s2["vvoo"] +=  4.0000 * np.einsum("klAB,aAik,bBjl->abij",t2["ooVV"],t2["vVoo"],v["vVoo"],optimize="optimal")
	ws2s2["vvoo"] += -2.0000 * np.einsum("aAij,bBkl,klAB->abij",t2["vVoo"],t2["vVoo"],v["ooVV"],optimize="optimal")
	ws2s2["vvoo"] += -2.0000 * np.einsum("klcA,cAik,abjl->abij",t2["oovV"],t2["vVoo"],v["vvoo"],optimize="optimal")
	ws2s2["vvoo"] +=  1.0000 * np.einsum("klcA,aAij,bckl->abij",t2["oovV"],t2["vVoo"],v["vvoo"],optimize="optimal")
	ws2s2["vvoo"] +=  1.0000 * np.einsum("klcA,aAkl,bcij->abij",t2["oovV"],t2["vVoo"],v["vvoo"],optimize="optimal")
	ws2s2["vvoo"] += -1.0000 * np.einsum("klAB,aAij,bBkl->abij",t2["ooVV"],t2["vVoo"],v["vVoo"],optimize="optimal")
	ws2s2["vvoo"] += -1.0000 * np.einsum("klAB,ABik,abjl->abij",t2["ooVV"],t2["VVoo"],v["vvoo"],optimize="optimal")
	ws2s2["vvoo"] += -1.0000 * np.einsum("klAB,aAkl,bBij->abij",t2["ooVV"],t2["vVoo"],v["vVoo"],optimize="optimal")
	ws2s2["vvoo"] +=  0.5000 * np.einsum("klcA,cAij,abkl->abij",t2["oovV"],t2["vVoo"],v["vvoo"],optimize="optimal")
	ws2s2["vvoo"] +=  0.2500 * np.einsum("klAB,ABij,abkl->abij",t2["ooVV"],t2["VVoo"],v["vvoo"],optimize="optimal")
	ws2s2["vvoo"] = asym_term(ws2s2["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	ws2s2["ovvv"] +=  4.0000 * np.einsum("ijaA,bBjk,kAcB->ibac",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ovvv"] +=  4.0000 * np.einsum("jkaA,bAjl,ilkc->ibac",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ovvv"] +=  2.0000 * np.einsum("ijaA,dAjk,kcbd->icab",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
	ws2s2["ovvv"] += -2.0000 * np.einsum("ijaA,ABjk,kcbB->icab",t2["oovV"],t2["VVoo"],v["ovvV"],optimize="optimal")
	ws2s2["ovvv"] += -2.0000 * np.einsum("ijaA,bAkl,kljc->ibac",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
	ws2s2["ovvv"] +=  2.0000 * np.einsum("jkaA,bBjk,iAcB->ibac",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
	ws2s2["ovvv"] += -2.0000 * np.einsum("ijdA,aAjk,kdbc->iabc",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
	ws2s2["ovvv"] +=  2.0000 * np.einsum("ijAB,aAjk,kBbc->iabc",t2["ooVV"],t2["vVoo"],v["oVvv"],optimize="optimal")
	ws2s2["ovvv"] +=  1.0000 * np.einsum("jkaA,dAjk,icbd->icab",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
	ws2s2["ovvv"] += -1.0000 * np.einsum("jkaA,ABjk,icbB->icab",t2["oovV"],t2["VVoo"],v["ovvV"],optimize="optimal")
	ws2s2["ovvv"] +=  1.0000 * np.einsum("ijdA,dAjk,kcab->icab",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
	ws2s2["ovvv"] += -0.5000 * np.einsum("jkdA,aAjk,idbc->iabc",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
	ws2s2["ovvv"] +=  0.5000 * np.einsum("ijAB,ABjk,kcab->icab",t2["ooVV"],t2["VVoo"],v["ovvv"],optimize="optimal")
	ws2s2["ovvv"] +=  0.5000 * np.einsum("jkAB,aAjk,iBbc->iabc",t2["ooVV"],t2["vVoo"],v["oVvv"],optimize="optimal")
	ws2s2["ovvv"] = asym_term(ws2s2["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	ws2s2["vvov"] +=  4.0000 * np.einsum("jkaA,bBij,cAkB->bcia",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws2s2["vvov"] +=  4.0000 * np.einsum("jkaA,bAjl,lcik->bcia",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["vvov"] += -2.0000 * np.einsum("jkaA,dAij,bckd->bcia",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
	ws2s2["vvov"] +=  2.0000 * np.einsum("jkaA,ABij,bckB->bcia",t2["oovV"],t2["VVoo"],v["vvoV"],optimize="optimal")
	ws2s2["vvov"] += -2.0000 * np.einsum("jkaA,bAil,lcjk->bcia",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
	ws2s2["vvov"] +=  2.0000 * np.einsum("jkaA,bBjk,cAiB->bcia",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
	ws2s2["vvov"] +=  2.0000 * np.einsum("jkdA,aAij,cdkb->acib",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
	ws2s2["vvov"] += -2.0000 * np.einsum("jkAB,aAij,cBkb->acib",t2["ooVV"],t2["vVoo"],v["vVov"],optimize="optimal")
	ws2s2["vvov"] +=  1.0000 * np.einsum("jkdA,dAij,bcka->bcia",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
	ws2s2["vvov"] +=  1.0000 * np.einsum("jkdA,aAjk,cdib->acib",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
	ws2s2["vvov"] += -1.0000 * np.einsum("jkAB,aAjk,cBib->acib",t2["ooVV"],t2["vVoo"],v["vVov"],optimize="optimal")
	ws2s2["vvov"] += -0.5000 * np.einsum("jkaA,dAjk,bcid->bcia",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
	ws2s2["vvov"] +=  0.5000 * np.einsum("jkaA,ABjk,bciB->bcia",t2["oovV"],t2["VVoo"],v["vvoV"],optimize="optimal")
	ws2s2["vvov"] +=  0.5000 * np.einsum("jkAB,ABij,bcka->bcia",t2["ooVV"],t2["VVoo"],v["vvov"],optimize="optimal")
	ws2s2["vvov"] = asym_term(ws2s2["vvov"],"vvov")

	# vvvv = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_virt_int))
	ws2s2["vvvv"] +=  8.0000 * np.einsum("ijaA,bAik,kdjc->bdac",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
	ws2s2["vvvv"] += -4.0000 * np.einsum("ijaA,bBij,dAcB->bdac",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
	ws2s2["vvvv"] +=  1.0000 * np.einsum("ijaA,eAij,cdbe->cdab",t2["oovV"],t2["vVoo"],v["vvvv"],optimize="optimal")
	ws2s2["vvvv"] += -1.0000 * np.einsum("ijaA,ABij,cdbB->cdab",t2["oovV"],t2["VVoo"],v["vvvV"],optimize="optimal")
	ws2s2["vvvv"] +=  1.0000 * np.einsum("ijeA,aAij,debc->adbc",t2["oovV"],t2["vVoo"],v["vvvv"],optimize="optimal")
	ws2s2["vvvv"] += -1.0000 * np.einsum("ijAB,aAij,dBbc->adbc",t2["ooVV"],t2["vVoo"],v["vVvv"],optimize="optimal")
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
		ws2s2["oooooo"] +=  2.0000 * np.einsum("ijaA,aBkl,nAmB->ijnklm",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
		ws2s2["oooooo"] +=  2.0000 * np.einsum("ijaA,bAkl,namb->ijnklm",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["oooooo"] += -2.0000 * np.einsum("ijaA,ABkl,namB->ijnklm",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
		ws2s2["oooooo"] += -2.0000 * np.einsum("ijAB,aAkl,nBma->ijnklm",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
		ws2s2["oooooo"] +=  2.0000 * np.einsum("ijAB,ACkl,nBmC->ijnklm",t2["ooVV"],t2["VVoo"],v["oVoV"],optimize="optimal")
		ws2s2["oooooo"] +=  1.0000 * np.einsum("ijaA,aAkp,nplm->ijnklm",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
		ws2s2["oooooo"] +=  1.0000 * np.einsum("ipaA,aAjk,mnlp->imnjkl",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
		ws2s2["oooooo"] +=  0.5000 * np.einsum("ijAB,ABkp,nplm->ijnklm",t2["ooVV"],t2["VVoo"],v["oooo"],optimize="optimal")
		ws2s2["oooooo"] +=  0.5000 * np.einsum("ipAB,ABjk,mnlp->imnjkl",t2["ooVV"],t2["VVoo"],v["oooo"],optimize="optimal")
		ws2s2["oooooo"] = asym_term(ws2s2["oooooo"],"oooooo")

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["ooooov"] += -(2./3.) * np.einsum("ijaA,bBkl,mAbB->ijmkla",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
		ws2s2["ooooov"] += -(2./3.) * np.einsum("ijaA,bAkn,mnlb->ijmkla",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s2["ooooov"] +=  (2./3.) * np.einsum("ijaA,ABkn,mnlB->ijmkla",t2["oovV"],t2["VVoo"],v["oooV"],optimize="optimal")
		ws2s2["ooooov"] +=  (2./3.) * np.einsum("inaA,bAjk,lmnb->ilmjka",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s2["ooooov"] += -(2./3.) * np.einsum("inaA,ABjk,lmnB->ilmjka",t2["oovV"],t2["VVoo"],v["oooV"],optimize="optimal")
		ws2s2["ooooov"] += -(2./3.) * np.einsum("inaA,bAjn,lmkb->ilmjka",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s2["ooooov"] +=  (2./3.) * np.einsum("inaA,ABjn,lmkB->ilmjka",t2["oovV"],t2["VVoo"],v["oooV"],optimize="optimal")
		ws2s2["ooooov"] +=  (2./3.) * np.einsum("ijbA,bBkl,mAaB->ijmkla",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
		ws2s2["ooooov"] +=  (2./3.) * np.einsum("ijbA,cAkl,mbac->ijmkla",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
		ws2s2["ooooov"] += -(2./3.) * np.einsum("ijbA,ABkl,mbaB->ijmkla",t2["oovV"],t2["VVoo"],v["ovvV"],optimize="optimal")
		ws2s2["ooooov"] +=  (2./3.) * np.einsum("ijbA,bAkn,mnla->ijmkla",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s2["ooooov"] += -(2./3.) * np.einsum("ijAB,bAkl,mBab->ijmkla",t2["ooVV"],t2["vVoo"],v["oVvv"],optimize="optimal")
		ws2s2["ooooov"] +=  (2./3.) * np.einsum("ijAB,ACkl,mBaC->ijmkla",t2["ooVV"],t2["VVoo"],v["oVvV"],optimize="optimal")
		ws2s2["ooooov"] += -(1./3.) * np.einsum("ijaA,BCkl,mABC->ijmkla",t2["oovV"],t2["VVoo"],v["oVVV"],optimize="optimal")
		ws2s2["ooooov"] += -(1./3.) * np.einsum("inbA,bAjk,lmna->ilmjka",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s2["ooooov"] +=  (1./3.) * np.einsum("ijAB,ABkn,mnla->ijmkla",t2["ooVV"],t2["VVoo"],v["ooov"],optimize="optimal")
		ws2s2["ooooov"] += -(1./6.) * np.einsum("inAB,ABjk,lmna->ilmjka",t2["ooVV"],t2["VVoo"],v["ooov"],optimize="optimal")
		ws2s2["ooooov"] = asym_term(ws2s2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		ws2s2["oovooo"] += -(2./3.) * np.einsum("ijbA,aBkl,bAmB->ijaklm",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
		ws2s2["oovooo"] +=  (2./3.) * np.einsum("ijbA,bBkl,aAmB->ijaklm",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
		ws2s2["oovooo"] +=  (2./3.) * np.einsum("ijbA,cAkl,abmc->ijaklm",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
		ws2s2["oovooo"] += -(2./3.) * np.einsum("ijbA,ABkl,abmB->ijaklm",t2["oovV"],t2["VVoo"],v["vvoV"],optimize="optimal")
		ws2s2["oovooo"] +=  (2./3.) * np.einsum("ijbA,aAkn,nblm->ijaklm",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s2["oovooo"] += -(2./3.) * np.einsum("inbA,aAjk,mbln->imajkl",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s2["oovooo"] +=  (2./3.) * np.einsum("inbA,bAjk,maln->imajkl",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s2["oovooo"] += -(2./3.) * np.einsum("inbA,aAjn,mbkl->imajkl",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s2["oovooo"] += -(2./3.) * np.einsum("ijAB,bAkl,aBmb->ijaklm",t2["ooVV"],t2["vVoo"],v["vVov"],optimize="optimal")
		ws2s2["oovooo"] +=  (2./3.) * np.einsum("ijAB,ACkl,aBmC->ijaklm",t2["ooVV"],t2["VVoo"],v["vVoV"],optimize="optimal")
		ws2s2["oovooo"] += -(2./3.) * np.einsum("ijAB,aAkn,nBlm->ijaklm",t2["ooVV"],t2["vVoo"],v["oVoo"],optimize="optimal")
		ws2s2["oovooo"] +=  (2./3.) * np.einsum("inAB,aAjk,mBln->imajkl",t2["ooVV"],t2["vVoo"],v["oVoo"],optimize="optimal")
		ws2s2["oovooo"] +=  (2./3.) * np.einsum("inAB,aAjn,mBkl->imajkl",t2["ooVV"],t2["vVoo"],v["oVoo"],optimize="optimal")
		ws2s2["oovooo"] += -(1./3.) * np.einsum("ijbA,bAkn,nalm->ijaklm",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s2["oovooo"] += -(1./3.) * np.einsum("ijAB,aCkl,ABmC->ijaklm",t2["ooVV"],t2["vVoo"],v["VVoV"],optimize="optimal")
		ws2s2["oovooo"] +=  (1./3.) * np.einsum("inAB,ABjk,maln->imajkl",t2["ooVV"],t2["VVoo"],v["ovoo"],optimize="optimal")
		ws2s2["oovooo"] += -(1./6.) * np.einsum("ijAB,ABkn,nalm->ijaklm",t2["ooVV"],t2["VVoo"],v["ovoo"],optimize="optimal")
		ws2s2["oovooo"] = asym_term(ws2s2["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["oooovv"] += -(4./3.) * np.einsum("ijaA,kmbB,ABlm->ijklab",t2["oovV"],t2["oovV"],v["VVoo"],optimize="optimal")
		ws2s2["oooovv"] +=  (2./3.) * np.einsum("ijaA,cAkm,lmbc->ijlkab",t2["oovV"],t2["vVoo"],v["oovv"],optimize="optimal")
		ws2s2["oooovv"] += -(2./3.) * np.einsum("ijaA,ABkm,lmbB->ijlkab",t2["oovV"],t2["VVoo"],v["oovV"],optimize="optimal")
		ws2s2["oooovv"] +=  (2./3.) * np.einsum("imaA,cAjm,klbc->ikljab",t2["oovV"],t2["vVoo"],v["oovv"],optimize="optimal")
		ws2s2["oooovv"] += -(2./3.) * np.einsum("imaA,ABjm,klbB->ikljab",t2["oovV"],t2["VVoo"],v["oovV"],optimize="optimal")
		ws2s2["oooovv"] +=  (1./3.) * np.einsum("ijcA,cAkm,lmab->ijlkab",t2["oovV"],t2["vVoo"],v["oovv"],optimize="optimal")
		ws2s2["oooovv"] +=  (1./6.) * np.einsum("ijAB,ABkm,lmab->ijlkab",t2["ooVV"],t2["VVoo"],v["oovv"],optimize="optimal")
		ws2s2["oooovv"] = asym_term(ws2s2["oooovv"],"oooovv")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s2["oovoov"] += -(8./9.) * np.einsum("imaA,bBjm,lAkB->ilbjka",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
		ws2s2["oovoov"] +=  (8./9.) * np.einsum("imaA,bAjn,lnkm->ilbjka",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
		ws2s2["oovoov"] +=  (4./9.) * np.einsum("ijaA,bBkm,mAlB->ijbkla",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
		ws2s2["oovoov"] +=  (4./9.) * np.einsum("imaA,bBjk,lAmB->ilbjka",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
		ws2s2["oovoov"] +=  (4./9.) * np.einsum("imaA,cAjk,lbmc->ilbjka",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["oovoov"] += -(4./9.) * np.einsum("imaA,ABjk,lbmB->ilbjka",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
		ws2s2["oovoov"] += -(4./9.) * np.einsum("imaA,cAjm,lbkc->ilbjka",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["oovoov"] +=  (4./9.) * np.einsum("imaA,ABjm,lbkB->ilbjka",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
		ws2s2["oovoov"] +=  (4./9.) * np.einsum("ijcA,aAkm,mclb->ijaklb",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["oovoov"] += -(4./9.) * np.einsum("imcA,aAjm,lckb->ilajkb",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["oovoov"] += -(4./9.) * np.einsum("ijAB,aAkm,mBlb->ijaklb",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
		ws2s2["oovoov"] +=  (4./9.) * np.einsum("imAB,aAjm,lBkb->ilajkb",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
		ws2s2["oovoov"] += -(2./9.) * np.einsum("ijaA,cBkl,bAcB->ijbkla",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
		ws2s2["oovoov"] +=  (2./9.) * np.einsum("ijaA,cAkm,mblc->ijbkla",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["oovoov"] += -(2./9.) * np.einsum("ijaA,ABkm,mblB->ijbkla",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
		ws2s2["oovoov"] +=  (2./9.) * np.einsum("imaA,bAmn,lnjk->ilbjka",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
		ws2s2["oovoov"] +=  (2./9.) * np.einsum("mnaA,bAim,kljn->klbija",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
		ws2s2["oovoov"] += -(2./9.) * np.einsum("ijcA,aBkl,cAbB->ijaklb",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
		ws2s2["oovoov"] +=  (2./9.) * np.einsum("ijcA,cBkl,bAaB->ijbkla",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
		ws2s2["oovoov"] +=  (2./9.) * np.einsum("ijcA,dAkl,bcad->ijbkla",t2["oovV"],t2["vVoo"],v["vvvv"],optimize="optimal")
		ws2s2["oovoov"] += -(2./9.) * np.einsum("ijcA,ABkl,bcaB->ijbkla",t2["oovV"],t2["VVoo"],v["vvvV"],optimize="optimal")
		ws2s2["oovoov"] += -(2./9.) * np.einsum("ijcA,cAkm,mbla->ijbkla",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["oovoov"] +=  (2./9.) * np.einsum("imcA,aAjk,lcmb->ilajkb",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["oovoov"] += -(2./9.) * np.einsum("imcA,cAjk,lbma->ilbjka",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["oovoov"] += -(2./9.) * np.einsum("ijAB,cAkl,bBac->ijbkla",t2["ooVV"],t2["vVoo"],v["vVvv"],optimize="optimal")
		ws2s2["oovoov"] +=  (2./9.) * np.einsum("ijAB,ACkl,bBaC->ijbkla",t2["ooVV"],t2["VVoo"],v["vVvV"],optimize="optimal")
		ws2s2["oovoov"] += -(2./9.) * np.einsum("imAB,aAjk,lBmb->ilajkb",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
		ws2s2["oovoov"] += -(1./9.) * np.einsum("ijAB,ABkm,mbla->ijbkla",t2["ooVV"],t2["VVoo"],v["ovov"],optimize="optimal")
		ws2s2["oovoov"] += -(1./9.) * np.einsum("ijaA,BCkl,bABC->ijbkla",t2["oovV"],t2["VVoo"],v["vVVV"],optimize="optimal")
		ws2s2["oovoov"] += -(1./9.) * np.einsum("ijAB,aCkl,ABbC->ijaklb",t2["ooVV"],t2["vVoo"],v["VVvV"],optimize="optimal")
		ws2s2["oovoov"] += -(1./9.) * np.einsum("imAB,ABjk,lbma->ilbjka",t2["ooVV"],t2["VVoo"],v["ovov"],optimize="optimal")
		ws2s2["oovoov"] += -(1./18.) * np.einsum("ijaA,bAmn,mnkl->ijbkla",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
		ws2s2["oovoov"] += -(1./18.) * np.einsum("mnaA,bAij,klmn->klbija",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
		ws2s2["oovoov"] = asym_term(ws2s2["oovoov"],"oovoov")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2s2["ovvooo"] += -(4./3.) * np.einsum("aAij,bBkm,lmAB->labijk",t2["vVoo"],t2["vVoo"],v["ooVV"],optimize="optimal")
		ws2s2["ovvooo"] +=  (2./3.) * np.einsum("imcA,aAjk,bclm->iabjkl",t2["oovV"],t2["vVoo"],v["vvoo"],optimize="optimal")
		ws2s2["ovvooo"] +=  (2./3.) * np.einsum("imcA,aAjm,bckl->iabjkl",t2["oovV"],t2["vVoo"],v["vvoo"],optimize="optimal")
		ws2s2["ovvooo"] += -(2./3.) * np.einsum("imAB,aAjk,bBlm->iabjkl",t2["ooVV"],t2["vVoo"],v["vVoo"],optimize="optimal")
		ws2s2["ovvooo"] += -(2./3.) * np.einsum("imAB,aAjm,bBkl->iabjkl",t2["ooVV"],t2["vVoo"],v["vVoo"],optimize="optimal")
		ws2s2["ovvooo"] +=  (1./3.) * np.einsum("imcA,cAjk,ablm->iabjkl",t2["oovV"],t2["vVoo"],v["vvoo"],optimize="optimal")
		ws2s2["ovvooo"] +=  (1./6.) * np.einsum("imAB,ABjk,ablm->iabjkl",t2["ooVV"],t2["VVoo"],v["vvoo"],optimize="optimal")
		ws2s2["ovvooo"] = asym_term(ws2s2["ovvooo"],"ovvooo")

		# ooovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["ooovvv"] += 4.0000 * np.einsum("ijaA,klbB,ABlc->ijkabc",t2["oovV"],t2["oovV"],v["VVov"],optimize="optimal")
		ws2s2["ooovvv"] = asym_term(ws2s2["ooovvv"],"ooovvv")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s2["oovovv"] +=  (8./9.) * np.einsum("ilaA,bBjl,kAcB->ikbjac",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
		ws2s2["oovovv"] +=  (8./9.) * np.einsum("ilaA,bAjm,kmlc->ikbjac",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s2["oovovv"] += -(4./9.) * np.einsum("ijaA,bBkl,lAcB->ijbkac",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
		ws2s2["oovovv"] +=  (4./9.) * np.einsum("ilaA,dAjl,kcbd->ikcjab",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
		ws2s2["oovovv"] += -(4./9.) * np.einsum("ilaA,ABjl,kcbB->ikcjab",t2["oovV"],t2["VVoo"],v["ovvV"],optimize="optimal")
		ws2s2["oovovv"] += -(4./9.) * np.einsum("ilaA,bAlm,kmjc->ikbjac",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s2["oovovv"] += -(2./9.) * np.einsum("ijaA,dAkl,lcbd->ijckab",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
		ws2s2["oovovv"] +=  (2./9.) * np.einsum("ijaA,ABkl,lcbB->ijckab",t2["oovV"],t2["VVoo"],v["ovvV"],optimize="optimal")
		ws2s2["oovovv"] +=  (2./9.) * np.einsum("lmaA,bAil,jkmc->jkbiac",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s2["oovovv"] +=  (2./9.) * np.einsum("ijdA,aAkl,ldbc->ijakbc",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
		ws2s2["oovovv"] += -(2./9.) * np.einsum("ildA,aAjl,kdbc->ikajbc",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
		ws2s2["oovovv"] += -(2./9.) * np.einsum("ijAB,aAkl,lBbc->ijakbc",t2["ooVV"],t2["vVoo"],v["oVvv"],optimize="optimal")
		ws2s2["oovovv"] +=  (2./9.) * np.einsum("ilAB,aAjl,kBbc->ikajbc",t2["ooVV"],t2["vVoo"],v["oVvv"],optimize="optimal")
		ws2s2["oovovv"] +=  (1./9.) * np.einsum("ijaA,bAlm,lmkc->ijbkac",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s2["oovovv"] += -(1./9.) * np.einsum("ijdA,dAkl,lcab->ijckab",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
		ws2s2["oovovv"] += -(1./18.) * np.einsum("ijAB,ABkl,lcab->ijckab",t2["ooVV"],t2["VVoo"],v["ovvv"],optimize="optimal")
		ws2s2["oovovv"] = asym_term(ws2s2["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s2["ovvoov"] +=  (8./9.) * np.einsum("ilaA,bBjl,cAkB->ibcjka",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
		ws2s2["ovvoov"] +=  (8./9.) * np.einsum("ilaA,bAjm,mckl->ibcjka",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s2["ovvoov"] += -(4./9.) * np.einsum("ilaA,bBjk,cAlB->ibcjka",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
		ws2s2["ovvoov"] += -(4./9.) * np.einsum("lmaA,bAil,kcjm->kbcija",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s2["ovvoov"] +=  (4./9.) * np.einsum("ildA,aAjl,cdkb->iacjkb",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
		ws2s2["ovvoov"] += -(4./9.) * np.einsum("ilAB,aAjl,cBkb->iacjkb",t2["ooVV"],t2["vVoo"],v["vVov"],optimize="optimal")
		ws2s2["ovvoov"] +=  (2./9.) * np.einsum("ilaA,dAjk,bcld->ibcjka",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
		ws2s2["ovvoov"] += -(2./9.) * np.einsum("ilaA,ABjk,bclB->ibcjka",t2["oovV"],t2["VVoo"],v["vvoV"],optimize="optimal")
		ws2s2["ovvoov"] += -(2./9.) * np.einsum("ilaA,dAjl,bckd->ibcjka",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
		ws2s2["ovvoov"] +=  (2./9.) * np.einsum("ilaA,ABjl,bckB->ibcjka",t2["oovV"],t2["VVoo"],v["vvoV"],optimize="optimal")
		ws2s2["ovvoov"] +=  (2./9.) * np.einsum("ilaA,bAlm,mcjk->ibcjka",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s2["ovvoov"] += -(2./9.) * np.einsum("ildA,aAjk,cdlb->iacjkb",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
		ws2s2["ovvoov"] +=  (2./9.) * np.einsum("ilAB,aAjk,cBlb->iacjkb",t2["ooVV"],t2["vVoo"],v["vVov"],optimize="optimal")
		ws2s2["ovvoov"] +=  (1./9.) * np.einsum("lmaA,bAij,kclm->kbcija",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s2["ovvoov"] += -(1./9.) * np.einsum("ildA,dAjk,bcla->ibcjka",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
		ws2s2["ovvoov"] += -(1./18.) * np.einsum("ilAB,ABjk,bcla->ibcjka",t2["ooVV"],t2["VVoo"],v["vvov"],optimize="optimal")
		ws2s2["ovvoov"] = asym_term(ws2s2["ovvoov"],"ovvoov")

		# vvvooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		ws2s2["vvvooo"] +=  4.0000 * np.einsum("aAij,bBkl,lcAB->abcijk",t2["vVoo"],t2["vVoo"],v["ovVV"],optimize="optimal")
		ws2s2["vvvooo"] = asym_term(ws2s2["vvvooo"],"vvvooo")

		# oovvvv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["oovvvv"] +=  (2./3.) * np.einsum("ikaA,bAkl,jlcd->ijbacd",t2["oovV"],t2["vVoo"],v["oovv"],optimize="optimal")
		ws2s2["oovvvv"] += -(1./6.) * np.einsum("ijaA,bAkl,klcd->ijbacd",t2["oovV"],t2["vVoo"],v["oovv"],optimize="optimal")
		ws2s2["oovvvv"] = asym_term(ws2s2["oovvvv"],"oovvvv")

		# ovvovv = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s2["ovvovv"] += -(8./9.) * np.einsum("ikaA,bBjk,dAcB->ibdjac",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
		ws2s2["ovvovv"] +=  (8./9.) * np.einsum("ikaA,bAjl,ldkc->ibdjac",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["ovvovv"] += -(4./9.) * np.einsum("ikaA,bAkl,ldjc->ibdjac",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["ovvovv"] += -(4./9.) * np.einsum("klaA,bAik,jdlc->jbdiac",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["ovvovv"] +=  (2./9.) * np.einsum("ikaA,eAjk,cdbe->icdjab",t2["oovV"],t2["vVoo"],v["vvvv"],optimize="optimal")
		ws2s2["ovvovv"] += -(2./9.) * np.einsum("ikaA,ABjk,cdbB->icdjab",t2["oovV"],t2["VVoo"],v["vvvV"],optimize="optimal")
		ws2s2["ovvovv"] +=  (2./9.) * np.einsum("ikeA,aAjk,debc->iadjbc",t2["oovV"],t2["vVoo"],v["vvvv"],optimize="optimal")
		ws2s2["ovvovv"] += -(2./9.) * np.einsum("ikAB,aAjk,dBbc->iadjbc",t2["ooVV"],t2["vVoo"],v["vVvv"],optimize="optimal")
		ws2s2["ovvovv"] = asym_term(ws2s2["ovvovv"],"ovvovv")

		#vvvoov = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		ws2s2["vvvoov"] +=  (2./3.) * np.einsum("klaA,bAik,cdjl->bcdija",t2["oovV"],t2["vVoo"],v["vvoo"],optimize="optimal")
		ws2s2["vvvoov"] += -(1./6.) * np.einsum("klaA,bAij,cdkl->bcdija",t2["oovV"],t2["vVoo"],v["vvoo"],optimize="optimal")
		ws2s2["vvvoov"] = asym_term(ws2s2["vvvoov"],"vvvoov")

		# ovvvvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["ovvvvv"] +=  (2./3.) * np.einsum("ijaA,bAjk,kecd->ibeacd",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
		ws2s2["ovvvvv"] = asym_term(ws2s2["ovvvvv"],"ovvvvv") 

		# vvvovv = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_virt_int,n_virt_int))
		ws2s2["vvvovv"] +=  (2./3.) * np.einsum("jkaA,bAij,dekc->bdeiac",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
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
		ws2s2["ooooooov"] += -0.1250 * np.einsum("ijaA,bAkl,npmb->ijnpklma",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s2["ooooooov"] +=  0.1250 * np.einsum("ijaA,ABkl,npmB->ijnpklma",t2["oovV"],t2["VVoo"],v["oooV"],optimize="optimal")
		ws2s2["ooooooov"] = asym_term(ws2s2["ooooooov"],"ooooooov")

		# ooovoooo = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		ws2s2["ooovoooo"] += -0.1250 * np.einsum("ijbA,aAkl,pbmn->ijpaklmn",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s2["ooovoooo"] +=  0.1250 * np.einsum("ijAB,aAkl,pBmn->ijpaklmn",t2["ooVV"],t2["vVoo"],v["oVoo"],optimize="optimal")
		ws2s2["ooovoooo"] = asym_term(ws2s2["ooovoooo"],"ooovoooo")

		# oooooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["oooooovv"] +=  (1./12.) * np.einsum("ijaA,cAkl,mnbc->ijmnklab",t2["oovV"],t2["vVoo"],v["oovv"],optimize="optimal")
		ws2s2["oooooovv"] += -(1./12.) * np.einsum("ijaA,ABkl,mnbB->ijmnklab",t2["oovV"],t2["VVoo"],v["oovV"],optimize="optimal")
		ws2s2["oooooovv"] += -(1./12.) * np.einsum("ijaA,klbB,ABmn->ijklmnab",t2["oovV"],t2["oovV"],v["VVoo"],optimize="optimal")
		ws2s2["oooooovv"] = asym_term(ws2s2["oooooovv"],"oooooovv")

		# ooovooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["ooovooov"] += -0.1250 * np.einsum("ijaA,bBkl,nAmB->ijnbklma",t2["oovV"],t2["vVoo"],v["oVoV"],optimize="optimal")
		ws2s2["ooovooov"] += -0.0625 * np.einsum("ijaA,cAkl,nbmc->ijnbklma",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["ooovooov"] +=  0.0625 * np.einsum("ijaA,ABkl,nbmB->ijnbklma",t2["oovV"],t2["VVoo"],v["ovoV"],optimize="optimal")
		ws2s2["ooovooov"] += -0.0625 * np.einsum("ijaA,bAkp,nplm->ijnbklma",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
		ws2s2["ooovooov"] += -0.0625 * np.einsum("ipaA,bAjk,mnlp->imnbjkla",t2["oovV"],t2["vVoo"],v["oooo"],optimize="optimal")
		ws2s2["ooovooov"] += -0.0625 * np.einsum("ijcA,aAkl,ncmb->ijnaklmb",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["ooovooov"] +=  0.0625 * np.einsum("ijAB,aAkl,nBmb->ijnaklmb",t2["ooVV"],t2["vVoo"],v["oVov"],optimize="optimal")
		ws2s2["ooovooov"] = asym_term(ws2s2["ooovooov"],"ooovooov")

		# oovvoooo = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		ws2s2["oovvoooo"] += -(1./12.) * np.einsum("aAij,bBkl,mnAB->mnabijkl",t2["vVoo"],t2["vVoo"],v["ooVV"],optimize="optimal")
		ws2s2["oovvoooo"] +=  (1./12.) * np.einsum("ijcA,aAkl,bcmn->ijabklmn",t2["oovV"],t2["vVoo"],v["vvoo"],optimize="optimal")
		ws2s2["oovvoooo"] += -(1./12.) * np.einsum("ijAB,aAkl,bBmn->ijabklmn",t2["ooVV"],t2["vVoo"],v["vVoo"],optimize="optimal")
		ws2s2["oovvoooo"] = asym_term(ws2s2["oovvoooo"],"oovvoooo")

		# ooooovvv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["ooooovvv"] += -0.2500 * np.einsum("ijaA,klbB,ABmc->ijklmabc",t2["oovV"],t2["oovV"],v["VVov"],optimize="optimal")
		ws2s2["ooooovvv"] = asym_term(ws2s2["ooooovvv"],"ooooovvv")

		# ooovoovv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["ooovoovv"] +=  (1./12.) * np.einsum("ijaA,bBkl,mAcB->ijmbklac",t2["oovV"],t2["vVoo"],v["oVvV"],optimize="optimal")
		ws2s2["ooovoovv"] +=  (1./12.) * np.einsum("ijaA,bAkn,mnlc->ijmbklac",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s2["ooovoovv"] +=  (1./24.) * np.einsum("ijaA,dAkl,mcbd->ijmcklab",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
		ws2s2["ooovoovv"] += -(1./24.) * np.einsum("ijaA,ABkl,mcbB->ijmcklab",t2["oovV"],t2["VVoo"],v["ovvV"],optimize="optimal")
		ws2s2["ooovoovv"] += -(1./24.) * np.einsum("inaA,bAjk,lmnc->ilmbjkac",t2["oovV"],t2["vVoo"],v["ooov"],optimize="optimal")
		ws2s2["ooovoovv"] += -(1./48.) * np.einsum("ijdA,aAkl,mdbc->ijmaklbc",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
		ws2s2["ooovoovv"] +=  (1./48.) * np.einsum("ijAB,aAkl,mBbc->ijmaklbc",t2["ooVV"],t2["vVoo"],v["oVvv"],optimize="optimal")
		ws2s2["ooovoovv"] = asym_term(ws2s2["ooovoovv"],"ooovoovv")

		# oovvooov = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["oovvooov"] +=  (1./12.) * np.einsum("ijaA,bBkl,cAmB->ijbcklma",t2["oovV"],t2["vVoo"],v["vVoV"],optimize="optimal")
		ws2s2["oovvooov"] +=  (1./12.) * np.einsum("inaA,bAjk,mcln->imbcjkla",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s2["oovvooov"] += -(1./24.) * np.einsum("ijaA,bAkn,nclm->ijbcklma",t2["oovV"],t2["vVoo"],v["ovoo"],optimize="optimal")
		ws2s2["oovvooov"] +=  (1./24.) * np.einsum("ijdA,aAkl,cdmb->ijacklmb",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
		ws2s2["oovvooov"] += -(1./24.) * np.einsum("ijAB,aAkl,cBmb->ijacklmb",t2["ooVV"],t2["vVoo"],v["vVov"],optimize="optimal")
		ws2s2["oovvooov"] += -(1./48.) * np.einsum("ijaA,dAkl,bcmd->ijbcklma",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
		ws2s2["oovvooov"] +=  (1./48.) * np.einsum("ijaA,ABkl,bcmB->ijbcklma",t2["oovV"],t2["VVoo"],v["vvoV"],optimize="optimal")
		ws2s2["oovvooov"] = asym_term(ws2s2["oovvooov"],"oovvooov")

		# ovvvoooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		ws2s2["ovvvoooo"] += -0.2500 * np.einsum("aAij,bBkl,mcAB->mabcijkl",t2["vVoo"],t2["vVoo"],v["ovVV"],optimize="optimal")
		ws2s2["ovvvoooo"] = asym_term(ws2s2["ovvvoooo"],"ovvvoooo")

		# oooovvvv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["oooovvvv"] += -0.5000 * np.einsum("ijaA,klbB,ABcd->ijklabcd",t2["oovV"],t2["oovV"],v["VVvv"],optimize="optimal")
		ws2s2["oooovvvv"] = asym_term(ws2s2["oooovvvv"],"oooovvvv")

		# ooovovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["ooovovvv"] += -0.0625 * np.einsum("ijaA,bAkm,lmcd->ijlbkacd",t2["oovV"],t2["vVoo"],v["oovv"],optimize="optimal")
		ws2s2["ooovovvv"] = asym_term(ws2s2["ooovovvv"],"ooovovvv")

		# oovvoovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["oovvoovv"] += -(1./18.) * np.einsum("ijaA,bBkl,dAcB->ijbdklac",t2["oovV"],t2["vVoo"],v["vVvV"],optimize="optimal")
		ws2s2["oovvoovv"] +=  (1./18.) * np.einsum("ijaA,bAkm,mdlc->ijbdklac",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["oovvoovv"] +=  (1./18.) * np.einsum("imaA,bAjk,ldmc->ilbdjkac",t2["oovV"],t2["vVoo"],v["ovov"],optimize="optimal")
		ws2s2["oovvoovv"] +=  (1./72.) * np.einsum("ijaA,eAkl,cdbe->ijcdklab",t2["oovV"],t2["vVoo"],v["vvvv"],optimize="optimal")
		ws2s2["oovvoovv"] += -(1./72.) * np.einsum("ijaA,ABkl,cdbB->ijcdklab",t2["oovV"],t2["VVoo"],v["vvvV"],optimize="optimal")
		ws2s2["oovvoovv"] +=  (1./72.) * np.einsum("ijeA,aAkl,debc->ijadklbc",t2["oovV"],t2["vVoo"],v["vvvv"],optimize="optimal")
		ws2s2["oovvoovv"] += -(1./72.) * np.einsum("ijAB,aAkl,dBbc->ijadklbc",t2["ooVV"],t2["vVoo"],v["vVvv"],optimize="optimal")
		ws2s2["oovvoovv"] = asym_term(ws2s2["oovvoovv"],"oovvoovv")

		# ovvvooov = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		ws2s2["ovvvooov"] += -0.0625 * np.einsum("imaA,bAjk,cdlm->ibcdjkla",t2["oovV"],t2["vVoo"],v["vvoo"],optimize="optimal")
		ws2s2["ovvvooov"] = asym_term(ws2s2["ovvvooov"],"ovvvooov")

		# vvvvoooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		ws2s2["vvvvoooo"] += -0.5000 * np.einsum("aAij,bBkl,cdAB->abcdijkl",t2["vVoo"],t2["vVoo"],v["vvVV"],optimize="optimal")
		ws2s2["vvvvoooo"] = asym_term(ws2s2["vvvvoooo"],"vvvvoooo")

		# oovvovvv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_virt_int,n_virt_int,n_virt_int))
		ws2s2["oovvovvv"] += -(1./24.) * np.einsum("ijaA,bAkl,lecd->ijbekacd",t2["oovV"],t2["vVoo"],v["ovvv"],optimize="optimal")
		ws2s2["oovvovvv"] = asym_term(ws2s2["oovvovvv"],"oovvovvv")

		# ovvvoovv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int,n_virt_int))
		ws2s2["ovvvoovv"] += -(1./24.) * np.einsum("ilaA,bAjk,delc->ibdejkac",t2["oovV"],t2["vVoo"],v["vvov"],optimize="optimal")
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
	fs1s1s1["c"] += -4.000 * np.einsum("Ai,iB,jA,Bj->",f["Vo"],t1["oV"],t1["oV"],t1["Vo"],optimize="optimal")
	fs1s1s1["c"] += -4.000 * np.einsum("iA,jB,Aj,Bi->",f["oV"],t1["oV"],t1["Vo"],t1["Vo"],optimize="optimal")

	fs1s1s1["oo"] += -1.000 * np.einsum("Ai,jB,kA,Bk->ji",f["Vo"],t1["oV"],t1["oV"],t1["Vo"],optimize="optimal")
	fs1s1s1["oo"] += -3.000 * np.einsum("Ak,kB,iA,Bj->ij",f["Vo"],t1["oV"],t1["oV"],t1["Vo"],optimize="optimal")
	fs1s1s1["oo"] += -1.000 * np.einsum("iA,kB,Bj,Ak->ij",f["oV"],t1["oV"],t1["Vo"],t1["Vo"],optimize="optimal")
	fs1s1s1["oo"] += -3.000 * np.einsum("kA,iB,Aj,Bk->ij",f["oV"],t1["oV"],t1["Vo"],t1["Vo"],optimize="optimal")

	fs1s1s1["ov"] += -1.000 * np.einsum("Aa,iB,jA,Bj->ia",f["Vv"],t1["oV"],t1["oV"],t1["Vo"],optimize="optimal")

	fs1s1s1["vo"] += -1.000 * np.einsum("aA,jB,Bi,Aj->ai",f["vV"],t1["oV"],t1["Vo"],t1["Vo"],optimize="optimal")

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
	fs1s1s2["ov"] += -2.000 * np.einsum("jA,Ak,Bj,ikaB->ia",f["oV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s1s2["ov"] += -1.000 * np.einsum("Aj,kA,Bk,ijaB->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s1s2["ov"] += -1.000 * np.einsum("Aj,jB,Bk,ikaA->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")

	fs1s1s2["vo"] += -2.000 * np.einsum("Aj,jB,kA,aBik->ai",f["Vo"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s1s2["vo"] += -1.000 * np.einsum("jA,kB,Bj,aAik->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs1s1s2["vo"] += -1.000 * np.einsum("jA,kB,Ak,aBij->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")

	fs1s1s2["ooov"] +=  0.500 * np.einsum("lA,Ai,Bl,jkaB->jkia",f["oV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s1s2["ooov"] +=  0.250 * np.einsum("Ai,lA,Bl,jkaB->jkia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s1s2["ooov"] +=  0.250 * np.einsum("Al,lB,Bi,jkaA->jkia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")

	fs1s1s2["ovoo"] +=  0.500 * np.einsum("Al,lB,iA,aBjk->iajk",f["Vo"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s1s2["ovoo"] +=  0.250 * np.einsum("iA,lB,Al,aBjk->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs1s1s2["ovoo"] +=  0.250 * np.einsum("lA,iB,Bl,aAjk->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")

	fs1s1s2["oovv"] +=  0.250 * np.einsum("Aa,kA,Bk,ijbB->ijab",f["Vv"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s1s2["oovv"] += -0.250 * np.einsum("Aa,kA,Bk,ijbB->ijba",f["Vv"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")

	fs1s1s2["vvoo"] +=  0.250 * np.einsum("aA,kB,Ak,bBij->abij",f["vV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs1s1s2["vvoo"] += -0.250 * np.einsum("aA,kB,Ak,bBij->baij",f["vV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")

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
	fs1s2s1["c"] +=  1.000 * np.einsum("ji,Ak,Bj,ikAB->",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs1s2s1["c"] +=  1.000 * np.einsum("ji,kA,iB,ABjk->",f["oo"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs1s2s1["c"] += -1.000 * np.einsum("Aa,iB,jA,aBij->",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2s1["c"] += -1.000 * np.einsum("aA,Bi,Aj,ijaB->",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2s1["c"] += -1.000 * np.einsum("BA,Ci,Aj,ijBC->",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs1s2s1["c"] += -1.000 * np.einsum("BA,iC,jB,ACij->",f["VV"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")

	fs1s2s1["oo"] +=  1.000 * np.einsum("ki,Al,Bk,jlAB->ji",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs1s2s1["oo"] +=  1.000 * np.einsum("ik,lA,kB,ABjl->ij",f["oo"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs1s2s1["oo"] += -1.000 * np.einsum("lk,Ai,Bl,jkAB->ji",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs1s2s1["oo"] += -1.000 * np.einsum("lk,iA,kB,ABjl->ij",f["oo"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs1s2s1["oo"] += -1.000 * np.einsum("Aa,iB,kA,aBjk->ij",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2s1["oo"] +=  1.000 * np.einsum("Aa,kB,iA,aBjk->ij",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2s1["oo"] +=  1.000 * np.einsum("aA,Ai,Bk,jkaB->ji",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2s1["oo"] += -1.000 * np.einsum("aA,Bi,Ak,jkaB->ji",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2s1["oo"] +=  1.000 * np.einsum("BA,Ai,Ck,jkBC->ji",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs1s2s1["oo"] += -1.000 * np.einsum("BA,Ci,Ak,jkBC->ji",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs1s2s1["oo"] += -1.000 * np.einsum("BA,iC,kB,ACjk->ij",f["VV"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs1s2s1["oo"] +=  1.000 * np.einsum("BA,kC,iB,ACjk->ij",f["VV"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")

	fs1s2s1["ov"] += -1.000 * np.einsum("Aj,iA,Bk,jkaB->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2s1["ov"] += -1.000 * np.einsum("Aj,kA,Bk,ijaB->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2s1["ov"] += -1.000 * np.einsum("Aj,jB,Bk,ikaA->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2s1["ov"] +=  1.000 * np.einsum("ja,Ak,Bj,ikAB->ia",f["ov"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs1s2s1["ov"] +=  1.000 * np.einsum("iA,Bj,Ak,jkaB->ia",f["oV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2s1["ov"] += -2.000 * np.einsum("jA,Ak,Bj,ikaB->ia",f["oV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")

	fs1s2s1["vo"] +=  1.000 * np.einsum("Ai,jB,kA,aBjk->ai",f["Vo"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2s1["vo"] +=  1.000 * np.einsum("aj,kA,jB,ABik->ai",f["vo"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs1s2s1["vo"] += -2.000 * np.einsum("Aj,jB,kA,aBik->ai",f["Vo"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2s1["vo"] += -1.000 * np.einsum("jA,kB,Ai,aBjk->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs1s2s1["vo"] += -1.000 * np.einsum("jA,kB,Bj,aAik->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs1s2s1["vo"] += -1.000 * np.einsum("jA,kB,Ak,aBij->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")

	fs1s2s1["vv"] += 1.000 * np.einsum("Aa,iB,jA,bBij->ba",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2s1["vv"] += 1.000 * np.einsum("aA,Bi,Aj,ijbB->ab",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs1s2s1["oooo"] +=  0.500 * np.einsum("mi,Aj,Bm,klAB->klij",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs1s2s1["oooo"] +=  0.500 * np.einsum("im,jA,mB,ABkl->ijkl",f["oo"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs1s2s1["oooo"] += -0.500 * np.einsum("Aa,iB,jA,aBkl->ijkl",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2s1["oooo"] += -0.500 * np.einsum("aA,Bi,Aj,klaB->klij",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2s1["oooo"] += -0.500 * np.einsum("BA,Ci,Aj,klBC->klij",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs1s2s1["oooo"] += -0.500 * np.einsum("BA,iC,jB,ACkl->ijkl",f["VV"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs1s2s1["oooo"] = asym_term(fs1s2s1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs1s2s1["ooov"] += -0.500 * np.einsum("Al,iA,Bj,klaB->ikja",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2s1["ooov"] += -0.500 * np.einsum("iA,Bj,Al,klaB->ikja",f["oV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2s1["ooov"] +=  0.250 * np.einsum("Al,lB,Bi,jkaA->jkia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2s1["ooov"] += -0.250 * np.einsum("la,Ai,Bl,jkAB->jkia",f["ov"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs1s2s1["ooov"] +=  0.250 * np.einsum("lA,Ai,Bl,jkaB->jkia",f["oV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs1s2s1["ooov"] = asym_term(fs1s2s1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs1s2s1["ovoo"] += -0.500 * np.einsum("Ai,jB,lA,aBkl->jaik",f["Vo"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2s1["ovoo"] += -0.500 * np.einsum("lA,iB,Aj,aBkl->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs1s2s1["ovoo"] += -0.250 * np.einsum("al,iA,lB,ABjk->iajk",f["vo"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs1s2s1["ovoo"] +=  0.250 * np.einsum("Al,lB,iA,aBjk->iajk",f["Vo"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2s1["ovoo"] +=  0.250 * np.einsum("lA,iB,Bl,aAjk->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs1s2s1["ovoo"] = asym_term(fs1s2s1["ovoo"],"ovoo")

	fs1s2s1["ovov"] += 0.250 * np.einsum("Aa,iB,kA,bBjk->ibja",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs1s2s1["ovov"] += 0.250 * np.einsum("aA,Bi,Ak,jkbB->jaib",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")

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
	fs1s2s2["c"] += -2.000 * np.einsum("Ai,jA,ikaB,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["c"] += -2.000 * np.einsum("iA,Aj,jkaB,aBik->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["c"] += -1.000 * np.einsum("ai,iA,jkaB,ABjk->",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["c"] += -1.000 * np.einsum("Ai,jA,ikBC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["c"] += -1.000 * np.einsum("Ai,iB,jkaA,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["c"] += -1.000 * np.einsum("Ai,iB,jkAC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["c"] += -1.000 * np.einsum("ia,Ai,jkAB,aBjk->",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["c"] += -1.000 * np.einsum("iA,Bi,jkaB,aAjk->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["c"] += -1.000 * np.einsum("iA,Bi,jkBC,ACjk->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["c"] += -1.000 * np.einsum("iA,Aj,jkBC,BCik->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")

	fs1s2s2["oo"] += -2.000 * np.einsum("ak,kA,ilaB,ABjl->ij",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oo"] += -2.000 * np.einsum("Ak,lA,ikaB,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oo"] += -2.000 * np.einsum("Ak,kB,ilaA,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oo"] += -2.000 * np.einsum("Ak,kB,ilAC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oo"] += -2.000 * np.einsum("ka,Ak,ilAB,aBjl->ij",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oo"] += -2.000 * np.einsum("kA,Bk,ilaB,aAjl->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oo"] += -2.000 * np.einsum("kA,Bk,ilBC,ACjl->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oo"] += -2.000 * np.einsum("kA,Al,ilaB,aBjk->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oo"] += -1.000 * np.einsum("Ai,kA,jlaB,aBkl->ji",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oo"] += -1.000 * np.einsum("Ak,iA,klaB,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oo"] += -1.000 * np.einsum("Ak,lA,ikBC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oo"] += -1.000 * np.einsum("iA,Ak,klaB,aBjl->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oo"] += -1.000 * np.einsum("kA,Ai,jlaB,aBkl->ji",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oo"] += -1.000 * np.einsum("kA,Al,ilBC,BCjk->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oo"] += -0.500 * np.einsum("kA,Ai,jlBC,BCkl->ji",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oo"] += -0.500 * np.einsum("Ai,kA,jlBC,BCkl->ji",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oo"] += -0.500 * np.einsum("Ak,iA,klBC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oo"] += -0.500 * np.einsum("iA,Ak,klBC,BCjl->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")

	fs1s2s2["ov"] += -1.000 * np.einsum("kj,jA,ilaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["ov"] += -1.000 * np.einsum("Aa,jA,ikbB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ov"] +=  1.000 * np.einsum("Ab,jA,ikaB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ov"] +=  1.000 * np.einsum("BA,jB,ikaC,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["ov"] +=  0.500 * np.einsum("ij,jA,klaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["ov"] += -0.500 * np.einsum("Aa,jA,ikBC,BCjk->ia",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["ov"] += -0.500 * np.einsum("Ab,iA,jkaB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ov"] += -0.500 * np.einsum("BA,iB,jkaC,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")

	fs1s2s2["vo"] += -1.000 * np.einsum("kj,Ak,jlAB,aBil->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["vo"] += -1.000 * np.einsum("aA,Aj,jkbB,bBik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["vo"] +=  1.000 * np.einsum("bA,Aj,jkbB,aBik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["vo"] +=  1.000 * np.einsum("BA,Aj,jkBC,aCik->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["vo"] +=  0.500 * np.einsum("ji,Aj,klAB,aBkl->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["vo"] += -0.500 * np.einsum("aA,Aj,jkBC,BCik->ai",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["vo"] += -0.500 * np.einsum("bA,Ai,jkbB,aBjk->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["vo"] += -0.500 * np.einsum("BA,Ai,jkBC,aCjk->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")

	fs1s2s2["vv"] +=  2.000 * np.einsum("Ai,jA,ikaB,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["vv"] +=  2.000 * np.einsum("iA,Aj,jkaB,bBik->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["vv"] +=  1.000 * np.einsum("Ai,iB,jkaA,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["vv"] +=  1.000 * np.einsum("iA,Bi,jkaB,bAjk->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["vv"] +=  0.500 * np.einsum("ai,iA,jkbB,ABjk->ab",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["vv"] +=  0.500 * np.einsum("ia,Ai,jkAB,bBjk->ba",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs1s2s2["oooo"] +=  0.500 * np.einsum("Ai,mA,jkaB,aBlm->jkil",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oooo"] += -0.500 * np.einsum("am,mA,ijaB,ABkl->ijkl",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oooo"] +=  0.500 * np.einsum("Am,iA,jmaB,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oooo"] += -0.500 * np.einsum("Am,mB,ijaA,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oooo"] += -0.500 * np.einsum("Am,mB,ijAC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oooo"] += -0.500 * np.einsum("ma,Am,ijAB,aBkl->ijkl",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oooo"] +=  0.500 * np.einsum("iA,Am,jmaB,aBkl->ijkl",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oooo"] +=  0.500 * np.einsum("mA,Ai,jkaB,aBlm->jkil",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oooo"] += -0.500 * np.einsum("mA,Bm,ijaB,aAkl->ijkl",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["oooo"] += -0.500 * np.einsum("mA,Bm,ijBC,ACkl->ijkl",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oooo"] +=  0.250 * np.einsum("Ai,mA,jkBC,BClm->jkil",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oooo"] +=  0.250 * np.einsum("Am,iA,jmBC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oooo"] +=  0.250 * np.einsum("iA,Am,jmBC,BCkl->ijkl",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oooo"] +=  0.250 * np.einsum("mA,Ai,jkBC,BClm->jkil",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["oooo"] = asym_term(fs1s2s2["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs1s2s2["ooov"] += -0.500 * np.einsum("il,lA,jmaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["ooov"] +=  0.500 * np.einsum("Ab,iA,jlaB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ooov"] +=  0.500 * np.einsum("BA,iB,jlaC,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["ooov"] += -0.250 * np.einsum("ml,lA,ijaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["ooov"] += -0.250 * np.einsum("Aa,lA,ijbB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ooov"] +=  0.250 * np.einsum("Ab,lA,ijaB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ooov"] +=  0.250 * np.einsum("BA,lB,ijaC,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["ooov"] += -0.125 * np.einsum("Aa,lA,ijBC,BCkl->ijka",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["ooov"] = asym_term(fs1s2s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs1s2s2["ovoo"] += -0.500 * np.einsum("li,Al,jmAB,aBkm->jaik",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovoo"] +=  0.500 * np.einsum("bA,Ai,jlbB,aBkl->jaik",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovoo"] +=  0.500 * np.einsum("BA,Ai,jlBC,aCkl->jaik",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovoo"] += -0.250 * np.einsum("ml,Am,ilAB,aBjk->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovoo"] += -0.250 * np.einsum("aA,Al,ilbB,bBjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovoo"] +=  0.250 * np.einsum("bA,Al,ilbB,aBjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovoo"] +=  0.250 * np.einsum("BA,Al,ilBC,aCjk->iajk",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovoo"] += -0.125 * np.einsum("aA,Al,ilBC,BCjk->iajk",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["ovoo"] = asym_term(fs1s2s2["ovoo"],"ovoo")

	fs1s2s2["ovov"] += 0.500 * np.einsum("Ak,lA,ikaB,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovov"] += 0.500 * np.einsum("Ak,kB,ilaA,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovov"] += 0.500 * np.einsum("kA,Bk,ilaB,bAjl->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovov"] += 0.500 * np.einsum("kA,Al,ilaB,bBjk->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovov"] += 0.250 * np.einsum("Ai,kA,jlaB,bBkl->jbia",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovov"] += 0.250 * np.einsum("ak,kA,ilbB,ABjl->iajb",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs1s2s2["ovov"] += 0.250 * np.einsum("Ak,iA,klaB,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovov"] += 0.250 * np.einsum("ka,Ak,ilAB,bBjl->ibja",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovov"] += 0.250 * np.einsum("iA,Ak,klaB,bBjl->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovov"] += 0.250 * np.einsum("kA,Ai,jlaB,bBkl->jbia",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	fs1s2s2["ovvv"] += 0.500 * np.einsum("Aa,jA,ikbB,cBjk->icab",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["ovvv"] = asym_term(fs1s2s2["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	fs1s2s2["vvov"] += 0.500 * np.einsum("aA,Aj,jkbB,cBik->acib",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs1s2s2["vvov"] = asym_term(fs1s2s2["vvov"],"vvov")

	if(inc_3_body):
		fs1s2s2["ooooov"] = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs1s2s2["oovooo"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs1s2s2["oovoov"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs1s2s2["oovovv"] = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs1s2s2["ovvoov"] = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs1s2s2["ooooov"] +=  (1./12.) * np.einsum("in,nA,jkaB,ABlm->ijklma",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs1s2s2["ooooov"] += -(1./12.) * np.einsum("Ab,iA,jkaB,bBlm->ijklma",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs1s2s2["ooooov"] += -(1./12.) * np.einsum("BA,iB,jkaC,AClm->ijklma",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs1s2s2["ooooov"] = asym_term(fs1s2s2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs1s2s2["oovooo"] +=  (1./12.) * np.einsum("ni,An,jkAB,aBlm->jkailm",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs1s2s2["oovooo"] += -(1./12.) * np.einsum("bA,Ai,jkbB,aBlm->jkailm",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs1s2s2["oovooo"] += -(1./12.) * np.einsum("BA,Ai,jkBC,aClm->jkailm",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs1s2s2["oovooo"] = asym_term(fs1s2s2["oovooo"],"oovooo")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs1s2s2["oovoov"] += -(1./18.) * np.einsum("Ai,mA,jkaB,bBlm->jkbila",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs1s2s2["oovoov"] += -(1./18.) * np.einsum("Am,iA,jmaB,bBkl->ijbkla",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs1s2s2["oovoov"] +=  (1./18.) * np.einsum("Am,mB,ijaA,bBkl->ijbkla",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs1s2s2["oovoov"] += -(1./18.) * np.einsum("iA,Am,jmaB,bBkl->ijbkla",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs1s2s2["oovoov"] += -(1./18.) * np.einsum("mA,Ai,jkaB,bBlm->jkbila",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs1s2s2["oovoov"] +=  (1./18.) * np.einsum("mA,Bm,ijaB,bAkl->ijbkla",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs1s2s2["oovoov"] +=  (1./36.) * np.einsum("am,mA,ijbB,ABkl->ijaklb",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs1s2s2["oovoov"] +=  (1./36.) * np.einsum("ma,Am,ijAB,bBkl->ijbkla",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs1s2s2["oovoov"] = asym_term(fs1s2s2["oovoov"],"oovoov")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs1s2s2["oovovv"] +=  (1./18.) * np.einsum("Aa,lA,ijbB,cBkl->ijckab",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs1s2s2["oovovv"] = asym_term(fs1s2s2["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		fs1s2s2["ovvoov"] +=  (1./18.) * np.einsum("aA,Al,ilbB,cBjk->iacjkb",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
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
	fs2s1s1["c"] +=  2.000 * np.einsum("ji,Ak,Bj,ikAB->",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1s1["c"] +=  2.000 * np.einsum("ji,kA,iB,ABjk->",f["oo"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs2s1s1["c"] += -2.000 * np.einsum("Aa,iB,jA,aBij->",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1s1["c"] += -2.000 * np.einsum("aA,Bi,Aj,ijaB->",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1s1["c"] += -2.000 * np.einsum("BA,Ci,Aj,ijBC->",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1s1["c"] += -2.000 * np.einsum("BA,iC,jB,ACij->",f["VV"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")

	fs2s1s1["oo"] += -2.000 * np.einsum("ki,jA,lB,ABkl->ji",f["oo"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs2s1s1["oo"] += -2.000 * np.einsum("ik,Aj,Bl,klAB->ij",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1s1["oo"] += -2.000 * np.einsum("lk,Ai,Bl,jkAB->ji",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1s1["oo"] += -2.000 * np.einsum("lk,iA,kB,ABjl->ij",f["oo"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs2s1s1["oo"] += -2.000 * np.einsum("Aa,iB,kA,aBjk->ij",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1s1["oo"] +=  2.000 * np.einsum("Aa,kB,iA,aBjk->ij",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1s1["oo"] +=  2.000 * np.einsum("aA,Ai,Bk,jkaB->ji",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1s1["oo"] += -2.000 * np.einsum("aA,Bi,Ak,jkaB->ji",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1s1["oo"] +=  2.000 * np.einsum("BA,Ai,Ck,jkBC->ji",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1s1["oo"] += -2.000 * np.einsum("BA,Ci,Ak,jkBC->ji",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1s1["oo"] += -2.000 * np.einsum("BA,iC,kB,ACjk->ij",f["VV"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs2s1s1["oo"] +=  2.000 * np.einsum("BA,kC,iB,ACjk->ij",f["VV"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")

	fs2s1s1["ov"] += -2.000 * np.einsum("Aj,iA,Bk,jkaB->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1s1["ov"] += -2.000 * np.einsum("Aj,kA,Bk,ijaB->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1s1["ov"] +=  1.000 * np.einsum("Aj,iB,Bk,jkaA->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1s1["ov"] += -2.000 * np.einsum("Aj,jB,Bk,ikaA->ia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1s1["ov"] += -2.000 * np.einsum("ja,iA,kB,ABjk->ia",f["ov"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")

	fs2s1s1["vo"] += -2.000 * np.einsum("aj,Ai,Bk,jkAB->ai",f["vo"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1s1["vo"] += -2.000 * np.einsum("jA,kB,Ai,aBjk->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs2s1s1["vo"] +=  1.000 * np.einsum("jA,kB,Bi,aAjk->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs2s1s1["vo"] += -2.000 * np.einsum("jA,kB,Bj,aAik->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs2s1s1["vo"] += -2.000 * np.einsum("jA,kB,Ak,aBij->ai",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs2s1s1["oooo"] +=  1.000 * np.einsum("mi,jA,kB,ABlm->jkil",f["oo"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs2s1s1["oooo"] +=  1.000 * np.einsum("im,Aj,Bk,lmAB->iljk",f["oo"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1s1["oooo"] += -1.000 * np.einsum("Aa,iB,jA,aBkl->ijkl",f["Vv"],t1["oV"],t1["oV"],t2["vVoo"],optimize="optimal")
	fs2s1s1["oooo"] += -1.000 * np.einsum("aA,Bi,Aj,klaB->klij",f["vV"],t1["Vo"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1s1["oooo"] += -1.000 * np.einsum("BA,Ci,Aj,klBC->klij",f["VV"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1s1["oooo"] += -1.000 * np.einsum("BA,iC,jB,ACkl->ijkl",f["VV"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs2s1s1["oooo"] = asym_term(fs2s1s1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2s1s1["ooov"] += -1.000 * np.einsum("Al,iA,Bj,klaB->ikja",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1s1["ooov"] += -0.500 * np.einsum("Ai,jB,Bl,klaA->jkia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1s1["ooov"] += -0.500 * np.einsum("la,iA,jB,ABkl->ijka",f["ov"],t1["oV"],t1["oV"],t2["VVoo"],optimize="optimal")
	fs2s1s1["ooov"] +=  0.250 * np.einsum("Al,lB,Bi,jkaA->jkia",f["Vo"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1s1["ooov"] = asym_term(fs2s1s1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2s1s1["ovoo"] += -1.000 * np.einsum("lA,iB,Aj,aBkl->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs2s1s1["ovoo"] += -0.500 * np.einsum("al,Ai,Bj,klAB->kaij",f["vo"],t1["Vo"],t1["Vo"],t2["ooVV"],optimize="optimal")
	fs2s1s1["ovoo"] += -0.500 * np.einsum("iA,lB,Bj,aAkl->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs2s1s1["ovoo"] +=  0.250 * np.einsum("lA,iB,Bl,aAjk->iajk",f["oV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
	fs2s1s1["ovoo"] = asym_term(fs2s1s1["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	fs2s1s1["oovv"] += -1.000 * np.einsum("Aa,iB,Bk,jkbA->ijab",f["Vv"],t1["oV"],t1["Vo"],t2["oovV"],optimize="optimal")
	fs2s1s1["oovv"] = asym_term(fs2s1s1["oovv"],"oovv")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	fs2s1s1["vvoo"] += -1.000 * np.einsum("aA,kB,Bi,bAjk->abij",f["vV"],t1["oV"],t1["Vo"],t2["vVoo"],optimize="optimal")
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
	fs2s1s2["c"] += -0.500 * np.einsum("ai,iA,jkaB,ABjk->",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["c"] += -1.000 * np.einsum("Ai,jA,ikaB,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["c"] += -0.500 * np.einsum("Ai,jA,ikBC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["c"] += -0.500 * np.einsum("Ai,iB,jkaA,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["c"] += -0.500 * np.einsum("Ai,iB,jkAC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["c"] += -0.500 * np.einsum("ia,Ai,jkAB,aBjk->",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["c"] += -0.500 * np.einsum("iA,Bi,jkaB,aAjk->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["c"] += -0.500 * np.einsum("iA,Bi,jkBC,ACjk->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["c"] += -1.000 * np.einsum("iA,Aj,jkaB,aBik->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["c"] += -0.500 * np.einsum("iA,Aj,jkBC,BCik->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")

	fs2s1s2["oo"] += -1.000 * np.einsum("ak,kA,ilaB,ABjl->ij",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oo"] += -1.000 * np.einsum("Ak,iA,klaB,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["oo"] += -0.500 * np.einsum("Ak,iA,klBC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oo"] += -1.000 * np.einsum("Ak,lA,ikaB,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["oo"] += -0.500 * np.einsum("Ak,lA,ikBC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oo"] += -1.000 * np.einsum("Ak,kB,ilaA,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["oo"] += -1.000 * np.einsum("Ak,kB,ilAC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oo"] += -1.000 * np.einsum("ka,Ak,ilAB,aBjl->ij",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["oo"] += -1.000 * np.einsum("kA,Ai,jlaB,aBkl->ji",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["oo"] += -0.500 * np.einsum("kA,Ai,jlBC,BCkl->ji",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oo"] += -1.000 * np.einsum("kA,Bk,ilaB,aAjl->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["oo"] += -1.000 * np.einsum("kA,Bk,ilBC,ACjl->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oo"] += -1.000 * np.einsum("kA,Al,ilaB,aBjk->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["oo"] += -0.500 * np.einsum("kA,Al,ilBC,BCjk->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")

	fs2s1s2["ov"] +=  1.000 * np.einsum("kj,iA,jlaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ov"] += -1.000 * np.einsum("kj,jA,ilaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ov"] +=  1.000 * np.einsum("kj,lA,ijaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ov"] += -0.500 * np.einsum("Ab,iA,jkaB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ov"] +=  1.000 * np.einsum("Ab,jA,ikaB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ov"] +=  0.500 * np.einsum("Ab,iB,jkaA,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ov"] += -1.000 * np.einsum("Ab,jB,ikaA,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ov"] += -0.500 * np.einsum("BA,iB,jkaC,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ov"] +=  1.000 * np.einsum("BA,jB,ikaC,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ov"] +=  0.500 * np.einsum("BA,iC,jkaB,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ov"] += -1.000 * np.einsum("BA,jC,ikaB,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")

	fs2s1s2["vo"] +=  1.000 * np.einsum("kj,Ai,jlAB,aBkl->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vo"] += -1.000 * np.einsum("kj,Ak,jlAB,aBil->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vo"] +=  1.000 * np.einsum("kj,Al,jlAB,aBik->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vo"] += -0.500 * np.einsum("bA,Ai,jkbB,aBjk->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vo"] +=  0.500 * np.einsum("bA,Bi,jkbB,aAjk->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vo"] +=  1.000 * np.einsum("bA,Aj,jkbB,aBik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vo"] += -1.000 * np.einsum("bA,Bj,jkbB,aAik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vo"] += -0.500 * np.einsum("BA,Ai,jkBC,aCjk->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vo"] +=  0.500 * np.einsum("BA,Ci,jkBC,aAjk->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vo"] +=  1.000 * np.einsum("BA,Aj,jkBC,aCik->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vo"] += -1.000 * np.einsum("BA,Cj,jkBC,aAik->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")

	fs2s1s2["vv"] += 1.000 * np.einsum("Ai,jA,ikaB,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vv"] += 0.500 * np.einsum("Ai,iB,jkaA,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vv"] += 0.500 * np.einsum("iA,Bi,jkaB,bAjk->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vv"] += 1.000 * np.einsum("iA,Aj,jkaB,bBik->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs2s1s2["oooo"] +=  0.500 * np.einsum("Am,iA,jmaB,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["oooo"] +=  0.500 * np.einsum("mA,Ai,jkaB,aBlm->jkil",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["oooo"] += -0.250 * np.einsum("am,mA,ijaB,ABkl->ijkl",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oooo"] +=  0.250 * np.einsum("Am,iA,jmBC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oooo"] += -0.250 * np.einsum("Am,mB,ijaA,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["oooo"] += -0.250 * np.einsum("Am,mB,ijAC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oooo"] += -0.250 * np.einsum("ma,Am,ijAB,aBkl->ijkl",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["oooo"] +=  0.250 * np.einsum("mA,Ai,jkBC,BClm->jkil",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oooo"] += -0.250 * np.einsum("mA,Bm,ijaB,aAkl->ijkl",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["oooo"] += -0.250 * np.einsum("mA,Bm,ijBC,ACkl->ijkl",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oooo"] = asym_term(fs2s1s2["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2s1s2["ooov"] += -0.500 * np.einsum("li,jA,kmaB,ABlm->jkia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ooov"] += -0.500 * np.einsum("ml,iA,jlaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ooov"] +=  0.500 * np.einsum("Ab,iA,jlaB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ooov"] += -0.500 * np.einsum("Ab,iB,jlaA,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ooov"] +=  0.500 * np.einsum("BA,iB,jlaC,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ooov"] += -0.500 * np.einsum("BA,iC,jlaB,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ooov"] += -0.250 * np.einsum("li,mA,jkaB,ABlm->jkia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ooov"] += -0.250 * np.einsum("ml,lA,ijaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ooov"] +=  0.250 * np.einsum("Ab,lA,ijaB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ooov"] += -0.250 * np.einsum("Ab,lB,ijaA,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ooov"] +=  0.250 * np.einsum("BA,lB,ijaC,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ooov"] += -0.250 * np.einsum("BA,lC,ijaB,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["ooov"] = asym_term(fs2s1s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2s1s2["ovoo"] += -0.500 * np.einsum("il,Aj,lmAB,aBkm->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovoo"] += -0.500 * np.einsum("ml,Ai,jlAB,aBkm->jaik",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovoo"] +=  0.500 * np.einsum("bA,Ai,jlbB,aBkl->jaik",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovoo"] += -0.500 * np.einsum("bA,Bi,jlbB,aAkl->jaik",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovoo"] +=  0.500 * np.einsum("BA,Ai,jlBC,aCkl->jaik",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovoo"] += -0.500 * np.einsum("BA,Ci,jlBC,aAkl->jaik",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovoo"] += -0.250 * np.einsum("il,Am,lmAB,aBjk->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovoo"] += -0.250 * np.einsum("ml,Am,ilAB,aBjk->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovoo"] +=  0.250 * np.einsum("bA,Al,ilbB,aBjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovoo"] += -0.250 * np.einsum("bA,Bl,ilbB,aAjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovoo"] +=  0.250 * np.einsum("BA,Al,ilBC,aCjk->iajk",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovoo"] += -0.250 * np.einsum("BA,Cl,ilBC,aAjk->iajk",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovoo"] = asym_term(fs2s1s2["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	fs2s1s2["oovv"] += -2.000 * np.einsum("Ak,Bl,ikaB,jlbA->ijab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"],optimize="optimal")
	fs2s1s2["oovv"] += -1.000 * np.einsum("ka,iA,jlbB,ABkl->ijab",f["ov"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oovv"] +=  0.500 * np.einsum("Ak,Bl,ijaB,klbA->ijab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"],optimize="optimal")
	fs2s1s2["oovv"] +=  0.500 * np.einsum("Ak,Bl,klaB,ijbA->ijab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"],optimize="optimal")
	fs2s1s2["oovv"] += -0.500 * np.einsum("ka,lA,ijbB,ABkl->ijab",f["ov"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s1s2["oovv"] = asym_term(fs2s1s2["oovv"],"oovv")

	fs2s1s2["ovov"] += 0.250 * np.einsum("Ak,iA,klaB,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovov"] += 0.250 * np.einsum("Ak,lA,ikaB,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovov"] += 0.250 * np.einsum("Ak,kB,ilaA,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovov"] += 0.250 * np.einsum("kA,Ai,jlaB,bBkl->jbia",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovov"] += 0.250 * np.einsum("kA,Bk,ilaB,bAjl->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["ovov"] += 0.250 * np.einsum("kA,Al,ilaB,bBjk->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	fs2s1s2["vvoo"] += -2.000 * np.einsum("kA,lB,aAil,bBjk->abij",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vvoo"] += -1.000 * np.einsum("ak,Ai,klAB,bBjl->abij",f["vo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vvoo"] += -0.500 * np.einsum("ak,Al,klAB,bBij->abij",f["vo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vvoo"] += -0.500 * np.einsum("kA,lB,aAij,bBkl->abij",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s1s2["vvoo"] +=  0.500 * np.einsum("kA,lB,aBij,bAkl->abij",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
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
		fs2s1s2["ooooov"] += -(1./6.) * np.einsum("ni,jA,klaB,ABmn->jklima",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s1s2["ooooov"] += -(1./12.) * np.einsum("Ab,iA,jkaB,bBlm->ijklma",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s1s2["ooooov"] +=  (1./12.) * np.einsum("Ab,iB,jkaA,bBlm->ijklma",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s1s2["ooooov"] += -(1./12.) * np.einsum("BA,iB,jkaC,AClm->ijklma",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s1s2["ooooov"] +=  (1./12.) * np.einsum("BA,iC,jkaB,AClm->ijklma",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s1s2["ooooov"] = asym_term(fs2s1s2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s1s2["oovooo"] += -(1./6.) * np.einsum("in,Aj,knAB,aBlm->ikajlm",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s1s2["oovooo"] += -(1./12.) * np.einsum("bA,Ai,jkbB,aBlm->jkailm",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s1s2["oovooo"] +=  (1./12.) * np.einsum("bA,Bi,jkbB,aAlm->jkailm",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s1s2["oovooo"] += -(1./12.) * np.einsum("BA,Ai,jkBC,aClm->jkailm",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s1s2["oovooo"] +=  (1./12.) * np.einsum("BA,Ci,jkBC,aAlm->jkailm",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s1s2["oovooo"] = asym_term(fs2s1s2["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s1s2["oooovv"] +=  (1./6.) * np.einsum("Ai,Bm,jkaB,lmbA->jkliab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"],optimize="optimal")
		fs2s1s2["oooovv"] += -(1./6.) * np.einsum("Am,Bi,jmaB,klbA->jkliab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"],optimize="optimal")
		fs2s1s2["oooovv"] +=  (1./6.) * np.einsum("ma,iA,jkbB,ABlm->ijklab",f["ov"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s1s2["oooovv"] = asym_term(fs2s1s2["oooovv"],"oooovv")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s1s2["oovoov"] += -(1./18.) * np.einsum("Am,iA,jmaB,bBkl->ijbkla",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s1s2["oovoov"] += -(1./18.) * np.einsum("mA,Ai,jkaB,bBlm->jkbila",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s1s2["oovoov"] +=  (1./36.) * np.einsum("Am,mB,ijaA,bBkl->ijbkla",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s1s2["oovoov"] +=  (1./36.) * np.einsum("mA,Bm,ijaB,bAkl->ijbkla",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s1s2["oovoov"] = asym_term(fs2s1s2["oovoov"],"oovoov")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s1s2["ovvooo"] +=  (1./6.) * np.einsum("am,Ai,jmAB,bBkl->jabikl",f["vo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s1s2["ovvooo"] +=  (1./6.) * np.einsum("iA,mB,aBjk,bAlm->iabjkl",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s1s2["ovvooo"] +=  (1./6.) * np.einsum("mA,iB,aAjk,bBlm->iabjkl",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s1s2["ovvooo"] = asym_term(fs2s1s2["ovvooo"],"ovvooo")

		# ooovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		fs2s1s2["ooovvv"] +=  0.500 * np.einsum("Aa,Bl,ijbB,klcA->ijkabc",f["Vv"],t1["Vo"],t2["oovV"],t2["oovV"],optimize="optimal")
		fs2s1s2["ooovvv"] = asym_term(fs2s1s2["ooovvv"],"ooovvv")

		# vvvooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s1s2["vvvooo"] +=  0.500 * np.einsum("aA,lB,bBij,cAkl->abcijk",f["vV"],t1["oV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
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
	fs2s2s1["c"] += -0.500 * np.einsum("ai,iA,jkaB,ABjk->",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["c"] +=  1.000 * np.einsum("ai,jA,ikaB,ABjk->",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["c"] += -1.000 * np.einsum("Ai,jA,ikaB,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["c"] += -0.500 * np.einsum("Ai,jA,ikBC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["c"] += -0.500 * np.einsum("Ai,iB,jkaA,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["c"] += -0.500 * np.einsum("Ai,iB,jkAC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["c"] +=  1.000 * np.einsum("Ai,jB,ikaA,aBjk->",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["c"] +=  1.000 * np.einsum("Ai,jB,ikAC,BCjk->",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["c"] += -0.500 * np.einsum("ia,Ai,jkAB,aBjk->",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["c"] +=  1.000 * np.einsum("ia,Aj,jkAB,aBik->",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["c"] += -0.500 * np.einsum("iA,Bi,jkaB,aAjk->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["c"] += -0.500 * np.einsum("iA,Bi,jkBC,ACjk->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["c"] += -1.000 * np.einsum("iA,Aj,jkaB,aBik->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["c"] += -0.500 * np.einsum("iA,Aj,jkBC,BCik->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["c"] +=  1.000 * np.einsum("iA,Bj,jkaB,aAik->",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["c"] +=  1.000 * np.einsum("iA,Bj,jkBC,ACik->",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")

	fs2s2s1["oo"] += -0.500 * np.einsum("ai,jA,klaB,ABkl->ji",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("ai,kA,jlaB,ABkl->ji",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] += -0.500 * np.einsum("Ai,jB,klaA,aBkl->ji",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] += -0.500 * np.einsum("Ai,jB,klAC,BCkl->ji",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("Ai,kB,jlaA,aBkl->ji",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("Ai,kB,jlAC,BCkl->ji",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("ak,iA,klaB,ABjl->ij",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] += -1.000 * np.einsum("ak,kA,ilaB,ABjl->ij",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("ak,lA,ikaB,ABjl->ij",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] += -1.000 * np.einsum("Ak,iA,klaB,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] += -0.500 * np.einsum("Ak,iA,klBC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] += -1.000 * np.einsum("Ak,lA,ikaB,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] += -0.500 * np.einsum("Ak,lA,ikBC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("Ak,iB,klaA,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("Ak,iB,klAC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] += -1.000 * np.einsum("Ak,kB,ilaA,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] += -1.000 * np.einsum("Ak,kB,ilAC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("Ak,lB,ikaA,aBjl->ij",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("Ak,lB,ikAC,BCjl->ij",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] += -0.500 * np.einsum("ia,Aj,klAB,aBkl->ij",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("ia,Ak,klAB,aBjl->ij",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("ka,Ai,jlAB,aBkl->ji",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] += -1.000 * np.einsum("ka,Ak,ilAB,aBjl->ij",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("ka,Al,ilAB,aBjk->ij",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] += -0.500 * np.einsum("iA,Bj,klaB,aAkl->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] += -0.500 * np.einsum("iA,Bj,klBC,ACkl->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("iA,Bk,klaB,aAjl->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("iA,Bk,klBC,ACjl->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] += -1.000 * np.einsum("kA,Ai,jlaB,aBkl->ji",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] += -0.500 * np.einsum("kA,Ai,jlBC,BCkl->ji",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("kA,Bi,jlaB,aAkl->ji",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("kA,Bi,jlBC,ACkl->ji",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] += -1.000 * np.einsum("kA,Bk,ilaB,aAjl->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] += -1.000 * np.einsum("kA,Bk,ilBC,ACjl->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] += -1.000 * np.einsum("kA,Al,ilaB,aBjk->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] += -0.500 * np.einsum("kA,Al,ilBC,BCjk->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("kA,Bl,ilaB,aAjk->ij",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oo"] +=  1.000 * np.einsum("kA,Bl,ilBC,ACjk->ij",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")

	fs2s2s1["ov"] += -1.000 * np.einsum("ij,kA,jlaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ov"] +=  2.000 * np.einsum("kj,iA,jlaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ov"] += -1.000 * np.einsum("kj,jA,ilaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ov"] +=  2.000 * np.einsum("kj,lA,ijaB,ABkl->ia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ov"] += -0.500 * np.einsum("ba,iA,jkbB,ABjk->ia",f["vv"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ov"] +=  1.000 * np.einsum("ba,jA,ikbB,ABjk->ia",f["vv"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ov"] += -0.500 * np.einsum("Aa,iB,jkbA,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ov"] += -0.500 * np.einsum("Aa,iB,jkAC,BCjk->ia",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ov"] +=  1.000 * np.einsum("Aa,jB,ikbA,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ov"] +=  1.000 * np.einsum("Aa,jB,ikAC,BCjk->ia",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ov"] += -0.500 * np.einsum("Ab,iA,jkaB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ov"] +=  1.000 * np.einsum("Ab,jA,ikaB,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ov"] +=  1.000 * np.einsum("Ab,iB,jkaA,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ov"] += -2.000 * np.einsum("Ab,jB,ikaA,bBjk->ia",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ov"] += -0.500 * np.einsum("BA,iB,jkaC,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ov"] +=  1.000 * np.einsum("BA,jB,ikaC,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ov"] +=  1.000 * np.einsum("BA,iC,jkaB,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ov"] += -2.000 * np.einsum("BA,jC,ikaB,ACjk->ia",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")

	fs2s2s1["vo"] += -1.000 * np.einsum("ji,Ak,klAB,aBjl->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] +=  2.000 * np.einsum("kj,Ai,jlAB,aBkl->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] += -1.000 * np.einsum("kj,Ak,jlAB,aBil->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] +=  2.000 * np.einsum("kj,Al,jlAB,aBik->ai",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] += -0.500 * np.einsum("ab,Ai,jkAB,bBjk->ai",f["vv"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] +=  1.000 * np.einsum("ab,Aj,jkAB,bBik->ai",f["vv"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] += -0.500 * np.einsum("aA,Bi,jkbB,bAjk->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] += -0.500 * np.einsum("aA,Bi,jkBC,ACjk->ai",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["vo"] +=  1.000 * np.einsum("aA,Bj,jkbB,bAik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] +=  1.000 * np.einsum("aA,Bj,jkBC,ACik->ai",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["vo"] += -0.500 * np.einsum("bA,Ai,jkbB,aBjk->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] +=  1.000 * np.einsum("bA,Bi,jkbB,aAjk->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] +=  1.000 * np.einsum("bA,Aj,jkbB,aBik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] += -2.000 * np.einsum("bA,Bj,jkbB,aAik->ai",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] += -0.500 * np.einsum("BA,Ai,jkBC,aCjk->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] +=  1.000 * np.einsum("BA,Ci,jkBC,aAjk->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] +=  1.000 * np.einsum("BA,Aj,jkBC,aCik->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vo"] += -2.000 * np.einsum("BA,Cj,jkBC,aAik->ai",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")

	fs2s2s1["vv"] += -1.000 * np.einsum("ai,jA,ikbB,ABjk->ab",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["vv"] +=  1.000 * np.einsum("Ai,jA,ikaB,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vv"] +=  0.500 * np.einsum("Ai,iB,jkaA,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vv"] += -1.000 * np.einsum("Ai,jB,ikaA,bBjk->ba",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vv"] += -1.000 * np.einsum("ia,Aj,jkAB,bBik->ba",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vv"] +=  0.500 * np.einsum("iA,Bi,jkaB,bAjk->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vv"] +=  1.000 * np.einsum("iA,Aj,jkaB,bBik->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vv"] += -1.000 * np.einsum("iA,Bj,jkaB,bAik->ba",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")

	# oooo = np.zeros((n_occ,n_occ,n_occ,n_occ))
	fs2s2s1["oooo"] += -1.000 * np.einsum("ai,jA,kmaB,ABlm->jkil",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -1.000 * np.einsum("Ai,jB,kmaA,aBlm->jkil",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -1.000 * np.einsum("Ai,jB,kmAC,BClm->jkil",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -1.000 * np.einsum("ia,Aj,kmAB,aBlm->ikjl",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -1.000 * np.einsum("iA,Bj,kmaB,aAlm->ikjl",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -1.000 * np.einsum("iA,Bj,kmBC,AClm->ikjl",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.500 * np.einsum("ai,mA,jkaB,ABlm->jkil",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.500 * np.einsum("Ai,mB,jkaA,aBlm->jkil",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.500 * np.einsum("Ai,mB,jkAC,BClm->jkil",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.500 * np.einsum("am,iA,jmaB,ABkl->ijkl",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] +=  0.500 * np.einsum("Am,iA,jmaB,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.500 * np.einsum("Am,iB,jmaA,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.500 * np.einsum("Am,iB,jmAC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.500 * np.einsum("ia,Am,jmAB,aBkl->ijkl",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.500 * np.einsum("ma,Ai,jkAB,aBlm->jkil",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.500 * np.einsum("iA,Bm,jmaB,aAkl->ijkl",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.500 * np.einsum("iA,Bm,jmBC,ACkl->ijkl",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] +=  0.500 * np.einsum("mA,Ai,jkaB,aBlm->jkil",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.500 * np.einsum("mA,Bi,jkaB,aAlm->jkil",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.500 * np.einsum("mA,Bi,jkBC,AClm->jkil",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.250 * np.einsum("am,mA,ijaB,ABkl->ijkl",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] +=  0.250 * np.einsum("Am,iA,jmBC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.250 * np.einsum("Am,mB,ijaA,aBkl->ijkl",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.250 * np.einsum("Am,mB,ijAC,BCkl->ijkl",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.250 * np.einsum("ma,Am,ijAB,aBkl->ijkl",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] +=  0.250 * np.einsum("mA,Ai,jkBC,BClm->jkil",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.250 * np.einsum("mA,Bm,ijaB,aAkl->ijkl",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["oooo"] += -0.250 * np.einsum("mA,Bm,ijBC,ACkl->ijkl",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oooo"] = asym_term(fs2s2s1["oooo"],"oooo")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2s2s1["ooov"] += -1.000 * np.einsum("ml,iA,jlaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] += -1.000 * np.einsum("Ab,iB,jlaA,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ooov"] += -1.000 * np.einsum("BA,iC,jlaB,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] += -0.500 * np.einsum("li,jA,kmaB,ABlm->jkia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] +=  0.500 * np.einsum("il,jA,lmaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] +=  0.500 * np.einsum("il,mA,jlaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] +=  0.500 * np.einsum("ba,iA,jlbB,ABkl->ijka",f["vv"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] +=  0.500 * np.einsum("Aa,iB,jlbA,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ooov"] +=  0.500 * np.einsum("Aa,iB,jlAC,BCkl->ijka",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] +=  0.500 * np.einsum("Ab,iA,jlaB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ooov"] += -0.500 * np.einsum("Ab,lB,ijaA,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ooov"] +=  0.500 * np.einsum("BA,iB,jlaC,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] += -0.500 * np.einsum("BA,lC,ijaB,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] += -0.250 * np.einsum("li,mA,jkaB,ABlm->jkia",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] += -0.250 * np.einsum("ml,lA,ijaB,ABkm->ijka",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] +=  0.250 * np.einsum("ba,lA,ijbB,ABkl->ijka",f["vv"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] +=  0.250 * np.einsum("Aa,lB,ijbA,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ooov"] +=  0.250 * np.einsum("Aa,lB,ijAC,BCkl->ijka",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] +=  0.250 * np.einsum("Ab,lA,ijaB,bBkl->ijka",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ooov"] +=  0.250 * np.einsum("BA,lB,ijaC,ACkl->ijka",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ooov"] = asym_term(fs2s2s1["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2s2s1["ovoo"] += -1.000 * np.einsum("ml,Ai,jlAB,aBkm->jaik",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] += -1.000 * np.einsum("bA,Bi,jlbB,aAkl->jaik",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] += -1.000 * np.einsum("BA,Ci,jlBC,aAkl->jaik",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] +=  0.500 * np.einsum("li,Aj,kmAB,aBlm->kaij",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] +=  0.500 * np.einsum("li,Am,jmAB,aBkl->jaik",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] += -0.500 * np.einsum("il,Aj,lmAB,aBkm->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] +=  0.500 * np.einsum("ab,Ai,jlAB,bBkl->jaik",f["vv"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] +=  0.500 * np.einsum("aA,Bi,jlbB,bAkl->jaik",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] +=  0.500 * np.einsum("aA,Bi,jlBC,ACkl->jaik",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ovoo"] +=  0.500 * np.einsum("bA,Ai,jlbB,aBkl->jaik",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] += -0.500 * np.einsum("bA,Bl,ilbB,aAjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] +=  0.500 * np.einsum("BA,Ai,jlBC,aCkl->jaik",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] += -0.500 * np.einsum("BA,Cl,ilBC,aAjk->iajk",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] += -0.250 * np.einsum("il,Am,lmAB,aBjk->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] += -0.250 * np.einsum("ml,Am,ilAB,aBjk->iajk",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] +=  0.250 * np.einsum("ab,Al,ilAB,bBjk->iajk",f["vv"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] +=  0.250 * np.einsum("aA,Bl,ilbB,bAjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] +=  0.250 * np.einsum("aA,Bl,ilBC,ACjk->iajk",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ovoo"] +=  0.250 * np.einsum("bA,Al,ilbB,aBjk->iajk",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] +=  0.250 * np.einsum("BA,Al,ilBC,aCjk->iajk",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovoo"] = asym_term(fs2s2s1["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	fs2s2s1["oovv"] += -2.000 * np.einsum("Ak,Bl,ikaB,jlbA->ijab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"],optimize="optimal")
	fs2s2s1["oovv"] +=  1.000 * np.einsum("Ak,Bl,klaB,ijbA->ijab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"],optimize="optimal")
	fs2s2s1["oovv"] += -1.000 * np.einsum("ka,iA,jlbB,ABkl->ijab",f["ov"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oovv"] += -0.500 * np.einsum("ka,lA,ijbB,ABkl->ijab",f["ov"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["oovv"] = asym_term(fs2s2s1["oovv"],"oovv")

	fs2s2s1["ovov"] += -0.250 * np.einsum("Ai,kB,jlaA,bBkl->jbia",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] += -0.250 * np.einsum("ak,iA,klbB,ABjl->iajb",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ovov"] += -0.250 * np.einsum("ak,lA,ikbB,ABjl->iajb",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s1["ovov"] +=  0.250 * np.einsum("Ak,iA,klaB,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] +=  0.250 * np.einsum("Ak,lA,ikaB,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] += -0.250 * np.einsum("Ak,iB,klaA,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] +=  0.250 * np.einsum("Ak,kB,ilaA,bBjl->ibja",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] += -0.250 * np.einsum("ka,Ai,jlAB,bBkl->jbia",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] += -0.250 * np.einsum("ka,Al,ilAB,bBjk->ibja",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] += -0.250 * np.einsum("iA,Bk,klaB,bAjl->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] +=  0.250 * np.einsum("kA,Ai,jlaB,bBkl->jbia",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] += -0.250 * np.einsum("kA,Bi,jlaB,bAkl->jbia",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] +=  0.250 * np.einsum("kA,Bk,ilaB,bAjl->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] +=  0.250 * np.einsum("kA,Al,ilaB,bBjk->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] +=  0.125 * np.einsum("Ai,jB,klaA,bBkl->jbia",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovov"] +=  0.125 * np.einsum("iA,Bj,klaB,bAkl->ibja",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	fs2s2s1["vvoo"] += -2.000 * np.einsum("kA,lB,aAil,bBjk->abij",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vvoo"] += -1.000 * np.einsum("ak,Ai,klAB,bBjl->abij",f["vo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vvoo"] += -1.000 * np.einsum("kA,lB,aAij,bBkl->abij",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vvoo"] += -0.500 * np.einsum("ak,Al,klAB,bBij->abij",f["vo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vvoo"] = asym_term(fs2s2s1["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	fs2s2s1["ovvv"] += -0.500 * np.einsum("Aa,jB,ikbA,cBjk->icab",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovvv"] +=  0.250 * np.einsum("Aa,iB,jkbA,cBjk->icab",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["ovvv"] = asym_term(fs2s2s1["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	fs2s2s1["vvov"] += -0.500 * np.einsum("aA,Bj,jkbB,cAik->acib",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s1["vvov"] +=  0.250 * np.einsum("aA,Bi,jkbB,cAjk->acib",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
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
		fs2s2s1["oooooo"] += -0.250 * np.einsum("ai,jA,klaB,ABmn->jklimn",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s1["oooooo"] += -0.250 * np.einsum("Ai,jB,klaA,aBmn->jklimn",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oooooo"] += -0.250 * np.einsum("Ai,jB,klAC,BCmn->jklimn",f["Vo"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
		fs2s2s1["oooooo"] += -0.250 * np.einsum("ia,Aj,klAB,aBmn->ikljmn",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oooooo"] += -0.250 * np.einsum("iA,Bj,klaB,aAmn->ikljmn",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oooooo"] += -0.250 * np.einsum("iA,Bj,klBC,ACmn->ikljmn",f["oV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
		fs2s2s1["oooooo"] = asym_term(fs2s2s1["oooooo"],"oooooo")

		# ooooov = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2s1["ooooov"] += -(1./6.) * np.einsum("ni,jA,klaB,ABmn->jklima",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s1["ooooov"] +=  (1./6.) * np.einsum("in,jA,knaB,ABlm->ijklma",f["oo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s1["ooooov"] +=  (1./6.) * np.einsum("Ab,iB,jkaA,bBlm->ijklma",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["ooooov"] +=  (1./6.) * np.einsum("BA,iC,jkaB,AClm->ijklma",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s1["ooooov"] += -(1./12.) * np.einsum("ba,iA,jkbB,ABlm->ijklma",f["vv"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s1["ooooov"] += -(1./12.) * np.einsum("Aa,iB,jkbA,bBlm->ijklma",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["ooooov"] += -(1./12.) * np.einsum("Aa,iB,jkAC,BClm->ijklma",f["Vv"],t1["oV"],t2["ooVV"],t2["VVoo"],optimize="optimal")
		fs2s2s1["ooooov"] += -(1./12.) * np.einsum("Ab,iA,jkaB,bBlm->ijklma",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["ooooov"] += -(1./12.) * np.einsum("BA,iB,jkaC,AClm->ijklma",f["VV"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s1["ooooov"] = asym_term(fs2s2s1["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s1["oovooo"] +=  (1./6.) * np.einsum("ni,Aj,klAB,aBmn->klaijm",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovooo"] += -(1./6.) * np.einsum("in,Aj,knAB,aBlm->ikajlm",f["oo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovooo"] +=  (1./6.) * np.einsum("bA,Bi,jkbB,aAlm->jkailm",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovooo"] +=  (1./6.) * np.einsum("BA,Ci,jkBC,aAlm->jkailm",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovooo"] += -(1./12.) * np.einsum("ab,Ai,jkAB,bBlm->jkailm",f["vv"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovooo"] += -(1./12.) * np.einsum("aA,Bi,jkbB,bAlm->jkailm",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovooo"] += -(1./12.) * np.einsum("aA,Bi,jkBC,AClm->jkailm",f["vV"],t1["Vo"],t2["ooVV"],t2["VVoo"],optimize="optimal")
		fs2s2s1["oovooo"] += -(1./12.) * np.einsum("bA,Ai,jkbB,aBlm->jkailm",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovooo"] += -(1./12.) * np.einsum("BA,Ai,jkBC,aClm->jkailm",f["VV"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovooo"] = asym_term(fs2s2s1["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s2s1["oooovv"] += -(1./3.) * np.einsum("Am,Bi,jmaB,klbA->jkliab",f["Vo"],t1["Vo"],t2["oovV"],t2["oovV"],optimize="optimal")
		fs2s2s1["oooovv"] +=  (1./6.) * np.einsum("ma,iA,jkbB,ABlm->ijklab",f["ov"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s1["oooovv"] = asym_term(fs2s2s1["oooovv"],"oooovv")

		# oovoov = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2s1["oovoov"] +=  (1./9.) * np.einsum("Ai,jB,kmaA,bBlm->jkbila",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovoov"] +=  (1./9.) * np.einsum("iA,Bj,kmaB,bAlm->ikbjla",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovoov"] +=  (1./18.) * np.einsum("am,iA,jmbB,ABkl->ijaklb",f["vo"],t1["oV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s1["oovoov"] += -(1./18.) * np.einsum("Am,iA,jmaB,bBkl->ijbkla",f["Vo"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovoov"] +=  (1./18.) * np.einsum("ma,Ai,jkAB,bBlm->jkbila",f["ov"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovoov"] += -(1./18.) * np.einsum("mA,Ai,jkaB,bBlm->jkbila",f["oV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovoov"] = asym_term(fs2s2s1["oovoov"],"oovoov")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s1["ovvooo"] +=  (1./3.) * np.einsum("mA,iB,aAjk,bBlm->iabjkl",f["oV"],t1["oV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s1["ovvooo"] +=  (1./6.) * np.einsum("am,Ai,jmAB,bBkl->jabikl",f["vo"],t1["Vo"],t2["ooVV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["ovvooo"] = asym_term(fs2s2s1["ovvooo"],"ovvooo")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs2s2s1["oovovv"] += -(1./9.) * np.einsum("Aa,iB,jlbA,cBkl->ijckab",f["Vv"],t1["oV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s1["oovovv"] = asym_term(fs2s2s1["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2s1["ovvoov"] += -(1./9.) * np.einsum("aA,Bi,jlbB,cAkl->jacikb",f["vV"],t1["Vo"],t2["oovV"],t2["vVoo"],optimize="optimal")
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
	fs2s2s2["ov"] += -4.000 * np.einsum("bj,jkaA,ilbB,ABkl->ia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ov"] += -4.000 * np.einsum("Aj,jkaB,ilbA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ov"] +=  4.000 * np.einsum("Aj,jkbB,ilaA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ov"] += -4.000 * np.einsum("Aj,ikAB,jlaC,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ov"] += -2.000 * np.einsum("Aj,ijaB,klbA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ov"] +=  2.000 * np.einsum("Aj,ijbB,klaA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ov"] +=  2.000 * np.einsum("Aj,klAB,ijaC,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ov"] +=  2.000 * np.einsum("Aj,jkBC,ilaA,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ov"] += -2.000 * np.einsum("bj,ijaA,klbB,ABkl->ia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ov"] +=  1.000 * np.einsum("bj,ikaA,jlbB,ABkl->ia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ov"] +=  1.000 * np.einsum("Aj,ikaB,jlbA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ov"] += -1.000 * np.einsum("Aj,ikbB,jlaA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ov"] +=  1.000 * np.einsum("Aj,jkAB,ilaC,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ov"] +=  1.000 * np.einsum("Aj,ijBC,klaA,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ov"] += -0.500 * np.einsum("bj,klaA,ijbB,ABkl->ia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ov"] += -0.500 * np.einsum("Aj,klaB,ijbA,bBkl->ia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ov"] +=  0.500 * np.einsum("Aj,ijAB,klaC,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ov"] += -0.500 * np.einsum("Aj,ikBC,jlaA,BCkl->ia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")

	fs2s2s2["vo"] += -4.000 * np.einsum("jb,klAB,bBil,aAjk->ai",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] += -4.000 * np.einsum("jA,klbB,bAil,aBjk->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] +=  4.000 * np.einsum("jA,klbB,aAil,bBjk->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] += -4.000 * np.einsum("jA,klBC,ABik,aCjl->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] += -2.000 * np.einsum("jb,klAB,bBkl,aAij->ai",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] +=  2.000 * np.einsum("jA,klbB,aAkl,bBij->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] += -2.000 * np.einsum("jA,klbB,bAkl,aBij->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] +=  2.000 * np.einsum("jA,klBC,BCjk,aAil->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] +=  2.000 * np.einsum("jA,klBC,ABkl,aCij->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] +=  1.000 * np.einsum("jb,klAB,aBil,bAjk->ai",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] += -1.000 * np.einsum("jA,klbB,aBil,bAjk->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] +=  1.000 * np.einsum("jA,klbB,bBil,aAjk->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] +=  1.000 * np.einsum("jA,klBC,BCij,aAkl->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] +=  1.000 * np.einsum("jA,klBC,ABjk,aCil->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] +=  0.500 * np.einsum("jb,klAB,aBkl,bAij->ai",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] += -0.500 * np.einsum("jA,klbB,aBkl,bAij->ai",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] +=  0.500 * np.einsum("jA,klBC,ABij,aCkl->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vo"] += -0.500 * np.einsum("jA,klBC,BCik,aAjl->ai",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")

	# ooov = np.zeros((n_occ,n_occ,n_occ,n_virt_int))
	fs2s2s2["ooov"] +=  2.000 * np.einsum("bl,ilaA,jmbB,ABkm->ijka",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  2.000 * np.einsum("Al,ilaB,jmbA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -2.000 * np.einsum("Al,ilbB,jmaA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  2.000 * np.einsum("Al,imAB,jlaC,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -1.000 * np.einsum("bl,lmaA,ijbB,ABkm->ijka",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -1.000 * np.einsum("Al,lmaB,ijbA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  1.000 * np.einsum("Al,ijAB,lmaC,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -1.000 * np.einsum("Al,ilBC,jmaA,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  0.750 * np.einsum("Al,lmbB,ijaA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -0.500 * np.einsum("bi,jlaA,kmbB,ABlm->jkia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -0.500 * np.einsum("Ai,jlaB,kmbA,bBlm->jkia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  0.500 * np.einsum("Ai,jlbB,kmaA,bBlm->jkia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  0.500 * np.einsum("Ai,jlAB,kmaC,BClm->jkia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -0.500 * np.einsum("bl,imaA,jlbB,ABkm->ijka",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -0.500 * np.einsum("Al,imaB,jlbA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -0.500 * np.einsum("Al,ilAB,jmaC,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  0.375 * np.einsum("Al,lmBC,ijaA,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  0.250 * np.einsum("Ai,jlBC,kmaA,BClm->jkia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -0.250 * np.einsum("bl,ijaA,lmbB,ABkm->ijka",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -0.250 * np.einsum("Al,ijaB,lmbA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  0.250 * np.einsum("Al,ijbB,lmaA,bBkm->ijka",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  0.250 * np.einsum("Al,lmAB,ijaC,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  0.125 * np.einsum("bi,jkaA,lmbB,ABlm->jkia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  0.125 * np.einsum("bi,lmaA,jkbB,ABlm->jkia",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  0.125 * np.einsum("Ai,jkaB,lmbA,bBlm->jkia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  0.125 * np.einsum("Ai,lmaB,jkbA,bBlm->jkia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -0.125 * np.einsum("Ai,jkbB,lmaA,bBlm->jkia",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -0.125 * np.einsum("Ai,jkAB,lmaC,BClm->jkia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -0.125 * np.einsum("Ai,lmAB,jkaC,BClm->jkia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] +=  0.125 * np.einsum("Al,ijBC,lmaA,BCkm->ijka",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] += -0.0625 * np.einsum("Ai,jkBC,lmaA,BClm->jkia",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ooov"] = asym_term(fs2s2s2["ooov"],"ooov")

	# ovoo = np.zeros((n_occ,n_virt_int,n_occ,n_occ))
	fs2s2s2["ovoo"] += -2.000 * np.einsum("lb,imAB,bBjm,aAkl->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  2.000 * np.einsum("lA,imbB,aAjm,bBkl->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -2.000 * np.einsum("lA,imbB,bAjm,aBkl->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  2.000 * np.einsum("lA,imBC,ABjm,aCkl->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -1.000 * np.einsum("lb,imAB,bBjk,aAlm->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -1.000 * np.einsum("lA,imbB,bAjk,aBlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  1.000 * np.einsum("lA,imBC,ABjk,aClm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -1.000 * np.einsum("lA,imBC,BCjl,aAkm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  0.750 * np.einsum("lA,imbB,aAjk,bBlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -0.500 * np.einsum("ib,lmAB,aBjm,bAkl->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  0.500 * np.einsum("lb,imAB,aBjm,bAkl->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  0.500 * np.einsum("iA,lmbB,aBjm,bAkl->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -0.500 * np.einsum("iA,lmbB,bBjm,aAkl->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  0.500 * np.einsum("iA,lmBC,ABjl,aCkm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -0.500 * np.einsum("lA,imbB,aBjm,bAkl->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -0.500 * np.einsum("lA,imBC,ABjl,aCkm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  0.375 * np.einsum("lA,imBC,BClm,aAjk->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  0.250 * np.einsum("lb,imAB,aBjk,bAlm->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  0.250 * np.einsum("iA,lmBC,BCjl,aAkm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -0.250 * np.einsum("lA,imbB,aBjk,bAlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  0.250 * np.einsum("lA,imbB,bBjk,aAlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  0.250 * np.einsum("lA,imBC,ABlm,aCjk->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -0.125 * np.einsum("ib,lmAB,aBjk,bAlm->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -0.125 * np.einsum("ib,lmAB,bAjk,aBlm->iajk",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  0.125 * np.einsum("iA,lmbB,aBjk,bAlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  0.125 * np.einsum("iA,lmbB,bAjk,aBlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -0.125 * np.einsum("iA,lmbB,bBjk,aAlm->iajk",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -0.125 * np.einsum("iA,lmBC,ABjk,aClm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -0.125 * np.einsum("iA,lmBC,ABlm,aCjk->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] +=  0.125 * np.einsum("lA,imBC,BCjk,aAlm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] += -0.0625 * np.einsum("iA,lmBC,BCjk,aAlm->iajk",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovoo"] = asym_term(fs2s2s2["ovoo"],"ovoo")

	# oovv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int))
	fs2s2s2["oovv"] +=  3.000 * np.einsum("lk,imaA,jkbB,ABlm->ijab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] += -3.000 * np.einsum("Ac,ikaB,jlbA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["oovv"] += -3.000 * np.einsum("BA,ikaC,jlbB,ACkl->ijab",f["VV"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  1.500 * np.einsum("lk,ijaA,kmbB,ABlm->ijab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  1.000 * np.einsum("ik,jlaA,kmbB,ABlm->ijab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] += -1.000 * np.einsum("ca,ikbA,jlcB,ABkl->ijab",f["vv"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] += -1.000 * np.einsum("Aa,ikbB,jlcA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  1.000 * np.einsum("Aa,ikcB,jlbA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  1.000 * np.einsum("Aa,ikAB,jlbC,BCkl->ijab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  0.750 * np.einsum("Ac,ijaB,klbA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  0.750 * np.einsum("Ac,klaB,ijbA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  0.750 * np.einsum("BA,ijaC,klbB,ACkl->ijab",f["VV"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  0.750 * np.einsum("BA,klaC,ijbB,ACkl->ijab",f["VV"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] += -0.500 * np.einsum("ik,lmaA,jkbB,ABlm->ijab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  0.500 * np.einsum("Aa,ikBC,jlbA,BCkl->ijab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  0.250 * np.einsum("ca,ijbA,klcB,ABkl->ijab",f["vv"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  0.250 * np.einsum("ca,klbA,ijcB,ABkl->ijab",f["vv"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  0.250 * np.einsum("Aa,ijbB,klcA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["oovv"] +=  0.250 * np.einsum("Aa,klbB,ijcA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["oovv"] += -0.250 * np.einsum("Aa,ijcB,klbA,cBkl->ijab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["oovv"] += -0.250 * np.einsum("Aa,ijAB,klbC,BCkl->ijab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] += -0.250 * np.einsum("Aa,klAB,ijbC,BCkl->ijab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] += -0.125 * np.einsum("Aa,ijBC,klbA,BCkl->ijab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["oovv"] = asym_term(fs2s2s2["oovv"],"oovv")

	# vvoo = np.zeros((n_virt_int,n_virt_int,n_occ,n_occ))
	fs2s2s2["vvoo"] += -3.000 * np.einsum("lk,kmAB,aBim,bAjl->abij",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] +=  3.000 * np.einsum("cA,klcB,aBil,bAjk->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] +=  3.000 * np.einsum("BA,klBC,aCil,bAjk->abij",f["VV"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] += -1.500 * np.einsum("lk,kmAB,aBij,bAlm->abij",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] +=  1.000 * np.einsum("ki,lmAB,aBjm,bAkl->abij",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] += -1.000 * np.einsum("ac,klAB,bBil,cAjk->abij",f["vv"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] +=  1.000 * np.einsum("aA,klcB,bBil,cAjk->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] += -1.000 * np.einsum("aA,klcB,cBil,bAjk->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] +=  1.000 * np.einsum("aA,klBC,ABik,bCjl->abij",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] += -0.750 * np.einsum("cA,klcB,aAij,bBkl->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] +=  0.750 * np.einsum("cA,klcB,aBij,bAkl->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] += -0.750 * np.einsum("BA,klBC,aAij,bCkl->abij",f["VV"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] +=  0.750 * np.einsum("BA,klBC,aCij,bAkl->abij",f["VV"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] +=  0.500 * np.einsum("ki,lmAB,aBlm,bAjk->abij",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] +=  0.500 * np.einsum("aA,klBC,BCik,bAjl->abij",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] += -0.250 * np.einsum("ac,klAB,bBij,cAkl->abij",f["vv"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] += -0.250 * np.einsum("ac,klAB,cAij,bBkl->abij",f["vv"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] +=  0.250 * np.einsum("aA,klcB,bBij,cAkl->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] +=  0.250 * np.einsum("aA,klcB,cAij,bBkl->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] += -0.250 * np.einsum("aA,klcB,cBij,bAkl->abij",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] += -0.250 * np.einsum("aA,klBC,ABij,bCkl->abij",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] += -0.250 * np.einsum("aA,klBC,ABkl,bCij->abij",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] += -0.125 * np.einsum("aA,klBC,BCij,bAkl->abij",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvoo"] = asym_term(fs2s2s2["vvoo"],"vvoo")

	# ovvv = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int))
	fs2s2s2["ovvv"] +=  2.000 * np.einsum("Aj,jkaB,ilbA,cBkl->icab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovvv"] +=  1.000 * np.einsum("Aj,ijaB,klbA,cBkl->icab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovvv"] += -0.500 * np.einsum("aj,ikbA,jlcB,ABkl->iabc",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ovvv"] += -0.500 * np.einsum("Aj,ikaB,jlbA,cBkl->icab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
	fs2s2s2["ovvv"] +=  0.250 * np.einsum("aj,klbA,ijcB,ABkl->iabc",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
	fs2s2s2["ovvv"] = asym_term(fs2s2s2["ovvv"],"ovvv")

	# vvov = np.zeros((n_virt_int,n_virt_int,n_occ,n_virt_int))
	fs2s2s2["vvov"] += -2.000 * np.einsum("jA,klaB,bAil,cBjk->bcia",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvov"] += -1.000 * np.einsum("jA,klaB,bAkl,cBij->bcia",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvov"] += -0.500 * np.einsum("ja,klAB,bBil,cAjk->bcia",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvov"] +=  0.500 * np.einsum("jA,klaB,bBil,cAjk->bcia",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
	fs2s2s2["vvov"] += -0.250 * np.einsum("ja,klAB,bBkl,cAij->bcia",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
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
		fs2s2s2["ooooov"] += -(1./3.) * np.einsum("bn,inaA,jkbB,ABlm->ijklma",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooov"] += -(1./3.) * np.einsum("An,inaB,jkbA,bBlm->ijklma",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooooov"] +=  (1./3.) * np.einsum("An,ijAB,knaC,BClm->ijklma",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooov"] +=  0.250 * np.einsum("An,inbB,jkaA,bBlm->ijklma",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooooov"] += -(1./6.) * np.einsum("bi,jkaA,lnbB,ABmn->jklima",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooov"] += -(1./6.) * np.einsum("bi,jnaA,klbB,ABmn->jklima",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooov"] += -(1./6.) * np.einsum("Ai,jkaB,lnbA,bBmn->jklima",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooooov"] += -(1./6.) * np.einsum("Ai,jnaB,klbA,bBmn->jklima",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooooov"] +=  (1./6.) * np.einsum("Ai,jkbB,lnaA,bBmn->jklima",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooooov"] +=  (1./6.) * np.einsum("Ai,jkAB,lnaC,BCmn->jklima",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooov"] +=  (1./6.) * np.einsum("Ai,jnAB,klaC,BCmn->jklima",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooov"] +=  0.125 * np.einsum("An,inBC,jkaA,BClm->ijklma",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooov"] +=  (1./12.) * np.einsum("Ai,jkBC,lnaA,BCmn->jklima",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooov"] += -(1./12.) * np.einsum("bn,ijaA,knbB,ABlm->ijklma",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooov"] += -(1./12.) * np.einsum("An,ijaB,knbA,bBlm->ijklma",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooooov"] +=  (1./12.) * np.einsum("An,inAB,jkaC,BClm->ijklma",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooov"] = asym_term(fs2s2s2["ooooov"],"ooooov")

		# oovooo = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s2["oovooo"] += -(1./3.) * np.einsum("nb,ijAB,bBkl,aAmn->ijaklm",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] += -(1./3.) * np.einsum("nA,ijbB,bAkl,aBmn->ijaklm",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] +=  (1./3.) * np.einsum("nA,ijBC,ABkl,aCmn->ijaklm",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] +=  0.250 * np.einsum("nA,ijbB,aAkl,bBmn->ijaklm",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] +=  (1./6.) * np.einsum("ib,jnAB,aBkl,bAmn->ijaklm",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] +=  (1./6.) * np.einsum("ib,jnAB,bAkl,aBmn->ijaklm",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] += -(1./6.) * np.einsum("iA,jnbB,aBkl,bAmn->ijaklm",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] += -(1./6.) * np.einsum("iA,jnbB,bAkl,aBmn->ijaklm",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] +=  (1./6.) * np.einsum("iA,jnbB,bBkl,aAmn->ijaklm",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] +=  (1./6.) * np.einsum("iA,jnBC,ABkl,aCmn->ijaklm",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] +=  (1./6.) * np.einsum("iA,jnBC,ABkn,aClm->ijaklm",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] +=  0.125 * np.einsum("nA,ijBC,BCkn,aAlm->ijaklm",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] +=  (1./12.) * np.einsum("nb,ijAB,aBkl,bAmn->ijaklm",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] +=  (1./12.) * np.einsum("iA,jnBC,BCkl,aAmn->ijaklm",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] += -(1./12.) * np.einsum("nA,ijbB,aBkl,bAmn->ijaklm",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] +=  (1./12.) * np.einsum("nA,ijBC,ABkn,aClm->ijaklm",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovooo"] = asym_term(fs2s2s2["oovooo"],"oovooo")

		# oooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s2s2["oooovv"] +=  0.500 * np.einsum("nm,ijaA,kmbB,ABln->ijklab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooovv"] +=  0.500 * np.einsum("Ac,ijaB,kmbA,cBlm->ijklab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oooovv"] +=  0.500 * np.einsum("Ac,imaB,jkbA,cBlm->ijklab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oooovv"] +=  0.500 * np.einsum("BA,ijaC,kmbB,AClm->ijklab",f["VV"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooovv"] +=  0.500 * np.einsum("BA,imaC,jkbB,AClm->ijklab",f["VV"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooovv"] +=  (1./3.) * np.einsum("mi,jkaA,lnbB,ABmn->jkliab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooovv"] +=  (1./3.) * np.einsum("im,jnaA,kmbB,ABln->ijklab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooovv"] +=  (1./6.) * np.einsum("im,jkaA,mnbB,ABln->ijklab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooovv"] +=  (1./6.) * np.einsum("ca,ijbA,kmcB,ABlm->ijklab",f["vv"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooovv"] +=  (1./6.) * np.einsum("ca,imbA,jkcB,ABlm->ijklab",f["vv"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooovv"] +=  (1./6.) * np.einsum("Aa,ijbB,kmcA,cBlm->ijklab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oooovv"] +=  (1./6.) * np.einsum("Aa,imbB,jkcA,cBlm->ijklab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oooovv"] += -(1./6.) * np.einsum("Aa,ijcB,kmbA,cBlm->ijklab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oooovv"] += -(1./6.) * np.einsum("Aa,ijAB,kmbC,BClm->ijklab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooovv"] += -(1./6.) * np.einsum("Aa,imAB,jkbC,BClm->ijklab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooovv"] += -(1./12.) * np.einsum("Aa,ijBC,kmbA,BClm->ijklab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooovv"] = asym_term(fs2s2s2["oooovv"],"oooovv")

		# ovvooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s2["ovvooo"] += -0.500 * np.einsum("nm,imAB,aBjk,bAln->iabjkl",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] += -0.500 * np.einsum("cA,imcB,aAjk,bBlm->iabjkl",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] +=  0.500 * np.einsum("cA,imcB,aBjk,bAlm->iabjkl",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] += -0.500 * np.einsum("BA,imBC,aAjk,bClm->iabjkl",f["VV"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] +=  0.500 * np.einsum("BA,imBC,aCjk,bAlm->iabjkl",f["VV"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] += -(1./3.) * np.einsum("mi,jnAB,aBkn,bAlm->jabikl",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] += -(1./3.) * np.einsum("im,mnAB,aBjk,bAln->iabjkl",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] += -(1./6.) * np.einsum("mi,jnAB,aBkl,bAmn->jabikl",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] += -(1./6.) * np.einsum("ac,imAB,bBjk,cAlm->iabjkl",f["vv"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] += -(1./6.) * np.einsum("ac,imAB,cAjk,bBlm->iabjkl",f["vv"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] +=  (1./6.) * np.einsum("aA,imcB,bBjk,cAlm->iabjkl",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] +=  (1./6.) * np.einsum("aA,imcB,cAjk,bBlm->iabjkl",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] += -(1./6.) * np.einsum("aA,imcB,cBjk,bAlm->iabjkl",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] += -(1./6.) * np.einsum("aA,imBC,ABjk,bClm->iabjkl",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] += -(1./6.) * np.einsum("aA,imBC,ABjm,bCkl->iabjkl",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] += -(1./12.) * np.einsum("aA,imBC,BCjk,bAlm->iabjkl",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvooo"] = asym_term(fs2s2s2["ovvooo"],"ovvooo")

		# ooovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		fs2s2s2["ooovvv"] += 1.000 * np.einsum("la,ijbA,kmcB,ABlm->ijkabc",f["ov"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooovvv"] = asym_term(fs2s2s2["ooovvv"],"ooovvv")

		# oovovv = np.zeros((n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int))
		fs2s2s2["oovovv"] += -(4./9.) * np.einsum("Al,ilaB,jmbA,cBkm->ijckab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovovv"] +=  (1./6.) * np.einsum("Al,lmaB,ijbA,cBkm->ijckab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovovv"] +=  (1./9.) * np.einsum("Ai,jlaB,kmbA,cBlm->jkciab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovovv"] +=  (1./9.) * np.einsum("al,imbA,jlcB,ABkm->ijakbc",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oovovv"] +=  (1./18.) * np.einsum("al,ijbA,lmcB,ABkm->ijakbc",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oovovv"] +=  (1./18.) * np.einsum("Al,ijaB,lmbA,cBkm->ijckab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovovv"] += -(1./36.) * np.einsum("Ai,jkaB,lmbA,cBlm->jkciab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovovv"] = asym_term(fs2s2s2["oovovv"],"oovovv")

		# ovvoov = np.zeros((n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2s2["ovvoov"] += -(4./9.) * np.einsum("lA,imaB,bAjm,cBkl->ibcjka",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvoov"] += -(1./6.) * np.einsum("lA,imaB,bAjk,cBlm->ibcjka",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvoov"] += -(1./9.) * np.einsum("la,imAB,bBjm,cAkl->ibcjka",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvoov"] += -(1./9.) * np.einsum("iA,lmaB,bBjm,cAkl->ibcjka",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvoov"] += -(1./18.) * np.einsum("la,imAB,bBjk,cAlm->ibcjka",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvoov"] +=  (1./18.) * np.einsum("lA,imaB,bBjk,cAlm->ibcjka",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvoov"] += -(1./36.) * np.einsum("iA,lmaB,bBjk,cAlm->ibcjka",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvoov"] = asym_term(fs2s2s2["ovvoov"],"ovvoov")

		# vvvooo = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ))
		fs2s2s2["vvvooo"] += -1.000 * np.einsum("al,lmAB,bBij,cAkm->abcijk",f["vo"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["vvvooo"] = asym_term(fs2s2s2["vvvooo"],"vvvooo")

		# oovvvv = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int,n_virt_int))
		fs2s2s2["oovvvv"] +=  (1./3.) * np.einsum("Aa,ikbB,jlcA,dBkl->ijdabc",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovvvv"] += -(1./12.) * np.einsum("Aa,ijbB,klcA,dBkl->ijdabc",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovvvv"] = asym_term(fs2s2s2["oovvvv"],"oovvvv")

		# vvvoov = np.zeros((n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_virt_int))
		fs2s2s2["vvvoov"] += -(1./3.) * np.einsum("aA,klbB,cBil,dAjk->acdijb",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["vvvoov"] += -(1./12.) * np.einsum("aA,klbB,cBij,dAkl->acdijb",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
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
		fs2s2s2["ooooooov"] +=  0.03125 * np.einsum("bi,jkaA,lmbB,ABnp->jklminpa",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooooov"] +=  0.03125 * np.einsum("Ai,jkaB,lmbA,bBnp->jklminpa",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooooooov"] += -0.03125 * np.einsum("Ai,jkAB,lmaC,BCnp->jklminpa",f["Vo"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooooov"] = asym_term(fs2s2s2["ooooooov"],"ooooooov")

		# ooovoooo = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		fs2s2s2["ooovoooo"] += -0.03125 * np.einsum("ib,jkAB,aBlm,bAnp->ijkalmnp",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooovoooo"] +=  0.03125 * np.einsum("iA,jkbB,aBlm,bAnp->ijkalmnp",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooovoooo"] += -0.03125 * np.einsum("iA,jkBC,ABlm,aCnp->ijkalmnp",f["oV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooovoooo"] = asym_term(fs2s2s2["ooovoooo"],"ooovoooo")

		# oooooovv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s2s2["oooooovv"] +=  0.0625 * np.einsum("Ac,ijaB,klbA,cBmn->ijklmnab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oooooovv"] +=  0.0625 * np.einsum("BA,ijaC,klbB,ACmn->ijklmnab",f["VV"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooooovv"] += -(1./24.) * np.einsum("pi,jkaA,lmbB,ABnp->jklminab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooooovv"] += -(1./24.) * np.einsum("ip,jkaA,lpbB,ABmn->ijklmnab",f["oo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooooovv"] +=  (1./48.) * np.einsum("ca,ijbA,klcB,ABmn->ijklmnab",f["vv"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooooovv"] +=  (1./48.) * np.einsum("Aa,ijbB,klcA,cBmn->ijklmnab",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oooooovv"] += -(1./48.) * np.einsum("Aa,ijAB,klbC,BCmn->ijklmnab",f["Vv"],t2["ooVV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["oooooovv"] = asym_term(fs2s2s2["oooooovv"],"oooooovv")

		# oovvoooo = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		fs2s2s2["oovvoooo"] +=  0.0625 * np.einsum("cA,ijcB,aBkl,bAmn->ijabklmn",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovvoooo"] +=  0.0625 * np.einsum("BA,ijBC,aCkl,bAmn->ijabklmn",f["VV"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")	
		fs2s2s2["oovvoooo"] +=  (1./24.) * np.einsum("pi,jkAB,aBlm,bAnp->jkabilmn",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovvoooo"] += -(1./24.) * np.einsum("ip,jpAB,aAkl,bBmn->ijabklmn",f["oo"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovvoooo"] += -(1./48.) * np.einsum("ac,ijAB,bBkl,cAmn->ijabklmn",f["vv"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovvoooo"] +=  (1./48.) * np.einsum("aA,ijcB,bBkl,cAmn->ijabklmn",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovvoooo"] += -(1./48.) * np.einsum("aA,ijBC,ABkl,bCmn->ijabklmn",f["vV"],t2["ooVV"],t2["VVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovvoooo"] = asym_term(fs2s2s2["oovvoooo"],"oovvoooo")

		# ooooovvv = np.zeros((n_occ,n_occ,n_occ,n_occ,n_occ,n_virt_int,n_virt_int,n_virt_int))
		fs2s2s2["ooooovvv"] +=  0.0625 * np.einsum("na,ijbA,klcB,ABmn->ijklmabc",f["ov"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooooovvv"] = asym_term(fs2s2s2["ooooovvv"],"ooooovvv")

		# ooovoovv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_occ,n_virt_int,n_virt_int))
		fs2s2s2["ooovoovv"] +=  0.03125 * np.einsum("An,inaB,jkbA,cBlm->ijkclmab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooovoovv"] +=  (1./48.) * np.einsum("Ai,jkaB,lnbA,cBmn->jklcimab",f["Vo"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooovoovv"] +=  (1./96.) * np.einsum("an,ijbA,kncB,ABlm->ijkalmbc",f["vo"],t2["oovV"],t2["oovV"],t2["VVoo"],optimize="optimal")
		fs2s2s2["ooovoovv"] = asym_term(fs2s2s2["ooovoovv"],"ooovoovv")

		# oovvooov = np.zeros((n_occ,n_occ,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2s2["oovvooov"] += -0.03125 * np.einsum("nA,ijaB,bAkl,cBmn->ijbcklma",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovvooov"] +=  (1./48.) * np.einsum("iA,jnaB,bBkl,cAmn->ijbcklma",f["oV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovvooov"] += -(1./96.) * np.einsum("na,ijAB,bBkl,cAmn->ijbcklma",f["ov"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["oovvooov"] = asym_term(fs2s2s2["oovvooov"],"oovvooov")

		# ovvvoooo = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_occ))
		fs2s2s2["ovvvoooo"] +=  0.0625 * np.einsum("an,inAB,bAjk,cBlm->iabcjklm",f["vo"],t2["ooVV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ovvvoooo"] = asym_term(fs2s2s2["ovvvoooo"],"ovvvoooo")

		# ooovovvv = np.zeros((n_occ,n_occ,n_occ,n_virt_int,n_occ,n_virt_int,n_virt_int,n_virt_int))
		fs2s2s2["ooovovvv"] += -0.03125 * np.einsum("Aa,ijbB,kmcA,dBlm->ijkdlabc",f["Vv"],t2["oovV"],t2["oovV"],t2["vVoo"],optimize="optimal")
		fs2s2s2["ooovovvv"] = asym_term(fs2s2s2["ooovovvv"],"ooovovvv")

		# ovvvooov = np.zeros((n_occ,n_virt_int,n_virt_int,n_virt_int,n_occ,n_occ,n_occ,n_virt_int))
		fs2s2s2["ovvvooov"] += -0.03125 * np.einsum("aA,imbB,cBjk,dAlm->iacdjklb",f["vV"],t2["oovV"],t2["vVoo"],t2["vVoo"],optimize="optimal")
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

    print("Size of the active space                       :%10i" %(n_act))

    fmat, vten = H(HamFormat.SPINORB_FV)
    ccsd_energy = calc_ccsd(fmat, vten, t1_amps, t2_amps, verbose=0)
    assert np.isclose(ccsd_energy+system.meanfield.e_tot, mccsd.e_tot, rtol=0.0, atol=1e-8)

    n_a = system.n_a
    n_b = system.n_b 

    key = approximation.strip().lower()
    if key == "a1":
        ham = eff_ham_a1(fmat,vten,n_a,n_b,n_act)    
    elif key == "a2":
        ham = eff_ham_a2(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=False)
    elif key == "a3":
        ham = eff_ham_a3(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=False)
    elif key == "a4":
        ham = eff_ham_a4(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=False)
    elif key == "a5":
        ham = eff_ham_a5(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=False,four_body=False)
    elif key == "a6":
        ham = eff_ham_a6(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=False,four_body=False)
    elif key == "a7":
        ham = eff_ham_a7(fmat,vten,t1_amps,t2_amps,n_a,n_b,n_act,three_body=False,four_body=False)        
    else:
        raise ValueError(f"Unsupported DUCC method {key!r}; choose between 'A1' to 'A7'.")

    use_pyscf = True    
    if use_pyscf:
        constant, h, g = ham(HamFormat.SPATORB_PV)
        cisolver = fci.direct_spin1.FCI()
        efci, _ = cisolver.kernel(h, g, n_act, (n_a,n_b), ecore=constant, nroots=1, verbose=5)
        print(f"DUCC_{key.upper()} Full CI PySCF                          :%21.15f"%(efci))

    use_openfermion = True
    if use_openfermion:
        ham_mat = ham(HamFormat.HILBERT)
        evals, evecs = scipy.sparse.linalg.eigsh(ham_mat, k=1, which="SA")
        print(f"DUCC_{key.upper()} Full CI OpenFermion                    :%21.15f"%(evals[0]))

    return efci