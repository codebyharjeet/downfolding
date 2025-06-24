import numpy as np 
import openfermion as of 
from openfermion import *

def asym_term(term,kind):
	"""Antisymmetrizes 2-,3-,and 4-body subtensor terms

	Parameters
	----------
	term : np.ndarray
		subtensor to be antisymmetrized
	kind : str
		dictionary key for the subtensor to inform antisymmetrization

	Returns
	-------
	term : np.ndarray
		antisymmetrized subtensor
	""" 
	bra = np.zeros_like(term)
	match kind:
		case "oooo":
			bra +=  0.5 * term 
			bra += -0.5 * np.einsum("ijkl->jikl",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.5 * bra 
			term += -0.5 * np.einsum("ijkl->ijlk",bra,optimize="optimal") 
			del bra 
			return term
			# term_as +=  0.25 * term 
			# term_as += -0.25 * np.einsum("ijkl->jikl",term,optimize="optimal") 
			# term_as += -0.25 * np.einsum("ijkl->ijlk",term,optimize="optimal")
			# term_as +=  0.25 * np.einsum("ijkl->jilk",term,optimize="optimal")
			# return term_as
		case "ooov":
			bra +=  0.5 * term
			bra += -0.5 * np.einsum("ijka->jika",term,optimize="optimal")
			term = np.zeros_like(bra)  
			term +=  bra
			del bra
			return term
			# term_as +=  0.5 * term 
			# term_as += -0.5 * np.einsum("ijka->jika",term,optimize="optimal")
			# return term_as
		case "ovoo":
			bra += term 
			term = np.zeros_like(bra)
			term +=  0.5 * bra 
			term += -0.5 * np.einsum("iajk->iakj",bra,optimize="optimal")
			del bra
			return term
			# term_as +=  0.5 * term 
			# term_as += -0.5 * np.einsum("iajk->iakj",term,optimize="optimal")
			# return term_as
		case "oovv": 
			bra +=  0.5 * term 
			bra += -0.5 * np.einsum("ijab->jiab",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.5 * bra 
			term += -0.5 * np.einsum("ijab->ijba",bra,optimize="optimal")
			del bra
			return term
			# term_as +=  0.25 * term 
			# term_as += -0.25 * np.einsum("ijab->jiab",term,optimize="optimal")
			# term_as += -0.25 * np.einsum("ijab->ijba",term,optimize="optimal")
			# term_as +=  0.25 * np.einsum("ijab->jiba",term,optimize="optimal")
			# return term_as 
		case "ovov":
			return term
			# term_as += term 
			# return term_as 
		case "vvoo": 
			bra +=  0.5 * term 
			bra += -0.5 * np.einsum("abij->baij",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.5 * bra 
			term += -0.5 * np.einsum("abij->abji",bra,optimize="optimal")
			del bra
			return term
			# term_as +=  0.25 * term 
			# term_as += -0.25 * np.einsum("abij->baij",term,optimize="optimal")
			# term_as += -0.25 * np.einsum("abij->abji",term,optimize="optimal")
			# term_as +=  0.25 * np.einsum("abij->baji",term,optimize="optimal")
			# return term_as
		case "ovvv":
			bra += term 
			term = np.zeros_like(bra)
			term +=  0.5 * bra
			term += -0.5 * np.einsum("iabc->iacb",bra,optimize="optimal")
			del bra 
			return term
			# term_as +=  0.5 * term 
			# term_as += -0.5 * np.einsum("iabc->iacb",term,optimize="optimal")
			# return term_as 
		case "vvov": 
			bra +=  0.5 * term
			bra += -0.5 * np.einsum("abic->baic",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  bra
			del bra
			return term
			# term_as +=  0.5 * term
			# term_as += -0.5 * np.einsum("abic->baic",term,optimize="optimal")
			# return term_as 
		case "vvvv":
			bra +=  0.5 * term 
			bra += -0.5 * np.einsum("abcd->bacd",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.5 * bra 
			term += -0.5 * np.einsum("abcd->abdc",bra,optimize="optimal")
			del bra
			return term
			# term_as +=  0.25 * term 
			# term_as += -0.25 * np.einsum("abcd->bacd",term,optimize="optimal")
			# term_as += -0.25 * np.einsum("abcd->abdc",term,optimize="optimal")
			# term_as +=  0.25 * np.einsum("abcd->badc",term,optimize="optimal")
			# return term_as 
		case "oooooo":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("ijklmn->ikjlmn",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijklmn->jiklmn",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijklmn->jkilmn",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijklmn->kijlmn",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijklmn->kjilmn",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("ijklmn->ijklnm",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijklmn->ijkmln",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijklmn->ijkmnl",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijklmn->ijknlm",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijklmn->ijknml",bra,optimize="optimal")
			del bra
			return term
			# term_as +=  (1./36.) * term 
			# term_as += -(1./36.) * np.einsum("ijklmn->ikjlmn",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->jiklmn",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->jkilmn",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->kijlmn",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->kjilmn",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->ijklnm",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->ikjlnm",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->jiklnm",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->jkilnm",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->kijlnm",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->kjilnm",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->ijkmln",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->ikjmln",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->jikmln",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->jkimln",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->kijmln",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->kjimln",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->ijkmnl",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->ikjmnl",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->jikmnl",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->jkimnl",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->kijmnl",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->kjimnl",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->ijknlm",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->ikjnlm",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->jiknlm",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->jkinlm",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->kijnlm",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->kjinlm",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->ijknml",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->ikjnml",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->jiknml",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->jkinml",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijklmn->kijnml",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijklmn->kjinml",term,optimize="optimal")
			# return term_as
		case "ooooov":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("ijklma->ikjlma",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijklma->jiklma",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijklma->jkilma",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijklma->kijlma",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijklma->kjilma",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.5 * bra 
			term += -0.5 * np.einsum("ijklma->ijkmla",bra,optimize="optimal")
			del bra
			return term
			# term_as +=  (1./12.) * term 
			# term_as += -(1./12.) * np.einsum("ijklma->ikjlma",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijklma->jiklma",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijklma->jkilma",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijklma->kijlma",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijklma->kjilma",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijklma->ijkmla",term,optimize="optimal") 
			# term_as +=  (1./12.) * np.einsum("ijklma->ikjmla",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijklma->jikmla",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijklma->jkimla",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijklma->kijmla",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijklma->kjimla",term,optimize="optimal")
			# return term_as
		case "oovooo":
			bra +=  0.5 * term
			bra += -0.5 * np.einsum("ijaklm->jiaklm",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("ijaklm->ijakml",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijaklm->ijalkm",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijaklm->ijalmk",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijaklm->ijamkl",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijaklm->ijamlk",bra,optimize="optimal")
			del bra 
			return term
			# term_as +=  (1./12.) * term 
			# term_as += -(1./12.) * np.einsum("ijaklm->jiaklm",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijaklm->ijakml",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijaklm->jiakml",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijaklm->ijalkm",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijaklm->jialkm",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijaklm->ijalmk",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijaklm->jialmk",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijaklm->ijamkl",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijaklm->jiamkl",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijaklm->ijamlk",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijaklm->jiamlk",term,optimize="optimal")
			# return term_as 
		case "oooovv":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("ijklab->ikjlab",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijklab->jiklab",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijklab->jkilab",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijklab->kijlab",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijklab->kjilab",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.5 * bra 
			term += -0.5 * np.einsum("ijklab->ijklba",bra,optimize="optimal")
			del bra 
			return term
			# term_as +=  (1./12.) * term 
			# term_as += -(1./12.) * np.einsum("ijklab->ikjlab",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijklab->jiklab",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijklab->jkilab",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijklab->kijlab",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijklab->kjilab",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijklab->ijklba",term,optimize="optimal") 
			# term_as +=  (1./12.) * np.einsum("ijklab->ikjlba",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijklab->jiklba",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijklab->jkilba",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijklab->kijlba",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijklab->kjilba",term,optimize="optimal")
			# return term_as 
		case "oovoov":
			bra +=  0.5 * term
			bra += -0.5 * np.einsum("ijaklb->jiaklb",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.5 * bra 
			term += -0.5 * np.einsum("ijaklb->ijalkb",bra,optimize="optimal")
			del bra 
			return term
			# term_as +=  0.25 * term 
			# term_as += -0.25 * np.einsum("ijaklb->jiaklb",term,optimize="optimal")
			# term_as += -0.25 * np.einsum("ijaklb->ijalkb",term,optimize="optimal")
			# term_as +=  0.25 * np.einsum("ijaklb->jialkb",term,optimize="optimal")
			# return term_as 
		case "ovvooo":
			bra +=  0.5 * term 
			bra += -0.5 * np.einsum("iabjkl->ibajkl",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("iabjkl->iabjlk",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("iabjkl->iabkjl",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("iabjkl->iabklj",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("iabjkl->iabljk",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("iabjkl->iablkj",bra,optimize="optimal")
			del bra
			return term
			# term_as +=  (1./12.) * term 
			# term_as += -(1./12.) * np.einsum("iabjkl->ibajkl",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("iabjkl->iabjlk",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("iabjkl->ibajlk",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("iabjkl->iabkjl",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("iabjkl->ibakjl",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("iabjkl->iabklj",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("iabjkl->ibaklj",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("iabjkl->iabljk",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("iabjkl->ibaljk",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("iabjkl->iablkj",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("iabjkl->ibalkj",term,optimize="optimal")
			# return term_as 
		case "ooovvv":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("ijkabc->ikjabc",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijkabc->jikabc",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijkabc->jkiabc",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijkabc->kijabc",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijkabc->kjiabc",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("ijkabc->ijkacb",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijkabc->ijkbac",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijkabc->ijkbca",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijkabc->ijkcab",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijkabc->ijkcba",bra,optimize="optimal")
			del bra
			return term 
			# term_as +=  (1./36.) * term 
			# term_as += -(1./36.) * np.einsum("ijkabc->ikjabc",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->jikabc",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->jkiabc",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->kijabc",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->kjiabc",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->ijkacb",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->ikjacb",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->jikacb",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->jkiacb",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->kijacb",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->kjiacb",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->ijkbac",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->ikjbac",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->jikbac",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->jkibac",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->kijbac",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->kjibac",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->ijkbca",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->ikjbca",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->jikbca",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->jkibca",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->kijbca",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->kjibca",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->ijkcab",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->ikjcab",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->jikcab",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->jkicab",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->kijcab",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->kjicab",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->ijkcba",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->ikjcba",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->jikcba",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->jkicba",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("ijkabc->kijcba",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("ijkabc->kjicba",term,optimize="optimal")
			# return term_as 
		case "oovovv":
			bra +=  0.5 * term 
			bra += -0.5 * np.einsum("ijakbc->jiakbc",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.5 * bra 
			term += -0.5 * np.einsum("ijakbc->ijakcb",bra,optimize="optimal")
			del bra 
			return term 
			# term_as +=  0.25 * term 
			# term_as += -0.25 * np.einsum("ijakbc->jiakbc",term,optimize="optimal")
			# term_as += -0.25 * np.einsum("ijakbc->ijakcb",term,optimize="optimal")
			# term_as +=  0.25 * np.einsum("ijakbc->jiakcb",term,optimize="optimal")
			# return term_as 
		case "ovvoov": 
			bra +=  0.5 * term 
			bra += -0.5 * np.einsum("iabjkc->ibajkc",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.5 * bra 
			term += -0.5 * np.einsum("iabjkc->iabkjc",bra,optimize="optimal")
			del bra 
			return term 
			# term_as +=  0.25 * term 
			# term_as += -0.25 * np.einsum("iabjkc->ibajkc",term,optimize="optimal")
			# term_as += -0.25 * np.einsum("iabjkc->iabkjc",term,optimize="optimal")
			# term_as +=  0.25 * np.einsum("iabjkc->ibakjc",term,optimize="optimal")
			# return term_as 
		case "vvvooo":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("abcijk->acbijk",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("abcijk->bacijk",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("abcijk->bcaijk",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("abcijk->cabijk",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("abcijk->cbaijk",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("abcijk->abcikj",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("abcijk->abcjik",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("abcijk->abcjki",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("abcijk->abckij",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("abcijk->abckji",bra,optimize="optimal")
			del bra 
			return term
			# term_as +=  (1./36.) * term 
			# term_as += -(1./36.) * np.einsum("abcijk->acbijk",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->bacijk",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->bcaijk",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->cabijk",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->cbaijk",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->abcikj",term,optimize="optimal") 
			# term_as +=  (1./36.) * np.einsum("abcijk->acbikj",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->bacikj",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->bcaikj",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->cabikj",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->cbaikj",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->abcjik",term,optimize="optimal") 
			# term_as +=  (1./36.) * np.einsum("abcijk->acbjik",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->bacjik",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->bcajik",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->cabjik",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->cbajik",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->abcjki",term,optimize="optimal") 
			# term_as += -(1./36.) * np.einsum("abcijk->acbjki",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->bacjki",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->bcajki",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->cabjki",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->cbajki",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->abckij",term,optimize="optimal") 
			# term_as += -(1./36.) * np.einsum("abcijk->acbkij",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->backij",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->bcakij",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->cabkij",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->cbakij",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->abckji",term,optimize="optimal") 
			# term_as +=  (1./36.) * np.einsum("abcijk->acbkji",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->backji",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->bcakji",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcijk->cabkji",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcijk->cbakji",term,optimize="optimal")
			# return term_as 
		case "oovvvv":
			bra +=  0.5 * term 
			bra += -0.5 * np.einsum("ijabcd->jiabcd",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("ijabcd->ijabdc",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijabcd->ijacbd",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijabcd->ijacdb",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijabcd->ijadbc",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijabcd->ijadcb",bra,optimize="optimal")
			del bra
			return term 
			# term_as +=  (1./12.) * term 
			# term_as += -(1./12.) * np.einsum("ijabcd->jiabcd",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijabcd->ijabdc",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijabcd->jiabdc",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijabcd->ijacbd",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijabcd->jiacbd",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijabcd->ijacdb",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijabcd->jiacdb",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijabcd->ijadbc",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijabcd->jiadbc",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("ijabcd->ijadcb",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("ijabcd->jiadcb",term,optimize="optimal")
			return term_as 
		case "ovvovv":
			bra +=  0.5 * term 
			bra += -0.5 * np.einsum("iabjcd->ibajcd",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.5 * bra 
			term += -0.5 * np.einsum("iabjcd->iabjdc",bra,optimize="optimal")
			del bra
			return term
			# term_as +=  0.25 * term 
			# term_as += -0.25 * np.einsum("iabjcd->ibajcd",term,optimize="optimal")
			# term_as += -0.25 * np.einsum("iabjcd->iabjdc",term,optimize="optimal")
			# term_as +=  0.25 * np.einsum("iabjcd->ibajdc",term,optimize="optimal")
			# return term_as 
		case "vvvoov":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("abcijd->acbijd",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("abcijd->bacijd",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("abcijd->bcaijd",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("abcijd->cabijd",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("abcijd->cbaijd",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.5 * bra
			term += -0.5 * np.einsum("abcijd->abcjid",bra,optimize="optimal")
			del bra 
			return term 
			# term_as +=  (1./12.) * term 
			# term_as += -(1./12.) * np.einsum("abcijd->acbijd",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("abcijd->bacijd",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("abcijd->bcaijd",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("abcijd->cabijd",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("abcijd->cbaijd",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("abcijd->abcjid",term,optimize="optimal") 
			# term_as +=  (1./12.) * np.einsum("abcijd->acbjid",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("abcijd->bacjid",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("abcijd->bcajid",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("abcijd->cabjid",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("abcijd->cbajid",term,optimize="optimal")
			# return term_as 
		case "ovvvvv":
			bra +=  0.5 * term 
			bra += -0.5 * np.einsum("iabcde->ibacde",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("iabcde->iabced",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("iabcde->iabdce",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("iabcde->iabdec",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("iabcde->iabecd",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("iabcde->iabedc",bra,optimize="optimal")
			del bra 
			return term 
			# term_as +=  (1./12.) * term 
			# term_as += -(1./12.) * np.einsum("iabcde->ibacde",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("iabcde->iabced",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("iabcde->ibaced",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("iabcde->iabdce",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("iabcde->ibadce",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("iabcde->iabdec",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("iabcde->ibadec",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("iabcde->iabecd",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("iabcde->ibaecd",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("iabcde->iabedc",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("iabcde->ibaedc",term,optimize="optimal")
			# return term_as 
		case "vvvovv":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("abcide->acbide",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("abcide->bacide",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("abcide->bcaide",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("abcide->cabide",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("abcide->cbaide",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.5 * bra 
			term += -0.5 * np.einsum("abcide->abcied",bra,optimize="optimal")
			del bra 
			return term  
			# term_as +=  (1./12.) * term 
			# term_as += -(1./12.) * np.einsum("abcide->acbide",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("abcide->bacide",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("abcide->bcaide",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("abcide->cabide",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("abcide->cbaide",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("abcide->abcied",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("abcide->acbied",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("abcide->bacied",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("abcide->bcaied",term,optimize="optimal")
			# term_as += -(1./12.) * np.einsum("abcide->cabied",term,optimize="optimal")
			# term_as +=  (1./12.) * np.einsum("abcide->cbaied",term,optimize="optimal")
			# return term_as 
		case "vvvvvv":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("abcdef->acbdef",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("abcdef->bacdef",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("abcdef->bcadef",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("abcdef->cabdef",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("abcdef->cbadef",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("abcdef->abcdfe",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("abcdef->abcedf",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("abcdef->abcefd",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("abcdef->abcfde",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("abcdef->abcfed",bra,optimize="optimal")
			del bra 
			return term  
			# term_as +=  (1./36.) * term 
			# term_as += -(1./36.) * np.einsum("abcdef->acbdef",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->bacdef",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->bcadef",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->cabdef",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->cbadef",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->abcdfe",term,optimize="optimal") 
			# term_as +=  (1./36.) * np.einsum("abcdef->acbdfe",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->bacdfe",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->bcadfe",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->cabdfe",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->cbadfe",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->abcedf",term,optimize="optimal") 
			# term_as +=  (1./36.) * np.einsum("abcdef->acbedf",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->bacedf",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->bcaedf",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->cabedf",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->cbaedf",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->abcefd",term,optimize="optimal") 
			# term_as += -(1./36.) * np.einsum("abcdef->acbefd",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->bacefd",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->bcaefd",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->cabefd",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->cbaefd",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->abcfde",term,optimize="optimal") 
			# term_as += -(1./36.) * np.einsum("abcdef->acbfde",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->bacfde",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->bcafde",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->cabfde",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->cbafde",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->abcfed",term,optimize="optimal") 
			# term_as +=  (1./36.) * np.einsum("abcdef->acbfed",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->bacfed",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->bcafed",term,optimize="optimal")
			# term_as += -(1./36.) * np.einsum("abcdef->cabfed",term,optimize="optimal")
			# term_as +=  (1./36.) * np.einsum("abcdef->cbafed",term,optimize="optimal")
			return term_as 
		case "ooooooov":
			bra +=  (1./24.) * term 
			bra += -(1./24.) * np.einsum("ijklmnoa->ijlkmnoa",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnoa->ikjlmnoa",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnoa->ikljmnoa",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnoa->iljkmnoa",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnoa->ilkjmnoa",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnoa->jiklmnoa",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnoa->jilkmnoa",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnoa->jkilmnoa",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnoa->jklimnoa",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnoa->jlikmnoa",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnoa->jlkimnoa",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnoa->kijlmnoa",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnoa->kiljmnoa",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnoa->kjilmnoa",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnoa->kjlimnoa",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnoa->klijmnoa",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnoa->kljimnoa",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnoa->lijkmnoa",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnoa->likjmnoa",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnoa->ljikmnoa",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnoa->ljkimnoa",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnoa->lkijmnoa",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnoa->lkjimnoa",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("ijklmnoa->ijklmona",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijklmnoa->ijklnmoa",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijklmnoa->ijklnoma",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijklmnoa->ijklomna",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijklmnoa->ijklonma",bra,optimize="optimal")
			del bra 
			return term
		case "ooovoooo":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("ijkalmno->ikjalmno",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijkalmno->jikalmno",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijkalmno->jkialmno",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijkalmno->kijalmno",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijkalmno->kjialmno",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./24.) * bra
			term += -(1./24.) * np.einsum("ijkalmno->ijkalmon",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijkalmno->ijkalnmo",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijkalmno->ijkalnom",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijkalmno->ijkalomn",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijkalmno->ijkalonm",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijkalmno->ijkamlno",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijkalmno->ijkamlon",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijkalmno->ijkamnlo",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijkalmno->ijkamnol",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijkalmno->ijkamoln",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijkalmno->ijkamonl",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijkalmno->ijkanlmo",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijkalmno->ijkanlom",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijkalmno->ijkanmlo",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijkalmno->ijkanmol",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijkalmno->ijkanolm",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijkalmno->ijkanoml",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijkalmno->ijkaolmn",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijkalmno->ijkaolnm",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijkalmno->ijkaomln",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijkalmno->ijkaomnl",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijkalmno->ijkaonlm",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijkalmno->ijkaonml",bra,optimize="optimal")
			del bra
			return term
		case "oooooovv":
			bra +=  (1./24.) * term 
			bra += -(1./24.) * np.einsum("ijklmnab->ijlkmnab",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnab->ikjlmnab",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnab->ikljmnab",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnab->iljkmnab",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnab->ilkjmnab",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnab->jiklmnab",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnab->jilkmnab",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnab->jkilmnab",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnab->jklimnab",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnab->jlikmnab",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnab->jlkimnab",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnab->kijlmnab",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnab->kiljmnab",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnab->kjilmnab",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnab->kjlimnab",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnab->klijmnab",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnab->kljimnab",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnab->lijkmnab",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnab->likjmnab",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnab->ljikmnab",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnab->ljkimnab",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmnab->lkijmnab",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmnab->lkjimnab",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.250 * bra 
			term += -0.250 * np.einsum("ijklmnab->ijklnmab",bra,optimize="optimal")
			term += -0.250 * np.einsum("ijklmnab->ijklmnba",bra,optimize="optimal")
			term +=  0.250 * np.einsum("ijklmnab->ijklnmba",bra,optimize="optimal")
			del bra
			return term
		case "ooovooov":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("ijkalmnb->ikjalmnb",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijkalmnb->jikalmnb",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijkalmnb->jkialmnb",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijkalmnb->kijalmnb",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijkalmnb->kjialmnb",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("ijkalmnb->ijkalnmb",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijkalmnb->ijkamlnb",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijkalmnb->ijkamnlb",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijkalmnb->ijkanlmb",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijkalmnb->ijkanmlb",bra,optimize="optimal")
			del bra
			return term 
		case "oovvoooo":
			bra = np.zeros_like(term)
			bra +=  0.250 * term 
			bra += -0.250 * np.einsum("ijabklmn->jiabklmn",term,optimize="optimal")
			bra += -0.250 * np.einsum("ijabklmn->ijbaklmn",term,optimize="optimal")
			bra +=  0.250 * np.einsum("ijabklmn->jibaklmn",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./24.) * bra 
			term += -(1./24.) * np.einsum("ijabklmn->ijabklnm",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijabklmn->ijabkmln",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijabklmn->ijabkmnl",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijabklmn->ijabknlm",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijabklmn->ijabknml",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijabklmn->ijablkmn",bra,optimize="optimal") 
			term +=  (1./24.) * np.einsum("ijabklmn->ijablknm",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijabklmn->ijablmkn",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijabklmn->ijablmnk",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijabklmn->ijablnkm",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijabklmn->ijablnmk",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijabklmn->ijabmkln",bra,optimize="optimal") 
			term += -(1./24.) * np.einsum("ijabklmn->ijabmknl",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijabklmn->ijabmlkn",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijabklmn->ijabmlnk",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijabklmn->ijabmnkl",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijabklmn->ijabmnlk",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijabklmn->ijabnklm",bra,optimize="optimal") 
			term +=  (1./24.) * np.einsum("ijabklmn->ijabnkml",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijabklmn->ijabnlkm",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijabklmn->ijabnlmk",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijabklmn->ijabnmkl",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijabklmn->ijabnmlk",bra,optimize="optimal")
			del bra
			return term
		case "ooooovvv":
			bra +=  (1./24.) * term 
			bra += -(1./24.) * np.einsum("ijklmabc->ijlkmabc",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmabc->ikjlmabc",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmabc->ikljmabc",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmabc->iljkmabc",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmabc->ilkjmabc",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmabc->jiklmabc",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmabc->jilkmabc",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmabc->jkilmabc",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmabc->jklimabc",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmabc->jlikmabc",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmabc->jlkimabc",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmabc->kijlmabc",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmabc->kiljmabc",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmabc->kjilmabc",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmabc->kjlimabc",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmabc->klijmabc",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmabc->kljimabc",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmabc->lijkmabc",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmabc->likjmabc",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmabc->ljikmabc",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmabc->ljkimabc",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklmabc->lkijmabc",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklmabc->lkjimabc",term,optimize="optimal")
			term = np.zeros_like(bra) 
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("ijklmabc->ijklmacb",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijklmabc->ijklmbac",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijklmabc->ijklmbca",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijklmabc->ijklmcab",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijklmabc->ijklmcba",bra,optimize="optimal")
			del bra
			return term
		case "ooovoovv":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("ijkalmbc->ikjalmbc",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijkalmbc->jikalmbc",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijkalmbc->jkialmbc",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijkalmbc->kijalmbc",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijkalmbc->kjialmbc",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.250 * bra 
			term += -0.250 * np.einsum("ijkalmbc->ijkamlbc",bra,optimize="optimal")
			term += -0.250 * np.einsum("ijkalmbc->ijkalmcb",bra,optimize="optimal")
			term +=  0.250 * np.einsum("ijkalmbc->ijkamlcb",bra,optimize="optimal")
			del bra 
			return term
		case "oovvooov":
			bra +=  0.250 * term 
			bra += -0.250 * np.einsum("ijabklmc->jiabklmc",term,optimize="optimal")
			bra += -0.250 * np.einsum("ijabklmc->ijbaklmc",term,optimize="optimal")
			bra +=  0.250 * np.einsum("ijabklmc->jibaklmc",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("ijabklmc->ijabkmlc",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijabklmc->ijablkmc",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijabklmc->ijablmkc",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijabklmc->ijabmklc",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijabklmc->ijabmlkc",bra,optimize="optimal")
			del bra 
			return term
		case "ovvvoooo":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("iabcjklm->iacbjklm",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("iabcjklm->ibacjklm",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("iabcjklm->ibcajklm",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("iabcjklm->icabjklm",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("iabcjklm->icbajklm",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./24.) * bra 
			term += -(1./24.) * np.einsum("iabcjklm->iabcjkml",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("iabcjklm->iabcjlkm",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("iabcjklm->iabcjlmk",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("iabcjklm->iabcjmkl",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("iabcjklm->iabcjmlk",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("iabcjklm->iabckjlm",bra,optimize="optimal") 
			term +=  (1./24.) * np.einsum("iabcjklm->iabckjml",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("iabcjklm->iabckljm",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("iabcjklm->iabcklmj",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("iabcjklm->iabckmjl",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("iabcjklm->iabckmlj",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("iabcjklm->iabcljkm",bra,optimize="optimal") 
			term += -(1./24.) * np.einsum("iabcjklm->iabcljmk",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("iabcjklm->iabclkjm",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("iabcjklm->iabclkmj",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("iabcjklm->iabclmjk",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("iabcjklm->iabclmkj",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("iabcjklm->iabcmjkl",bra,optimize="optimal") 
			term +=  (1./24.) * np.einsum("iabcjklm->iabcmjlk",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("iabcjklm->iabcmkjl",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("iabcjklm->iabcmklj",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("iabcjklm->iabcmljk",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("iabcjklm->iabcmlkj",bra,optimize="optimal")
			del bra 
			return term
		case "oooovvvv":
			bra +=  (1./24.) * term 
			bra += -(1./24.) * np.einsum("ijklabcd->ijlkabcd",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklabcd->ikjlabcd",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklabcd->ikljabcd",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklabcd->iljkabcd",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklabcd->ilkjabcd",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklabcd->jiklabcd",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklabcd->jilkabcd",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklabcd->jkilabcd",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklabcd->jkliabcd",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklabcd->jlikabcd",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklabcd->jlkiabcd",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklabcd->kijlabcd",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklabcd->kiljabcd",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklabcd->kjilabcd",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklabcd->kjliabcd",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklabcd->klijabcd",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklabcd->kljiabcd",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklabcd->lijkabcd",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklabcd->likjabcd",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklabcd->ljikabcd",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklabcd->ljkiabcd",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("ijklabcd->lkijabcd",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("ijklabcd->lkjiabcd",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./24.) * bra
			term += -(1./24.) * np.einsum("ijklabcd->ijklabdc",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijklabcd->ijklacbd",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijklabcd->ijklacdb",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijklabcd->ijkladbc",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijklabcd->ijkladcb",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijklabcd->ijklbacd",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijklabcd->ijklbadc",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijklabcd->ijklbcad",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijklabcd->ijklbcda",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijklabcd->ijklbdac",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijklabcd->ijklbdca",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijklabcd->ijklcabd",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijklabcd->ijklcadb",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijklabcd->ijklcbad",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijklabcd->ijklcbda",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijklabcd->ijklcdab",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijklabcd->ijklcdba",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijklabcd->ijkldabc",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijklabcd->ijkldacb",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijklabcd->ijkldbac",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijklabcd->ijkldbca",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("ijklabcd->ijkldcab",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("ijklabcd->ijkldcba",bra,optimize="optimal")
			del bra 
			return term
		case "ooovovvv":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("ijkalbcd->ikjalbcd",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijkalbcd->jikalbcd",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijkalbcd->jkialbcd",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("ijkalbcd->kijalbcd",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("ijkalbcd->kjialbcd",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("ijkalbcd->ijkalbdc",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijkalbcd->ijkalcbd",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijkalbcd->ijkalcdb",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijkalbcd->ijkaldbc",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijkalbcd->ijkaldcb",bra,optimize="optimal")
			del bra 
			return term 
		case "oovvoovv":
			bra +=  0.250 * term 
			bra += -0.250 * np.einsum("ijabklcd->jiabklcd",term,optimize="optimal")
			bra += -0.250 * np.einsum("ijabklcd->ijbaklcd",term,optimize="optimal")
			bra +=  0.250 * np.einsum("ijabklcd->jibaklcd",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.250 * bra 
			term += -0.250 * np.einsum("ijabklcd->ijablkcd",bra,optimize="optimal")
			term += -0.250 * np.einsum("ijabklcd->ijabkldc",bra,optimize="optimal")
			term +=  0.250 * np.einsum("ijabklcd->ijablkdc",bra,optimize="optimal")
			del bra 
			return term 
		case "ovvvooov":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("iabcjkld->iacbjkld",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("iabcjkld->ibacjkld",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("iabcjkld->ibcajkld",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("iabcjkld->icabjkld",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("iabcjkld->icbajkld",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("iabcjkld->iabcjlkd",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("iabcjkld->iabckjld",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("iabcjkld->iabckljd",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("iabcjkld->iabcljkd",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("iabcjkld->iabclkjd",bra,optimize="optimal")
			del bra
			return term 
		case "vvvvoooo":
			bra +=  (1./24.) * term
			bra += -(1./24.) * np.einsum("abcdijkl->abdcijkl",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("abcdijkl->acbdijkl",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("abcdijkl->acdbijkl",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("abcdijkl->adbcijkl",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("abcdijkl->adcbijkl",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("abcdijkl->bacdijkl",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("abcdijkl->badcijkl",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("abcdijkl->bcadijkl",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("abcdijkl->bcdaijkl",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("abcdijkl->bdacijkl",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("abcdijkl->bdcaijkl",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("abcdijkl->cabdijkl",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("abcdijkl->cadbijkl",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("abcdijkl->cbadijkl",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("abcdijkl->cbdaijkl",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("abcdijkl->cdabijkl",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("abcdijkl->cdbaijkl",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("abcdijkl->dabcijkl",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("abcdijkl->dacbijkl",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("abcdijkl->dbacijkl",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("abcdijkl->dbcaijkl",term,optimize="optimal")
			bra += -(1./24.) * np.einsum("abcdijkl->dcabijkl",term,optimize="optimal")
			bra +=  (1./24.) * np.einsum("abcdijkl->dcbaijkl",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./24.) * bra 
			term += -(1./24.) * np.einsum("abcdijkl->abcdijlk",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("abcdijkl->abcdikjl",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("abcdijkl->abcdiklj",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("abcdijkl->abcdiljk",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("abcdijkl->abcdilkj",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("abcdijkl->abcdjikl",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("abcdijkl->abcdjilk",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("abcdijkl->abcdjkil",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("abcdijkl->abcdjkli",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("abcdijkl->abcdjlik",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("abcdijkl->abcdjlki",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("abcdijkl->abcdkijl",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("abcdijkl->abcdkilj",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("abcdijkl->abcdkjil",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("abcdijkl->abcdkjli",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("abcdijkl->abcdklij",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("abcdijkl->abcdklji",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("abcdijkl->abcdlijk",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("abcdijkl->abcdlikj",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("abcdijkl->abcdljik",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("abcdijkl->abcdljki",bra,optimize="optimal")
			term += -(1./24.) * np.einsum("abcdijkl->abcdlkij",bra,optimize="optimal")
			term +=  (1./24.) * np.einsum("abcdijkl->abcdlkji",bra,optimize="optimal")
			del bra
			return term
		case "oovvovvv":
			bra +=  0.250 * term 
			bra += -0.250 * np.einsum("ijabkcde->jiabkcde",term,optimize="optimal")
			bra += -0.250 * np.einsum("ijabkcde->ijbakcde",term,optimize="optimal")
			bra +=  0.250 * np.einsum("ijabkcde->jibakcde",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  (1./6.) * bra 
			term += -(1./6.) * np.einsum("ijabkcde->ijabkced",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijabkcde->ijabkdce",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijabkcde->ijabkdec",bra,optimize="optimal")
			term +=  (1./6.) * np.einsum("ijabkcde->ijabkecd",bra,optimize="optimal")
			term += -(1./6.) * np.einsum("ijabkcde->ijabkedc",bra,optimize="optimal")
			del bra
			return term 
		case "ovvvoovv":
			bra +=  (1./6.) * term 
			bra += -(1./6.) * np.einsum("iabcjkde->iacbjkde",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("iabcjkde->ibacjkde",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("iabcjkde->ibcajkde",term,optimize="optimal")
			bra +=  (1./6.) * np.einsum("iabcjkde->icabjkde",term,optimize="optimal")
			bra += -(1./6.) * np.einsum("iabcjkde->icbajkde",term,optimize="optimal")
			term = np.zeros_like(bra)
			term +=  0.250 * bra 
			term += -0.250 * np.einsum("iabcjkde->iabckjde",bra,optimize="optimal")
			term += -0.250 * np.einsum("iabcjkde->iabcjked",bra,optimize="optimal")
			term +=  0.250 * np.einsum("iabcjkde->iabckjed",bra,optimize="optimal")
			del bra 
			return term 

def one_body_to_op(one_body_mat,n_occ,n_orb):
	"""Creates an OpenFermion FermionOperator from a one-body matrix
	
	Parameters
	----------
	one_body_mat : np.ndarray
		Fock-like matrix
	n_occ : int
		number of occupied orbitals (n_a + n_b)
	n_orb : int
		total number of spatial orbitals

	Returns
	-------
	one_body_op : of.FermionOperator
		particle-hole normal-ordered one-body operator 
	"""
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

def two_body_to_op(two_body_tens,n_occ,n_orb):
	"""Creates an OpenFermion FermionOperator from a two-body tensor
	
	Parameters
	----------
	two_body_tens : np.ndarray
		two-body tensor
	n_occ : int
		number of occupied orbitals (n_a + n_b)
	n_orb : int
		total number of spatial orbitals

	Returns
	-------
	two_body_op : of.FermionOperator
		particle-hole normal-ordered two-body operator 
	"""
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

def three_body_to_op(three_body_tens,n_occ,n_orb):
	"""Creates an OpenFermion FermionOperator from a three-body tensor
	
	Parameters
	----------
	three_body_tens : np.ndarray
		three-body tensor
	n_occ : int
		number of occupied orbitals (n_a + n_b)
	n_orb : int
		total number of spatial orbitals

	Returns
	-------
	three_body_op : of.FermionOperator
		particle-hole normal-ordered three-body operator 
	"""
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
	for i in range(0,n_occ):
		for j in range(0,n_occ):
			for k in range(0,n_occ):
				for a in range(n_occ,2*n_orb):
					for b in range(n_occ,2*n_orb):
						for c in range(n_occ,2*n_orb):
							# OOO|VVV
							three_body_op += of.FermionOperator(((c,0),(b,0),(a,0),(i,1),(j,1),(k,1)), -three_body_tens[i,j,k,a,b,c])
							# OOV|OVV
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)),  three_body_tens[i,j,a,k,b,c])
							# OVO|OVV
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)), -three_body_tens[i,a,j,k,b,c])
							# VOO|OVV
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)),  three_body_tens[a,i,j,k,b,c])
							# OOV|VOV
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)), -three_body_tens[i,j,a,b,k,c])
							# OVO|VOV
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)),  three_body_tens[i,a,j,b,k,c])
							# VOO|VOV
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)), -three_body_tens[a,i,j,b,k,c])
							# OOV|VVO
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)),  three_body_tens[i,j,a,b,c,k])
							# OVO|VVO
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)), -three_body_tens[i,a,j,b,c,k])
							# VOO|VVO
							three_body_op += of.FermionOperator(((a,1),(k,0),(i,1),(j,1),(c,0),(b,0)),  three_body_tens[a,i,j,b,c,k])
							# OVV|OOV
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)),  three_body_tens[i,a,b,j,k,c])
							# VOV|OOV
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)), -three_body_tens[a,i,b,j,k,c])
							# VVO|OOV
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)),  three_body_tens[a,b,i,j,k,c])
							# OVV|OVO
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)), -three_body_tens[i,a,b,j,c,k])
							# VOV|OVO
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)),  three_body_tens[a,i,b,j,c,k])
							# VVO|OVO
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)), -three_body_tens[a,b,i,j,c,k])
							# OVV|VOO
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)),  three_body_tens[i,a,b,c,j,k])
							# VOV|VOO
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)), -three_body_tens[a,i,b,c,j,k])
							# VVO|VOO
							three_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(j,0),(i,1),(c,0)),  three_body_tens[a,b,i,c,j,k])
							# VVV|OOO
							three_body_op += of.FermionOperator(((a,1),(b,1),(c,1),(k,0),(j,0),(i,0)),  three_body_tens[a,b,c,i,j,k])
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

def four_body_to_op(four_body_ten,n_occ,n_orb):
	"""Creates an OpenFermion FermionOperator from a four-body tensor
	
	Parameters
	----------
	four_body_tens : np.ndarray
		four-body tensor
	n_occ : int
		number of occupied orbitals (n_a + n_b)
	n_orb : int
		total number of spatial orbitals

	Returns
	-------
	four_body_op : of.FermionOperator
		particle-hole normal-ordered four-body operator 
	"""
	four_body_op = of.FermionOperator()
	for i in range(0,n_occ):
		for j in range(0,n_occ):
			for k in range(0,n_occ):
				for l in range(0,n_occ):
					for m in range(0,n_occ):
						for n in range(0,n_occ):
							for o in range(0,n_occ):
								for p in range(0,n_occ):
									# OOOO|OOOO
									four_body_op += of.FermionOperator(((p,0),(o,0),(n,0),(m,0),(i,1),(j,1),(k,1),(l,1)),  four_body_ten[i,j,k,l,m,n,o,p])
	for i in range(0,n_occ):
		for j in range(0,n_occ):
			for k in range(0,n_occ):
				for l in range(0,n_occ):
					for m in range(0,n_occ):
						for n in range(0,n_occ):
							for o in range(0,n_occ):
								for a in range(n_occ,2*n_orb):
									# OOOO|{OOOV}
									four_body_op += of.FermionOperator(((o,0),(n,0),(m,0),(i,1),(j,1),(k,1),(l,1),(a,0)), 
																		-four_body_ten[i,j,k,l,m,n,o,a]
																		+four_body_ten[i,j,k,l,m,n,a,o]
																		-four_body_ten[i,j,k,l,m,a,n,o]
																		+four_body_ten[i,j,k,l,a,m,n,o])
									# {OOOV}|OOOO
									four_body_op += of.FermionOperator(((a,1),(o,0),(n,0),(m,0),(l,0),(i,1),(j,1),(k,1)), 
																		-four_body_ten[i,j,k,a,l,m,n,o]
																		+four_body_ten[i,j,a,k,l,m,n,o]
																		-four_body_ten[i,a,j,k,l,m,n,o]
																		+four_body_ten[a,i,j,k,l,m,n,o])
	for i in range(0,n_occ):
		for j in range(0,n_occ):
			for k in range(0,n_occ):
				for l in range(0,n_occ):
					for m in range(0,n_occ):
						for n in range(0,n_occ):
							for a in range(n_occ,2*n_orb):
								for b in range(n_occ,2*n_orb):
									# OOOO|{OOVV}
									four_body_op += of.FermionOperator(((n,0),(m,0),(i,1),(j,1),(k,1),(l,1),(b,0),(a,0)),
																		 four_body_ten[i,j,k,l,m,n,a,b]
																		-four_body_ten[i,j,k,l,m,a,n,b]
																		+four_body_ten[i,j,k,l,a,m,n,b]
																		+four_body_ten[i,j,k,l,m,a,b,n]
																		-four_body_ten[i,j,k,l,a,m,b,n]
																		+four_body_ten[i,j,k,l,a,b,m,n])
									# {OOOV}|{OOOV}
									four_body_op += of.FermionOperator(((a,1),(n,0),(m,0),(l,0),(i,1),(j,1),(k,1),(b,0)),
																		-four_body_ten[i,j,k,a,l,m,n,b]
																		+four_body_ten[i,j,k,a,l,m,b,n]
																		-four_body_ten[i,j,k,a,l,b,m,n]
																		+four_body_ten[i,j,k,a,b,l,m,n]

																		+four_body_ten[i,j,a,k,l,m,n,b]
																		-four_body_ten[i,j,a,k,l,m,b,n]
																		+four_body_ten[i,j,a,k,l,b,m,n]
																		-four_body_ten[i,j,a,k,b,l,m,n]

																		-four_body_ten[i,a,j,k,l,m,n,b]
																		+four_body_ten[i,a,j,k,l,m,b,n]
																		-four_body_ten[i,a,j,k,l,b,m,n]
																		+four_body_ten[i,a,j,k,b,l,m,n]

																		+four_body_ten[a,i,j,k,l,m,n,b]
																		-four_body_ten[a,i,j,k,l,m,b,n]
																		+four_body_ten[a,i,j,k,l,b,m,n]
																		-four_body_ten[a,i,j,k,b,l,m,n])
									# {OOVV}|OOOO
									four_body_op += of.FermionOperator(((a,1),(b,1),(n,0),(m,0),(l,0),(k,0),(i,1),(j,1)),
																		 four_body_ten[i,j,a,b,k,l,m,n]
																		-four_body_ten[i,a,j,b,k,l,m,n]
																		+four_body_ten[a,i,j,b,k,l,m,n]
																		+four_body_ten[i,a,b,j,k,l,m,n]
																		-four_body_ten[a,i,b,j,k,l,m,n]
																		+four_body_ten[a,b,i,j,k,l,m,n])
	for i in range(0,n_occ):
		for j in range(0,n_occ):
			for k in range(0,n_occ):
				for l in range(0,n_occ):
					for m in range(0,n_occ):
						for a in range(n_occ,2*n_orb):
							for b in range(n_occ,2*n_orb):
								for c in range(n_occ,2*n_orb):
									# OOOO|{OVVV}
									four_body_op += of.FermionOperator(((m,0),(i,1),(j,1),(k,1),(l,1),(c,0),(b,0),(a,0)),
																		-four_body_ten[i,j,k,l,m,a,b,c]
																		+four_body_ten[i,j,k,l,a,m,b,c]
																		-four_body_ten[i,j,k,l,a,b,m,c]
																		+four_body_ten[i,j,k,l,a,b,c,m])
									# {OOOV}|{OOVV}
									four_body_op += of.FermionOperator(((a,1),(m,0),(l,0),(i,1),(j,1),(k,1),(c,0),(b,0)),
																		-four_body_ten[i,j,k,a,l,m,b,c]
																		+four_body_ten[i,j,k,a,l,b,m,c]
																		-four_body_ten[i,j,k,a,b,l,m,c]
																		-four_body_ten[i,j,k,a,l,b,c,m]
																		+four_body_ten[i,j,k,a,b,l,c,m]
																		-four_body_ten[i,j,k,a,b,c,l,m]
																		+four_body_ten[i,j,a,k,l,m,b,c]
																		-four_body_ten[i,j,a,k,l,b,m,c]
																		+four_body_ten[i,j,a,k,b,l,m,c]
																		+four_body_ten[i,j,a,k,l,b,c,m]
																		-four_body_ten[i,j,a,k,b,l,c,m]
																		+four_body_ten[i,j,a,k,b,c,l,m]
																		-four_body_ten[i,a,j,k,l,m,b,c]
																		+four_body_ten[i,a,j,k,l,b,m,c]
																		-four_body_ten[i,a,j,k,b,l,m,c]
																		-four_body_ten[i,a,j,k,l,b,c,m]
																		+four_body_ten[i,a,j,k,b,l,c,m]
																		-four_body_ten[i,a,j,k,b,c,l,m]
																		+four_body_ten[a,i,j,k,l,m,b,c]
																		-four_body_ten[a,i,j,k,l,b,m,c]
																		+four_body_ten[a,i,j,k,b,l,m,c]
																		+four_body_ten[a,i,j,k,l,b,c,m]
																		-four_body_ten[a,i,j,k,b,l,c,m]
																		+four_body_ten[a,i,j,k,b,c,l,m])
									# {OOVV}|{OOOV}
									four_body_op += of.FermionOperator(((a,1),(b,1),(m,0),(l,0),(k,0),(i,1),(j,1),(c,0)),
																		-four_body_ten[i,j,a,b,k,l,m,c]
																		+four_body_ten[i,j,a,b,k,l,c,m]
																		-four_body_ten[i,j,a,b,k,c,l,m]
																		+four_body_ten[i,j,a,b,c,k,l,m]
																		+four_body_ten[i,a,j,b,k,l,m,c]
																		-four_body_ten[i,a,j,b,k,l,c,m]
																		+four_body_ten[i,a,j,b,k,c,l,m]
																		-four_body_ten[i,a,j,b,c,k,l,m]
																		-four_body_ten[a,i,j,b,k,l,m,c]
																		+four_body_ten[a,i,j,b,k,l,c,m]
																		-four_body_ten[a,i,j,b,k,c,l,m]
																		+four_body_ten[a,i,j,b,c,k,l,m]
																		-four_body_ten[i,a,b,j,k,l,m,c]
																		+four_body_ten[i,a,b,j,k,l,c,m]
																		-four_body_ten[i,a,b,j,k,c,l,m]
																		+four_body_ten[i,a,b,j,c,k,l,m]
																		+four_body_ten[a,i,b,j,k,l,m,c]
																		-four_body_ten[a,i,b,j,k,l,c,m]
																		+four_body_ten[a,i,b,j,k,c,l,m]
																		-four_body_ten[a,i,b,j,c,k,l,m]
																		-four_body_ten[a,b,i,j,k,l,m,c]
																		+four_body_ten[a,b,i,j,k,l,c,m]
																		-four_body_ten[a,b,i,j,k,c,l,m]
																		+four_body_ten[a,b,i,j,c,k,l,m])
									# {OVVV}|OOOO
									four_body_op += of.FermionOperator(((a,1),(b,1),(c,1),(m,0),(l,0),(k,0),(j,0),(i,1)),
																		-four_body_ten[i,a,b,c,j,k,l,m]
																		+four_body_ten[a,i,b,c,j,k,l,m]
																		-four_body_ten[a,b,i,c,j,k,l,m]
																		+four_body_ten[a,b,c,i,j,k,l,m])
	for i in range(0,n_occ):
		for j in range(0,n_occ):
			for k in range(0,n_occ):
				for l in range(0,n_occ):
					for a in range(n_occ,2*n_orb):
						for b in range(n_occ,2*n_orb):
							for c in range(n_occ,2*n_orb):
								for d in range(n_occ,2*n_orb):
									# OOOO|VVVV
									four_body_op += of.FermionOperator(((i,1),(j,1),(k,1),(l,1),(d,0),(c,0),(b,0),(a,0)),  four_body_ten[i,j,k,l,a,b,c,d])
									# {OOOV}|{OVVV}
									four_body_op += of.FermionOperator(((a,1),(l,0),(i,1),(j,1),(k,1),(d,0),(c,0),(b,0)),
																		-four_body_ten[i,j,k,a,l,b,c,d]
																		+four_body_ten[i,j,k,a,b,l,c,d]
																		-four_body_ten[i,j,k,a,b,c,l,d]
																		+four_body_ten[i,j,k,a,b,c,d,l]
																		+four_body_ten[i,j,a,k,l,b,c,d]
																		-four_body_ten[i,j,a,k,b,l,c,d]
																		+four_body_ten[i,j,a,k,b,c,l,d]
																		-four_body_ten[i,j,a,k,b,c,d,l]
																		-four_body_ten[i,a,j,k,l,b,c,d]
																		+four_body_ten[i,a,j,k,b,l,c,d]
																		-four_body_ten[i,a,j,k,b,c,l,d]
																		+four_body_ten[i,a,j,k,b,c,d,l]
																		+four_body_ten[a,i,j,k,l,b,c,d]
																		-four_body_ten[a,i,j,k,b,l,c,d]
																		+four_body_ten[a,i,j,k,b,c,l,d]
																		-four_body_ten[a,i,j,k,b,c,d,l])
									# {OOVV}|{OOVV}
									four_body_op += of.FermionOperator(((a,1),(b,1),(l,0),(k,0),(i,1),(j,1),(d,0),(c,0)),
																		 four_body_ten[i,j,a,b,k,l,c,d]
																		-four_body_ten[i,j,a,b,k,c,l,d]
																		+four_body_ten[i,j,a,b,c,k,l,d]
																		+four_body_ten[i,j,a,b,k,c,d,l]
																		-four_body_ten[i,j,a,b,c,k,d,l]
																		+four_body_ten[i,j,a,b,c,d,k,l]
																		-four_body_ten[i,a,j,b,k,l,c,d]
																		+four_body_ten[i,a,j,b,k,c,l,d]
																		-four_body_ten[i,a,j,b,c,k,l,d]
																		-four_body_ten[i,a,j,b,k,c,d,l]
																		+four_body_ten[i,a,j,b,c,k,d,l]
																		-four_body_ten[i,a,j,b,c,d,k,l]
																		+four_body_ten[a,i,j,b,k,l,c,d]
																		-four_body_ten[a,i,j,b,k,c,l,d]
																		+four_body_ten[a,i,j,b,c,k,l,d]
																		+four_body_ten[a,i,j,b,k,c,d,l]
																		-four_body_ten[a,i,j,b,c,k,d,l]
																		+four_body_ten[a,i,j,b,c,d,k,l]
																		+four_body_ten[i,a,b,j,k,l,c,d]
																		-four_body_ten[i,a,b,j,k,c,l,d]
																		+four_body_ten[i,a,b,j,c,k,l,d]
																		+four_body_ten[i,a,b,j,k,c,d,l]
																		-four_body_ten[i,a,b,j,c,k,d,l]
																		+four_body_ten[i,a,b,j,c,d,k,l]
																		-four_body_ten[a,i,b,j,k,l,c,d]
																		+four_body_ten[a,i,b,j,k,c,l,d]
																		-four_body_ten[a,i,b,j,c,k,l,d]
																		-four_body_ten[a,i,b,j,k,c,d,l]
																		+four_body_ten[a,i,b,j,c,k,d,l]
																		-four_body_ten[a,i,b,j,c,d,k,l]
																		+four_body_ten[a,b,i,j,k,l,c,d]
																		-four_body_ten[a,b,i,j,k,c,l,d]
																		+four_body_ten[a,b,i,j,c,k,l,d]
																		+four_body_ten[a,b,i,j,k,c,d,l]
																		-four_body_ten[a,b,i,j,c,k,d,l]
																		+four_body_ten[a,b,i,j,c,d,k,l])
									# {OVVV}|{OOOV}
									four_body_op += of.FermionOperator(((a,1),(b,1),(c,1),(l,0),(k,0),(j,0),(i,1),(d,0)),
																		-four_body_ten[i,a,b,c,j,k,l,d]
																		+four_body_ten[i,a,b,c,j,k,d,l]
																		-four_body_ten[i,a,b,c,j,d,k,l]
																		+four_body_ten[i,a,b,c,d,j,k,l]
																		+four_body_ten[a,i,b,c,j,k,l,d]
																		-four_body_ten[a,i,b,c,j,k,d,l]
																		+four_body_ten[a,i,b,c,j,d,k,l]
																		-four_body_ten[a,i,b,c,d,j,k,l]
																		-four_body_ten[a,b,i,c,j,k,l,d]
																		+four_body_ten[a,b,i,c,j,k,d,l]
																		-four_body_ten[a,b,i,c,j,d,k,l]
																		+four_body_ten[a,b,i,c,d,j,k,l]
																		+four_body_ten[a,b,c,i,j,k,l,d]
																		-four_body_ten[a,b,c,i,j,k,d,l]
																		+four_body_ten[a,b,c,i,j,d,k,l]
																		-four_body_ten[a,b,c,i,d,j,k,l])
									# VVVV|OOOO
									four_body_op += of.FermionOperator(((a,1),(b,1),(c,1),(d,1),(l,0),(k,0),(j,0),(i,0)),  four_body_ten[a,b,c,d,i,j,k,l])
	for i in range(0,n_occ):
		for j in range(0,n_occ):
			for k in range(0,n_occ):
				for a in range(n_occ,2*n_orb):
					for b in range(n_occ,2*n_orb):
						for c in range(n_occ,2*n_orb):
							for d in range(n_occ,2*n_orb):
								for e in range(n_occ,2*n_orb):
									# {OOOV}|VVVV
									four_body_op += of.FermionOperator(((a,1),(i,1),(j,1),(k,1),(e,0),(d,0),(c,0),(b,0)),
																		-four_body_ten[i,j,k,a,b,c,d,e]
																		+four_body_ten[i,j,a,k,b,c,d,e]
																		-four_body_ten[i,a,j,k,b,c,d,e]
																		+four_body_ten[a,i,j,k,b,c,d,e])
									# {OOVV}|{OVVV}
									four_body_op += of.FermionOperator(((a,1),(b,1),(k,0),(i,1),(j,1),(e,0),(d,0),(c,0)),
																		-four_body_ten[i,j,a,b,k,c,d,e]
																		+four_body_ten[i,j,a,b,c,k,d,e]
																		-four_body_ten[i,j,a,b,c,d,k,e]
																		+four_body_ten[i,j,a,b,c,d,e,k]
																		+four_body_ten[i,a,j,b,k,c,d,e]
																		-four_body_ten[i,a,j,b,c,k,d,e]
																		+four_body_ten[i,a,j,b,c,d,k,e]
																		-four_body_ten[i,a,j,b,c,d,e,k]
																		-four_body_ten[a,i,j,b,k,c,d,e]
																		+four_body_ten[a,i,j,b,c,k,d,e]
																		-four_body_ten[a,i,j,b,c,d,k,e]
																		+four_body_ten[a,i,j,b,c,d,e,k]
																		-four_body_ten[i,a,b,j,k,c,d,e]
																		+four_body_ten[i,a,b,j,c,k,d,e]
																		-four_body_ten[i,a,b,j,c,d,k,e]
																		+four_body_ten[i,a,b,j,c,d,e,k]
																		+four_body_ten[a,i,b,j,k,c,d,e]
																		-four_body_ten[a,i,b,j,c,k,d,e]
																		+four_body_ten[a,i,b,j,c,d,k,e]
																		-four_body_ten[a,i,b,j,c,d,e,k]
																		-four_body_ten[a,b,i,j,k,c,d,e]
																		+four_body_ten[a,b,i,j,c,k,d,e]
																		-four_body_ten[a,b,i,j,c,d,k,e]
																		+four_body_ten[a,b,i,j,c,d,e,k])
									# {OVVV}|{OOVV}
									four_body_op += of.FermionOperator(((a,1),(b,1),(c,1),(k,0),(j,0),(i,1),(e,0),(d,0)),
																		-four_body_ten[i,a,b,c,j,k,d,e]
																		+four_body_ten[i,a,b,c,j,d,k,e]
																		-four_body_ten[i,a,b,c,d,j,k,e]
																		-four_body_ten[i,a,b,c,j,d,e,k]
																		+four_body_ten[i,a,b,c,d,j,e,k]
																		-four_body_ten[i,a,b,c,d,e,j,k]
																		+four_body_ten[a,i,b,c,j,k,d,e]
																		-four_body_ten[a,i,b,c,j,d,k,e]
																		+four_body_ten[a,i,b,c,d,j,k,e]
																		+four_body_ten[a,i,b,c,j,d,e,k]
																		-four_body_ten[a,i,b,c,d,j,e,k]
																		+four_body_ten[a,i,b,c,d,e,j,k]
																		-four_body_ten[a,b,i,c,j,k,d,e]
																		+four_body_ten[a,b,i,c,j,d,k,e]
																		-four_body_ten[a,b,i,c,d,j,k,e]
																		-four_body_ten[a,b,i,c,j,d,e,k]
																		+four_body_ten[a,b,i,c,d,j,e,k]
																		-four_body_ten[a,b,i,c,d,e,j,k]
																		+four_body_ten[a,b,c,i,j,k,d,e]
																		-four_body_ten[a,b,c,i,j,d,k,e]
																		+four_body_ten[a,b,c,i,d,j,k,e]
																		+four_body_ten[a,b,c,i,j,d,e,k]
																		-four_body_ten[a,b,c,i,d,j,e,k]
																		+four_body_ten[a,b,c,i,d,e,j,k])
									# VVVV|{OOOV}
									four_body_op += of.FermionOperator(((a,1),(b,1),(c,1),(d,1),(k,0),(j,0),(i,0),(e,0)),
																		-four_body_ten[a,b,c,d,i,j,k,e]
																		+four_body_ten[a,b,c,d,i,j,e,k]
																		-four_body_ten[a,b,c,d,i,e,j,k]
																		+four_body_ten[a,b,c,d,e,i,j,k])
	for i in range(0,n_occ):
		for j in range(0,n_occ):
			for a in range(n_occ,2*n_orb):
				for b in range(n_occ,2*n_orb):
					for c in range(n_occ,2*n_orb):
						for d in range(n_occ,2*n_orb):
							for e in range(n_occ,2*n_orb):
								for f in range(n_occ,2*n_orb):
									# {OOVV}|VVVV
									four_body_op += of.FermionOperator(((a,1),(b,1),(i,1),(j,1),(f,0),(e,0),(d,0),(c,0)),
																		 four_body_ten[i,j,a,b,c,d,e,f]
																		-four_body_ten[i,a,j,b,c,d,e,f]
																		+four_body_ten[a,i,j,b,c,d,e,f]
																		+four_body_ten[i,a,b,j,c,d,e,f]
																		-four_body_ten[a,i,b,j,c,d,e,f]
																		+four_body_ten[a,b,i,j,c,d,e,f])
									# {OVVV}|{OVVV}
									four_body_op += of.FermionOperator(((a,1),(b,1),(c,1),(j,0),(i,1),(f,0),(e,0),(d,0)),
																		-four_body_ten[i,a,b,c,j,d,e,f]
																		+four_body_ten[i,a,b,c,d,j,e,f]
																		-four_body_ten[i,a,b,c,d,e,j,f]
																		+four_body_ten[i,a,b,c,d,e,f,j]
																		+four_body_ten[a,i,b,c,j,d,e,f]
																		-four_body_ten[a,i,b,c,d,j,e,f]
																		+four_body_ten[a,i,b,c,d,e,j,f]
																		-four_body_ten[a,i,b,c,d,e,f,j]
																		-four_body_ten[a,b,i,c,j,d,e,f]
																		+four_body_ten[a,b,i,c,d,j,e,f]
																		-four_body_ten[a,b,i,c,d,e,j,f]
																		+four_body_ten[a,b,i,c,d,e,f,j]
																		+four_body_ten[a,b,c,i,j,d,e,f]
																		-four_body_ten[a,b,c,i,d,j,e,f]
																		+four_body_ten[a,b,c,i,d,e,j,f]
																		-four_body_ten[a,b,c,i,d,e,f,j])
									# VVVV|{OOVV}
									four_body_op += of.FermionOperator(((a,1),(b,1),(c,1),(d,1),(j,0),(i,0),(f,0),(e,0)),
																		 four_body_ten[a,b,c,d,i,j,e,f]
																		-four_body_ten[a,b,c,d,i,e,j,f]
																		+four_body_ten[a,b,c,d,e,i,j,f]
																		+four_body_ten[a,b,c,d,i,e,f,j]
																		-four_body_ten[a,b,c,d,e,i,f,j]
																		+four_body_ten[a,b,c,d,e,f,i,j])
	for i in range(0,n_occ):
		for a in range(n_occ,2*n_orb):
			for b in range(n_occ,2*n_orb):
				for c in range(n_occ,2*n_orb):
					for d in range(n_occ,2*n_orb):
						for e in range(n_occ,2*n_orb):
							for f in range(n_occ,2*n_orb):
								for g in range(n_occ,2*n_orb):
									# {OVVV}|VVVV
									four_body_op += of.FermionOperator(((a,1),(b,1),(c,1),(i,1),(g,0),(f,0),(e,0),(d,0)),
																		-four_body_ten[i,a,b,c,d,e,f,g]
																		+four_body_ten[a,i,b,c,d,e,f,g]
																		-four_body_ten[a,b,i,c,d,e,f,g]
																		+four_body_ten[a,b,c,i,d,e,f,g])
									# VVVV|{OVVV}
									four_body_op += of.FermionOperator(((a,1),(b,1),(c,1),(d,1),(i,0),(g,0),(f,0),(e,0)),
																		-four_body_ten[a,b,c,d,i,e,f,g]
																		+four_body_ten[a,b,c,d,e,i,f,g]
																		-four_body_ten[a,b,c,d,e,f,i,g]
																		+four_body_ten[a,b,c,d,e,f,g,i])
	for a in range(n_occ,2*n_orb):
		for b in range(n_occ,2*n_orb):
			for c in range(n_occ,2*n_orb):
				for d in range(n_occ,2*n_orb): 
					for e in range(n_occ,2*n_orb):
						for f in range(n_occ,2*n_orb):
							for g in range(n_occ,2*n_orb):
								for h in range(n_occ,2*n_orb):
									# VVVV|VVVV
									four_body_op += of.FermionOperator(((a,1),(b,1),(c,1),(d,1),(h,0),(g,0),(f,0),(e,0)),  four_body_ten[a,b,c,d,e,f,g,h])
	return four_body_op 

def one_body_mat2dic(mat,n_occ,n_act,n_orb):
	"""Converts one-body matrix into a dictionary of component parts

	Parameters
	----------
	mat : np.ndarray
		one-body matrix
	n_occ : int
		number of occupied orbitals (n_a + n_b)
	n_act : int
		number of active spatial orbitals
	n_orb : int
		total number of spatial orbitals
	
	Returns
	-------
	dic : dict
		dictionary of one-body submatrices with dimensions of occupied (o), 
		internal virtual (v), and external virtual (V)
	"""
	dic = {
		"oo": mat[0:n_occ,0:n_occ],
		"ov": mat[0:n_occ,n_occ:2*n_act],
		"vo": mat[n_occ:2*n_act,0:n_occ],
		"vv": mat[n_occ:2*n_act,n_occ:2*n_act]
	}
	if(n_orb > n_act):
		dic["oV"] = mat[0:n_occ,2*n_act:2*n_orb]
		dic["Vo"] = mat[2*n_act:2*n_orb,0:n_occ]
		dic["vV"] = mat[n_occ:2*n_act,2*n_act:2*n_orb]
		dic["Vv"] = mat[2*n_act:2*n_orb,n_occ:2*n_act]
		dic["VV"] = mat[2*n_act:2*n_orb,2*n_act:2*n_orb]
	return dic 

def one_body_dic2mat(dic,n_occ,n_act,n_orb):
	"""Converts dictionary of one-body submatrices to full matrix

	Parameters
	----------
	dic : dict
		dictionary of one-body submatrices with dimensions of occupied (o), 
		internal virtual (v), and external virtual (V)
	n_occ : int
		number of occupied orbitals (n_a + n_b)
	n_act : int
		number of active spatial orbitals
	n_orb : int
		total number of spatial orbitals
	
	Returns
	-------
	mat : np.ndarray
		one-body matrix
	"""
	if(n_orb > n_act):
		mat = np.zeros((2*n_orb,2*n_orb))
		for key in dic.keys():
			match key:
				case "oo":
					mat[0:n_occ,0:n_occ] = dic["oo"]
				case "ov":
					mat[0:n_occ,n_occ:2*n_act] = dic["ov"]
				case "vo":
					mat[n_occ:2*n_act,0:n_occ] = dic["vo"]
				case "vv":
					mat[n_occ:2*n_act,n_occ:2*n_act] = dic["vv"]
				case "oV":
					mat[0:n_occ,2*n_act:2*n_orb] = dic["oV"]
				case "Vo":
					mat[2*n_act:2*n_orb,0:n_occ] = dic["Vo"]
				case "vV":  
					mat[n_occ:2*n_act,2*n_act:2*n_orb] = dic["vV"] 
				case "Vv":
					mat[2*n_act:2*n_orb,n_occ:2*n_act] = dic["Vv"] 
				case "VV":
					mat[2*n_act:2*n_orb,2*n_act:2*n_orb] = dic["VV"]
		return mat 
	else:
		mat = np.zeros((2*n_act,2*n_act))
		for key in dic.keys():
			match key:
				case "oo":
					mat[0:n_occ,0:n_occ] = dic["oo"]
				case "ov":
					mat[0:n_occ,n_occ:2*n_act] = dic["ov"]
				case "vo":
					mat[n_occ:2*n_act,0:n_occ] = dic["vo"]
				case "vv":
					mat[n_occ:2*n_act,n_occ:2*n_act] = dic["vv"]
		return mat 

def two_body_ten2dic(ten,n_occ,n_act,n_orb):
	"""Converts two-body tensor into a dictionary of component parts

	Parameters
	----------
	ten : np.ndarray
		two-body tensor
	n_occ : int
		number of occupied orbitals (n_a + n_b)
	n_act : int
		number of active spatial orbitals
	n_orb : int
		total number of spatial orbitals
	
	Returns
	-------
	dic : dict
		dictionary of two-body submatrices with dimensions of occupied (o), 
		internal virtual (v), and external virtual (V)
	"""
	dic = {
		"oooo": ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ],
		"ooov": ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act],
		"oovv": ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act],
		"ovoo": ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ],
		"ovov": ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act],
		"ovvv": ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
		"vvoo": ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ],
		"vvov": ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act],
		"vvvv": ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
	}
	if(n_orb > n_act):
		dic["oooV"] = ten[0:n_occ,0:n_occ,0:n_occ,2*n_act:2*n_orb]
		dic["oovV"] = ten[0:n_occ,0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb]
		dic["ooVV"] = ten[0:n_occ,0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb]

		dic["ovoV"] = ten[0:n_occ,n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb]
		dic["ovvV"] = ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb]
		dic["ovVV"] = ten[0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb]

		dic["oVoo"] = ten[0:n_occ,2*n_act:2*n_orb,0:n_occ,0:n_occ]
		dic["oVov"] = ten[0:n_occ,2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act]
		dic["oVoV"] = ten[0:n_occ,2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb]
		dic["oVvv"] = ten[0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act]
		dic["oVvV"] = ten[0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb]
		dic["oVVV"] = ten[0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb]

		dic["vvoV"] = ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb]
		dic["vvvV"] = ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb]
		dic["vvVV"] = ten[n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb]

		dic["vVoo"] = ten[n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ,0:n_occ]
		dic["vVov"] = ten[n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act]
		dic["vVoV"] = ten[n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb]
		dic["vVvv"] = ten[n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act]
		dic["vVvV"] = ten[n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb]
		dic["vVVV"] = ten[n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb]

		dic["VVoo"] = ten[2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ,0:n_occ]
		dic["VVov"] = ten[2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act]
		dic["VVoV"] = ten[2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb]
		dic["VVvv"] = ten[2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act]
		dic["VVvV"] = ten[2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb]
		dic["VVVV"] = ten[2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb]
	return dic 

def two_body_dic2ten(dic,n_occ,n_act,n_orb):
	"""Converts dictionary of two-body subtensors to full tensor

	Parameters
	----------
	dic : dict
		dictionary of one-body submatrices with dimensions of occupied (o), 
		internal virtual (v), and external virtual (V)
	n_occ : int
		number of occupied orbitals (n_a + n_b)
	n_act : int
		number of active spatial orbitals
	n_orb : int
		total number of spatial orbitals
	
	Returns
	-------
	mat : np.ndarray
		one-body matrix
	"""
	if(n_orb > n_act):
		ten = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
		for key in dic.keys():
			match key:
				case "oooo":
					ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ] = dic["oooo"]
				case "ooov":
					ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] = dic["ooov"]
					ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijka->ijak",dic["ooov"],optimize="optimal")
				case "oovv":
					ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = dic["oovv"]
				case "ovoo":
					ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] = dic["ovoo"]
					ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("iajk->aijk",dic["ovoo"],optimize="optimal")
				case "ovov":
					ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  dic["ovov"]
					ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("iajb->iabj",dic["ovov"],optimize="optimal")
					ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("iajb->aijb",dic["ovov"],optimize="optimal")
					ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("iajb->aibj",dic["ovov"],optimize="optimal")
				case "ovvv":
					ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["ovvv"]
					ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iabc->aibc",dic["ovvv"],optimize="optimal") 
				case "vvoo":
					ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  dic["vvoo"]
				case "vvov":
					ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  dic["vvov"]
					ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("abic->abci",dic["vvov"],optimize="optimal")
				case "vvvv":
					ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["vvvv"]	
				case "oooV":
					ten[0:n_occ,0:n_occ,0:n_occ,2*n_act:2*n_orb] =  dic["oooV"]
					ten[0:n_occ,0:n_occ,2*n_act:2*n_orb,0:n_occ] = -np.einsum("ijkA->ijAk",dic["oooV"],optimize="optimal")
				case "oovV":
					ten[0:n_occ,0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb] =  dic["oovV"]
					ten[0:n_occ,0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act] = -np.einsum("ijaA->ijAa",dic["oovV"],optimize="optimal")
				case "ooVV":
					ten[0:n_occ,0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb] =  dic["ooVV"]
				case "ovoV":
					ten[0:n_occ,n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb] =  dic["ovoV"] 
					ten[n_occ:2*n_act,0:n_occ,0:n_occ,2*n_act:2*n_orb] = -np.einsum("iajA->aijA",dic["ovoV"],optimize="optimal")
					ten[0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ] = -np.einsum("iajA->iaAj",dic["ovoV"],optimize="optimal")
					ten[n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb,0:n_occ] =  np.einsum("iajA->aiAj",dic["ovoV"],optimize="optimal")
				case "ovvV":
					ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb] =  dic["ovvV"]
					ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb] = -np.einsum("iabA->aibA",dic["ovvV"],optimize="optimal")
					ten[0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act] = -np.einsum("iabA->iaAb",dic["ovvV"],optimize="optimal")
					ten[n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act] =  np.einsum("iabA->aiAb",dic["ovvV"],optimize="optimal")
				case "ovVV":
					ten[0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb] =  dic["ovVV"]
					ten[n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb] = -np.einsum("iaAB->aiAB",dic["ovVV"],optimize="optimal")
				case "oVoo":
					ten[0:n_occ,2*n_act:2*n_orb,0:n_occ,0:n_occ] =  dic["oVoo"]
					ten[2*n_act:2*n_orb,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("iAjk->Aijk",dic["oVoo"],optimize="optimal")
				case "oVov":
					ten[0:n_occ,2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act] =  dic["oVov"]
					ten[2*n_act:2*n_orb,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("iAja->Aija",dic["oVov"],optimize="optimal")
					ten[0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act,0:n_occ] = -np.einsum("iAja->iAaj",dic["oVov"],optimize="optimal") 
					ten[2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("iAja->Aiaj",dic["oVov"],optimize="optimal") 
				case "oVoV":
					ten[0:n_occ,2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb] =  dic["oVoV"]
					ten[2*n_act:2*n_orb,0:n_occ,0:n_occ,2*n_act:2*n_orb] = -np.einsum("iAjB->AijB",dic["oVoV"],optimize="optimal")
					ten[0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ] = -np.einsum("iAjB->iABj",dic["oVoV"],optimize="optimal")
					ten[2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb,0:n_occ] =  np.einsum("iAjB->AiBj",dic["oVoV"],optimize="optimal")
				case "oVvv":
					ten[0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act] =  dic["oVvv"]
					ten[2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iAab->Aiab",dic["oVvv"],optimize="optimal")
				case "oVvV":
					ten[0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb] =  dic["oVvV"]
					ten[2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act,2*n_act:2*n_orb] = -np.einsum("iAaB->AiaB",dic["oVvV"],optimize="optimal") 
					ten[0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act] = -np.einsum("iAaB->iABa",dic["oVvV"],optimize="optimal") 
					ten[2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb,n_occ:2*n_act] =  np.einsum("iAaB->AiBa",dic["oVvV"],optimize="optimal")
				case "oVVV":
					ten[0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb] =  dic["oVVV"]
					ten[2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb,2*n_act:2*n_orb] = -np.einsum("iABC->AiBC",dic["oVVV"],optimize="optimal")
				case "vvoV":
					ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb] =  dic["vvoV"]
					ten[n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ] = -np.einsum("abiA->abAi",dic["vvoV"],optimize="optimal")
				case "vvvV":
					ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb] =  dic["vvvV"]
					ten[n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act] = -np.einsum("abcA->abAc",dic["vvvV"],optimize="optimal")
				case "vvVV":
					ten[n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb] =  dic["vvVV"] 
				case "vVoo":
					ten[n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ,0:n_occ] =  dic["vVoo"]
					ten[2*n_act:2*n_orb,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("aAij->Aaij",dic["vVoo"],optimize="optimal") 
				case "vVov":
					ten[n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act] =  dic["vVov"]
					ten[2*n_act:2*n_orb,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("aAib->Aaib",dic["vVov"],optimize="optimal")
					ten[n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act,0:n_occ] = -np.einsum("aAib->aAbi",dic["vVov"],optimize="optimal")
					ten[2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("aAib->Aabi",dic["vVov"],optimize="optimal")
				case "vVoV":
					ten[n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb] =  dic["vVoV"]
					ten[2*n_act:2*n_orb,n_occ:2*n_act,0:n_occ,2*n_act:2*n_orb] = -np.einsum("aAiB->AaiB",dic["vVoV"],optimize="optimal")
					ten[n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ] = -np.einsum("aAiB->aABi",dic["vVoV"],optimize="optimal")
					ten[2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb,0:n_occ] =  np.einsum("aAiB->AaBi",dic["vVoV"],optimize="optimal")
				case "vVvv":
					ten[n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act] =  dic["vVvv"] 
					ten[2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("aAbc->Aabc",dic["vVvv"],optimize="optimal") 
				case "vVvV":
					ten[n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb] =  dic["vVvV"]
					ten[2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act,2*n_act:2*n_orb] = -np.einsum("aAbB->AabB",dic["vVvV"],optimize="optimal")
					ten[n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act] = -np.einsum("aAbB->aABb",dic["vVvV"],optimize="optimal") 
					ten[2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb,n_occ:2*n_act] =  np.einsum("aAbB->AaBb",dic["vVvV"],optimize="optimal") 
				case "vVVV":
					ten[n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb] =  dic["vVVV"]
					ten[2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb,2*n_act:2*n_orb] = -np.einsum("aABC->AaBC",dic["vVVV"],optimize="optimal") 
				case "VVoo":
					ten[2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ,0:n_occ] =  dic["VVoo"]
				case "VVov":
					ten[2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ,n_occ:2*n_act] =  dic["VVov"] 
					ten[2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act,0:n_occ] = -np.einsum("ABia->ABai",dic["VVov"],optimize="optimal") 
				case "VVoV":
					ten[2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ,2*n_act:2*n_orb] =  dic["VVoV"] 
					ten[2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb,0:n_occ] = -np.einsum("ABiC->ABCi",dic["VVoV"],optimize="optimal")
				case "VVvv":
					ten[2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act,n_occ:2*n_act] =  dic["VVvv"]
				case "VVvV":
					ten[2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act,2*n_act:2*n_orb] =  dic["VVvV"]
					ten[2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb,n_occ:2*n_act] = -np.einsum("ABaC->ABCa",dic["VVvV"],optimize="optimal")
				case "VVVV":
					ten[2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb,2*n_act:2*n_orb] = dic["VVVV"]
	else:
		ten = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act))
		for key in dic.keys():
			match key:
				case "oooo":
					ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ] = dic["oooo"]
				case "ooov":
					ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] = dic["ooov"]
					ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijka->ijak",dic["ooov"],optimize="optimal")
				case "oovv":
					ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = dic["oovv"]
				case "ovoo":
					ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] = dic["ovoo"]
					ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("iajk->aijk",dic["ovoo"],optimize="optimal")
				case "ovov":
					ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  dic["ovov"]
					ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("iajb->iabj",dic["ovov"],optimize="optimal")
					ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("iajb->aijb",dic["ovov"],optimize="optimal")
					ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("iajb->aibj",dic["ovov"],optimize="optimal")
				case "ovvv":
					ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["ovvv"]
					ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iabc->aibc",dic["ovvv"],optimize="optimal") 
				case "vvoo":
					ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  dic["vvoo"]
				case "vvov":
					ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  dic["vvov"]
					ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("abic->abci",dic["vvov"],optimize="optimal")
				case "vvvv":
					ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["vvvv"]	
	return ten 

def three_body_ten2dic(ten,n_occ,n_act):

	dic = {
		"oooooo": ten[0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ],
		"ooooov": ten[0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act],
		"oooovv": ten[0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act], 
		"ooovvv": ten[0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act], 
		"oovooo": ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ], 
		"oovoov": ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act], 
		"oovovv": ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act],
		"oovvvv": ten[0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act], 
		"ovvooo": ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ], 
		"ovvoov": ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act],
		"ovvovv": ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act],
		"ovvvvv": ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act],
		"vvvooo": ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ],
		"vvvoov": ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act],
		"vvvovv": ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act],
		"vvvvvv": ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act]
	}
	return dic

def three_body_dic2ten(dic,n_occ,n_act):
	ten = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))
	for key in dic.keys():
		match key:
			case "oooooo":
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ] =  dic["oooooo"]
			case "ooooov":
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] =  dic["ooooov"]
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijklma->ijklam",dic["ooooov"],optimize="optimal")
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijklma->ijkalm",dic["ooooov"],optimize="optimal")
			case "oooovv":
				ten[0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] =  dic["oooovv"]
				ten[0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] = -np.einsum("ijklab->ijkalb",dic["oooovv"],optimize="optimal")
				ten[0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] =  np.einsum("ijklab->ijkabl",dic["oooovv"],optimize="optimal")
			case "ooovvv":
				ten[0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] =  dic["ooovvv"]
			case "oovooo":
				ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ] =  dic["oovooo"]
				ten[0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ] = -np.einsum("ijaklm->iajklm",dic["oovooo"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ] =  np.einsum("ijaklm->aijklm",dic["oovooo"],optimize="optimal")
			case "oovoov":
				ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act] =  dic["oovoov"]
				ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ] = -np.einsum("ijaklb->ijakbl",dic["oovoov"],optimize="optimal")
				ten[0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ] =  np.einsum("ijaklb->ijabkl",dic["oovoov"],optimize="optimal")
				ten[0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act] = -np.einsum("ijaklb->iajklb",dic["oovoov"],optimize="optimal")
				ten[0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ] =  np.einsum("ijaklb->iajkbl",dic["oovoov"],optimize="optimal")
				ten[0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ] = -np.einsum("ijaklb->iajbkl",dic["oovoov"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act] =  np.einsum("ijaklb->aijklb",dic["oovoov"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ] = -np.einsum("ijaklb->aijkbl",dic["oovoov"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ] =  np.einsum("ijaklb->aijbkl",dic["oovoov"],optimize="optimal")
			case "oovovv":
				ten[0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] =  dic["oovovv"]
				ten[0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] = -np.einsum("ijakbc->ijabkc",dic["oovovv"],optimize="optimal")
				ten[0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] =  np.einsum("ijakbc->ijabck",dic["oovovv"],optimize="optimal")
				ten[0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] = -np.einsum("ijakbc->iajkbc",dic["oovovv"],optimize="optimal")
				ten[0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] =  np.einsum("ijakbc->iajbkc",dic["oovovv"],optimize="optimal")
				ten[0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] = -np.einsum("ijakbc->iajbck",dic["oovovv"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] =  np.einsum("ijakbc->aijkbc",dic["oovovv"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] = -np.einsum("ijakbc->aijbkc",dic["oovovv"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] =  np.einsum("ijakbc->aijbck",dic["oovovv"],optimize="optimal")
			case "oovvvv":
				ten[0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] =  dic["oovvvv"]
				ten[0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] = -np.einsum("ijabcd->iajbcd",dic["oovvvv"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] =  np.einsum("ijabcd->aijbcd",dic["oovvvv"],optimize="optimal")
			case "ovvooo":
				ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ] =  dic["ovvooo"]
				ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ] = -np.einsum("iabjkl->aibjkl",dic["ovvooo"],optimize="optimal")
				ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, 0:n_occ] =  np.einsum("iabjkl->abijkl",dic["ovvooo"],optimize="optimal")
			case "ovvoov":
				ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act] =  dic["ovvoov"]
				ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ] = -np.einsum("iabjkc->iabjck",dic["ovvoov"],optimize="optimal")
				ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ] =  np.einsum("iabjkc->iabcjk",dic["ovvoov"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act] = -np.einsum("iabjkc->aibjkc",dic["ovvoov"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ] =  np.einsum("iabjkc->aibjck",dic["ovvoov"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ] = -np.einsum("iabjkc->aibcjk",dic["ovvoov"],optimize="optimal")
				ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ, n_occ:2*n_act] =  np.einsum("iabjkc->abijkc",dic["ovvoov"],optimize="optimal")
				ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, 0:n_occ] = -np.einsum("iabjkc->abijck",dic["ovvoov"],optimize="optimal")
				ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, 0:n_occ] =  np.einsum("iabjkc->abicjk",dic["ovvoov"],optimize="optimal")
			case "ovvovv":
				ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] =  dic["ovvovv"]
				ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] = -np.einsum("iabjcd->iabcjd",dic["ovvovv"],optimize="optimal")
				ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] =  np.einsum("iabjcd->iabcdj",dic["ovvovv"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] = -np.einsum("iabjcd->aibjcd",dic["ovvovv"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] =  np.einsum("iabjcd->aibcjd",dic["ovvovv"],optimize="optimal")
				ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] = -np.einsum("iabjcd->aibcdj",dic["ovvovv"],optimize="optimal")
				ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] =  np.einsum("iabjcd->abijcd",dic["ovvovv"],optimize="optimal")
				ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] = -np.einsum("iabjcd->abicjd",dic["ovvovv"],optimize="optimal")
				ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] =  np.einsum("iabjcd->abicdj",dic["ovvovv"],optimize="optimal")
			case "ovvvvv":
				ten[0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] =  dic["ovvvvv"]
				ten[n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] = -np.einsum("iabcde->aibcde",dic["ovvvvv"],optimize="optimal")
				ten[n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] =  np.einsum("iabcde->abicde",dic["ovvvvv"],optimize="optimal")
			case "vvvooo":
				ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, 0:n_occ] =  dic["vvvooo"]
			case "vvvoov":
				ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ, n_occ:2*n_act] =  dic["vvvoov"]
				ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, 0:n_occ] = -np.einsum("abcijd->abcidj",dic["vvvoov"],optimize="optimal")
				ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, 0:n_occ] =  np.einsum("abcijd->abcdij",dic["vvvoov"],optimize="optimal")
			case "vvvovv":
				ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act, n_occ:2*n_act] =  dic["vvvovv"]
				ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ, n_occ:2*n_act] = -np.einsum("abcide->abcdie",dic["vvvovv"],optimize="optimal")
				ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, 0:n_occ] =  np.einsum("abcide->abcdei",dic["vvvovv"],optimize="optimal")
			case "vvvvvv":
				ten[n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act, n_occ:2*n_act] =  dic["vvvvvv"]
	return ten

def four_body_ten2dic(ten,n_occ,n_act):
	dic = {
		"oooooooo": ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ],
		"ooooooov": ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act],
		"oooooovv": ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act],
		"ooooovvv": ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
		"oooovvvv": ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
		"ooovoooo": ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ],
		"ooovooov": ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act],
		"ooovoovv": ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act],
		"ooovovvv": ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
		"ooovvvvv": ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
		"oovvoooo": ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ],
		"oovvooov": ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act],
		"oovvoovv": ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act],
		"oovvovvv": ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
		"oovvvvvv": ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
		"ovvvoooo": ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ],
		"ovvvooov": ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act],
		"ovvvoovv": ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act],
		"ovvvovvv": ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
		"ovvvvvvv": ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
		"vvvvoooo": ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ],
		"vvvvooov": ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act],
		"vvvvoovv": ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act],
		"vvvvovvv": ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
		"vvvvvvvv": ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act],
	}

def four_body_dic2ten(dic,n_occ,n_act):
	ten = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))
	for key in dic.keys():
		match key: 
			case "oooooooo":
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ] =  dic["oooooooo"]
			case "ooooooov":
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] =  dic["ooooooov"]
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijklmnoa->ijklmnao",dic["ooooooov"],optimize="optimal") #oooooovo
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijklmnoa->ijklmano",dic["ooooooov"],optimize="optimal") #ooooovoo
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("ijklmnoa->ijklamno",dic["ooooooov"],optimize="optimal") #oooovooo
			case "oooooovv":
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  dic["oooooovv"]
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("ijklmnab->ijklmanb",dic["oooooovv"],optimize="optimal") #ooooovov
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("ijklmnab->ijklamnb",dic["oooooovv"],optimize="optimal") #oooovoov 
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("ijklmnab->ijklmabn",dic["oooooovv"],optimize="optimal") #ooooovvo
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijklmnab->ijklambn",dic["oooooovv"],optimize="optimal") #oooovovo
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijklmnab->ijklabmn",dic["oooooovv"],optimize="optimal") #oooovvoo
			case "ooooovvv":
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["ooooovvv"]
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijklmabc->ijklambc",dic["ooooovvv"],optimize="optimal") #oooovovv
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("ijklmabc->ijklabmc",dic["ooooovvv"],optimize="optimal") #oooovvov
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("ijklmabc->ijklabcm",dic["ooooovvv"],optimize="optimal") #oooovvvo
			case "oooovvvv":
				ten[0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["oooovvvv"]
			case "ooovoooo":
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ] =  dic["ooovoooo"] 
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("ijkalmno->ijaklmno",dic["ooovoooo"],optimize="optimal") #oovooooo
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ] =  np.einsum("ijkalmno->iajklmno",dic["ooovoooo"],optimize="optimal") #ovoooooo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("ijkalmno->aijklmno",dic["ooovoooo"],optimize="optimal") #vooooooo
			case "ooovooov":
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] = dic["ooovooov"]
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijkalmnb->ijkalmbn",dic["ooovooov"],optimize="optimal")
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijkalmnb->ijkalbmn",dic["ooovooov"],optimize="optimal")
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("ijkalmnb->ijkablmn",dic["ooovooov"],optimize="optimal")

				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("ijkalmnb->ijaklmnb",dic["ooovooov"],optimize="optimal")
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("ijkalmnb->ijaklmbn",dic["ooovooov"],optimize="optimal")
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("ijkalmnb->ijaklbmn",dic["ooovooov"],optimize="optimal")
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] =  np.einsum("ijkalmnb->ijakblmn",dic["ooovooov"],optimize="optimal")

				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("ijkalmnb->iajklmnb",dic["ooovooov"],optimize="optimal")
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijkalmnb->iajklmbn",dic["ooovooov"],optimize="optimal")
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijkalmnb->iajklbmn",dic["ooovooov"],optimize="optimal")
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("ijkalmnb->iajkblmn",dic["ooovooov"],optimize="optimal")

				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("ijkalmnb->aijklmnb",dic["ooovooov"],optimize="optimal")
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("ijkalmnb->aijklmbn",dic["ooovooov"],optimize="optimal")
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("ijkalmnb->aijklbmn",dic["ooovooov"],optimize="optimal")
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] =  np.einsum("ijkalmnb->aijkblmn",dic["ooovooov"],optimize="optimal")
			case "ooovoovv":
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  dic["ooovoovv"]
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("ijkalmbc->ijkalbmc",dic["ooovoovv"],optimize="optimal") #ooovovov
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("ijkalmbc->ijkablmc",dic["ooovoovv"],optimize="optimal") #ooovvoov
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("ijkalmbc->ijkalbcm",dic["ooovoovv"],optimize="optimal") #ooovovvo
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijkalmbc->ijkablcm",dic["ooovoovv"],optimize="optimal") #ooovvovo
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijkalmbc->ijkabclm",dic["ooovoovv"],optimize="optimal") #ooovvvoo
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijkalmbc->ijaklmbc",dic["ooovoovv"],optimize="optimal") #oovooovv
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("ijkalmbc->ijaklbmc",dic["ooovoovv"],optimize="optimal") #oovoovov
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("ijkalmbc->ijakblmc",dic["ooovoovv"],optimize="optimal") #oovovoov
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("ijkalmbc->ijaklbcm",dic["ooovoovv"],optimize="optimal") #oovoovvo
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("ijkalmbc->ijakblcm",dic["ooovoovv"],optimize="optimal") #oovovovo
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("ijkalmbc->ijakbclm",dic["ooovoovv"],optimize="optimal") #oovovvoo
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijkalmbc->iajklmbc",dic["ooovoovv"],optimize="optimal") #ovoooovv
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("ijkalmbc->iajklbmc",dic["ooovoovv"],optimize="optimal") #ovooovov
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("ijkalmbc->iajkblmc",dic["ooovoovv"],optimize="optimal") #ovoovoov
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("ijkalmbc->iajklbcm",dic["ooovoovv"],optimize="optimal") #ovooovvo
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijkalmbc->iajkblcm",dic["ooovoovv"],optimize="optimal") #ovoovovo
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijkalmbc->iajkbclm",dic["ooovoovv"],optimize="optimal") #ovoovvoo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijkalmbc->aijklmbc",dic["ooovoovv"],optimize="optimal") #vooooovv
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("ijkalmbc->aijklbmc",dic["ooovoovv"],optimize="optimal") #voooovov
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("ijkalmbc->aijkblmc",dic["ooovoovv"],optimize="optimal") #vooovoov
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("ijkalmbc->aijklbcm",dic["ooovoovv"],optimize="optimal") #voooovvo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("ijkalmbc->aijkblcm",dic["ooovoovv"],optimize="optimal") #vooovovo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("ijkalmbc->aijkbclm",dic["ooovoovv"],optimize="optimal") #vooovvoo
			case "ooovovvv":
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["ooovovvv"]
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijkalbcd->ijkablcd",dic["ooovovvv"],optimize="optimal") #ooovvovv
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("ijkalbcd->ijkabcld",dic["ooovovvv"],optimize="optimal") #ooovvvov
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("ijkalbcd->ijkabcdl",dic["ooovovvv"],optimize="optimal") #ooovvvvo
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijkalbcd->ijaklbcd",dic["ooovovvv"],optimize="optimal") #oovovovv
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijkalbcd->ijakblcd",dic["ooovovvv"],optimize="optimal") #oovovovv
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("ijkalbcd->ijakbcld",dic["ooovovvv"],optimize="optimal") #oovovvov
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("ijkalbcd->ijakbcdl",dic["ooovovvv"],optimize="optimal") #oovovvvo
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijkalbcd->iajklbcd",dic["ooovovvv"],optimize="optimal") #ovoovovv
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijkalbcd->iajkblcd",dic["ooovovvv"],optimize="optimal") #ovoovovv
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("ijkalbcd->iajkbcld",dic["ooovovvv"],optimize="optimal") #ovoovvov
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("ijkalbcd->iajkbcdl",dic["ooovovvv"],optimize="optimal") #ovoovvvo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijkalbcd->aijklbcd",dic["ooovovvv"],optimize="optimal") #vooovovv
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijkalbcd->aijkblcd",dic["ooovovvv"],optimize="optimal") #vooovovv
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("ijkalbcd->aijkbcld",dic["ooovovvv"],optimize="optimal") #vooovvov
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("ijkalbcd->aijkbcdl",dic["ooovovvv"],optimize="optimal") #vooovvvo
			case "ooovvvvv":
				ten[0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["ooovvvvv"]
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijkabcde->ijakbcde",dic["ooovvvvv"],optimize="optimal") #oovovvvv
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijkabcde->iajkbcde",dic["ooovvvvv"],optimize="optimal") #ovoovvvv
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijkabcde->aijkbcde",dic["ooovvvvv"],optimize="optimal") #vooovvvv
			case "oovvoooo":
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ] =  dic["oovvoooo"]
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("ijabklmn->iajbklmn",dic["oovvoooo"],optimize="optimal") #ovovoooo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ] =  np.einsum("ijabklmn->aijbklmn",dic["oovvoooo"],optimize="optimal") #voovoooo
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ] =  np.einsum("ijabklmn->iabjklmn",dic["oovvoooo"],optimize="optimal") #ovvooooo
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("ijabklmn->aibjklmn",dic["oovvoooo"],optimize="optimal") #vovooooo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ] =  np.einsum("ijabklmn->abijklmn",dic["oovvoooo"],optimize="optimal") #vvoooooo
			case "oovvooov":
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] =  dic["oovvooov"]
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabklmc->ijabklcm",dic["oovvooov"],optimize="optimal") #oovvoovo
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijabklmc->ijabkclm",dic["oovvooov"],optimize="optimal") #oovvovoo
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("ijabklmc->ijabcklm",dic["oovvooov"],optimize="optimal") #oovvvooo
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("ijabklmc->iajbklmc",dic["oovvooov"],optimize="optimal") #ovovooov
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("ijabklmc->iajbklcm",dic["oovvooov"],optimize="optimal") #ovovoovo
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("ijabklmc->iajbkclm",dic["oovvooov"],optimize="optimal") #ovovovoo
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] =  np.einsum("ijabklmc->iajbcklm",dic["oovvooov"],optimize="optimal") #ovovvooo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabklmc->aijbklmc",dic["oovvooov"],optimize="optimal") #voovooov
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabklmc->aijbklcm",dic["oovvooov"],optimize="optimal") #voovoovo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijabklmc->aijbkclm",dic["oovvooov"],optimize="optimal") #voovovoo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("ijabklmc->aijbcklm",dic["oovvooov"],optimize="optimal") #voovvooo
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabklmc->iabjklmc",dic["oovvooov"],optimize="optimal") #ovvoooov
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabklmc->iabjklcm",dic["oovvooov"],optimize="optimal") #ovvooovo
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijabklmc->iabjkclm",dic["oovvooov"],optimize="optimal") #ovvoovoo
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("ijabklmc->iabjcklm",dic["oovvooov"],optimize="optimal") #ovvovooo
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("ijabklmc->aibjklmc",dic["oovvooov"],optimize="optimal") #vovoooov
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("ijabklmc->aibjklcm",dic["oovvooov"],optimize="optimal") #vovooovo
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("ijabklmc->aibjkclm",dic["oovvooov"],optimize="optimal") #vovoovoo
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] =  np.einsum("ijabklmc->aibjcklm",dic["oovvooov"],optimize="optimal") #vovovooo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabklmc->abijklmc",dic["oovvooov"],optimize="optimal") #vvooooov
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabklmc->abijklcm",dic["oovvooov"],optimize="optimal") #vvoooovo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijabklmc->abijkclm",dic["oovvooov"],optimize="optimal") #vvooovoo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("ijabklmc->abijcklm",dic["oovvooov"],optimize="optimal") #vvoovooo
			case "oovvoovv":
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  dic["oovvoovv"]
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("ijabklcd->ijabkcld",dic["oovvoovv"],optimize="optimal") #oovvovov
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabklcd->ijabckld",dic["oovvoovv"],optimize="optimal") #oovvvoov
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("ijabklcd->ijabkcdl",dic["oovvoovv"],optimize="optimal") #oovvovvo
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabklcd->ijabckdl",dic["oovvoovv"],optimize="optimal") #oovvvovo
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijabklcd->ijabcdkl",dic["oovvoovv"],optimize="optimal") #oovvvvoo
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijabklcd->iajbklcd",dic["oovvoovv"],optimize="optimal") #ovovoovv
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabklcd->iajbkcld",dic["oovvoovv"],optimize="optimal") #ovovovov
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("ijabklcd->iajbckld",dic["oovvoovv"],optimize="optimal") #ovovvoov
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabklcd->iajbkcdl",dic["oovvoovv"],optimize="optimal") #ovovovvo
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("ijabklcd->iajbckdl",dic["oovvoovv"],optimize="optimal") #ovovvovo
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("ijabklcd->iajbcdkl",dic["oovvoovv"],optimize="optimal") #ovovvvoo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijabklcd->aijbklcd",dic["oovvoovv"],optimize="optimal") #voovoovv
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("ijabklcd->aijbkcld",dic["oovvoovv"],optimize="optimal") #voovovov
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabklcd->aijbckld",dic["oovvoovv"],optimize="optimal") #voovvoov
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("ijabklcd->aijbkcdl",dic["oovvoovv"],optimize="optimal") #voovovvo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabklcd->aijbckdl",dic["oovvoovv"],optimize="optimal") #voovvovo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijabklcd->aijbcdkl",dic["oovvoovv"],optimize="optimal") #voovvvoo
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijabklcd->iabjklcd",dic["oovvoovv"],optimize="optimal") #ovvooovv
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("ijabklcd->iabjkcld",dic["oovvoovv"],optimize="optimal") #ovvoovov
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabklcd->iabjckld",dic["oovvoovv"],optimize="optimal") #ovvovoov
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("ijabklcd->iabjkcdl",dic["oovvoovv"],optimize="optimal") #ovvoovvo
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabklcd->iabjckdl",dic["oovvoovv"],optimize="optimal") #ovvovovo
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijabklcd->iabjcdkl",dic["oovvoovv"],optimize="optimal") #ovvovvoo
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijabklcd->aibjklcd",dic["oovvoovv"],optimize="optimal") #vovooovv
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabklcd->aibjkcld",dic["oovvoovv"],optimize="optimal") #vovoovov
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("ijabklcd->aibjckld",dic["oovvoovv"],optimize="optimal") #vovovoov
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabklcd->aibjkcdl",dic["oovvoovv"],optimize="optimal") #vovoovvo
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("ijabklcd->aibjckdl",dic["oovvoovv"],optimize="optimal") #vovovovo
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("ijabklcd->aibjcdkl",dic["oovvoovv"],optimize="optimal") #vovovvoo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijabklcd->abijklcd",dic["oovvoovv"],optimize="optimal") #vvoooovv
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("ijabklcd->abijkcld",dic["oovvoovv"],optimize="optimal") #vvooovov
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabklcd->abijckld",dic["oovvoovv"],optimize="optimal") #vvoovoov
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("ijabklcd->abijkcdl",dic["oovvoovv"],optimize="optimal") #vvooovvo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabklcd->abijckdl",dic["oovvoovv"],optimize="optimal") #vvoovovo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("ijabklcd->abijcdkl",dic["oovvoovv"],optimize="optimal") #vvoovvoo
			case "oovvovvv":
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["oovvovvv"]
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijabkcde->ijabckde",dic["oovvovvv"],optimize="optimal") #oovvvovv
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabkcde->ijabcdke",dic["oovvovvv"],optimize="optimal") #oovvvvov
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabkcde->ijabcdek",dic["oovvovvv"],optimize="optimal") #oovvvvvo
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijabkcde->iajbkcde",dic["oovvovvv"],optimize="optimal") #ovovovvv
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijabkcde->iajbckde",dic["oovvovvv"],optimize="optimal") #ovovvovv
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("ijabkcde->iajbcdke",dic["oovvovvv"],optimize="optimal") #ovovvvov
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("ijabkcde->iajbcdek",dic["oovvovvv"],optimize="optimal") #ovovvvvo
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijabkcde->aijbkcde",dic["oovvovvv"],optimize="optimal") #voovovvv
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijabkcde->aijbckde",dic["oovvovvv"],optimize="optimal") #voovvovv
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabkcde->aijbcdke",dic["oovvovvv"],optimize="optimal") #voovvvov
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabkcde->aijbcdek",dic["oovvovvv"],optimize="optimal") #voovvvvo
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijabkcde->iabjkcde",dic["oovvovvv"],optimize="optimal") #ovvoovvv
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijabkcde->iabjckde",dic["oovvovvv"],optimize="optimal") #ovvovovv
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabkcde->iabjcdke",dic["oovvovvv"],optimize="optimal") #ovvovvov
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabkcde->iabjcdek",dic["oovvovvv"],optimize="optimal") #ovvovvvo
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijabkcde->aibjkcde",dic["oovvovvv"],optimize="optimal") #vovoovvv
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijabkcde->aibjckde",dic["oovvovvv"],optimize="optimal") #vovovovv
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("ijabkcde->aibjcdke",dic["oovvovvv"],optimize="optimal") #vovovvov
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("ijabkcde->aibjcdek",dic["oovvovvv"],optimize="optimal") #vovovvvo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijabkcde->abijkcde",dic["oovvovvv"],optimize="optimal") #vvooovvv
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijabkcde->abijckde",dic["oovvovvv"],optimize="optimal") #vvoovovv
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("ijabkcde->abijcdke",dic["oovvovvv"],optimize="optimal") #vvoovvov
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("ijabkcde->abijcdek",dic["oovvovvv"],optimize="optimal") #vvoovvvo
			case "oovvvvvv":
				ten[0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["oovvvvvv"]
				ten[0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijabcedf->iajbcdef",dic["oovvvvvv"],optimize="optimal") #ovovvvvv
				ten[n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijabcedf->aijbcdef",dic["oovvvvvv"],optimize="optimal") #voovvvvv
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijabcedf->iabjcdef",dic["oovvvvvv"],optimize="optimal") #ovvovvvv
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("ijabcedf->aibjcdef",dic["oovvvvvv"],optimize="optimal") #vovovvvv
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("ijabcedf->abijcdef",dic["oovvvvvv"],optimize="optimal") #vvoovvvv
			case "ovvvoooo":
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ] =  dic["ovvvoooo"]
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("iabcjlkm->aibcjlkm",dic["ovvvoooo"],optimize="optimal") #vovvoooo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ] =  np.einsum("iabcjlkm->abicjlkm",dic["ovvvoooo"],optimize="optimal") #vvovoooo
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("iabcjlkm->abcijlkm",dic["ovvvoooo"],optimize="optimal") #vvvooooo
			case "ovvvooov":
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] =  dic["ovvvooov"] 
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("iabcjkld->iabcjkdl",dic["ovvvooov"],optimize="optimal") #ovvvoovo
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("iabcjkld->iabcjdkl",dic["ovvvooov"],optimize="optimal") #ovvvovoo
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("iabcjkld->iabcdjkl",dic["ovvvooov"],optimize="optimal") #ovvvvooo
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("iabcjkld->aibcjkld",dic["ovvvooov"],optimize="optimal") #vovvooov
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("iabcjkld->aibcjkdl",dic["ovvvooov"],optimize="optimal") #vovvoovo
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("iabcjkld->aibcjdkl",dic["ovvvooov"],optimize="optimal") #vovvovoo
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] =  np.einsum("iabcjkld->aibcdjkl",dic["ovvvooov"],optimize="optimal") #vovvvooo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("iabcjkld->abicjkld",dic["ovvvooov"],optimize="optimal") #vvovooov
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("iabcjkld->abicjkdl",dic["ovvvooov"],optimize="optimal") #vvovoovo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("iabcjkld->abicjdkl",dic["ovvvooov"],optimize="optimal") #vvovovoo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("iabcjkld->abicdjkl",dic["ovvvooov"],optimize="optimal") #vvovvooo
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("iabcjkld->abcijkld",dic["ovvvooov"],optimize="optimal") #vvvoooov
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("iabcjkld->abcijkdl",dic["ovvvooov"],optimize="optimal") #vvvooovo
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("iabcjkld->abcijdkl",dic["ovvvooov"],optimize="optimal") #vvvoovoo
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] =  np.einsum("iabcjkld->abcidjkl",dic["ovvvooov"],optimize="optimal") #vvvovooo
			case "ovvvoovv":
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  dic["ovvvoovv"]
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("iabcjkde->iabcjdke",dic["ovvvoovv"],optimize="optimal") #ovvvovov
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("iabcjkde->iabcdjke",dic["ovvvoovv"],optimize="optimal") #ovvvvoov
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("iabcjkde->iabcjdek",dic["ovvvoovv"],optimize="optimal") #ovvvovvo	
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("iabcjkde->iabcdjek",dic["ovvvoovv"],optimize="optimal") #ovvvvovo	
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("iabcjkde->iabcdejk",dic["ovvvoovv"],optimize="optimal") #ovvvvvoo	
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iabcjkde->aibcjkde",dic["ovvvoovv"],optimize="optimal") #vovvoovv
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("iabcjkde->aibcjdke",dic["ovvvoovv"],optimize="optimal") #vovvovov
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("iabcjkde->aibcdjke",dic["ovvvoovv"],optimize="optimal") #vovvvoov
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("iabcjkde->aibcjdek",dic["ovvvoovv"],optimize="optimal") #vovvovvo	
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("iabcjkde->aibcdjek",dic["ovvvoovv"],optimize="optimal") #vovvvovo	
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("iabcjkde->aibcdejk",dic["ovvvoovv"],optimize="optimal") #vovvvvoo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("iabcjkde->abicjkde",dic["ovvvoovv"],optimize="optimal") #vvovoovv
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("iabcjkde->abicjdke",dic["ovvvoovv"],optimize="optimal") #vvovovov
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("iabcjkde->abicdjke",dic["ovvvoovv"],optimize="optimal") #vvovvoov
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("iabcjkde->abicjdek",dic["ovvvoovv"],optimize="optimal") #vvovovvo	
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("iabcjkde->abicdjek",dic["ovvvoovv"],optimize="optimal") #vvovvovo	
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("iabcjkde->abicdejk",dic["ovvvoovv"],optimize="optimal") #vvovvvoo	
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iabcjkde->abcijkde",dic["ovvvoovv"],optimize="optimal") #vvvooovv
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("iabcjkde->abcijdke",dic["ovvvoovv"],optimize="optimal") #vvvoovov
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] = -np.einsum("iabcjkde->abcidjke",dic["ovvvoovv"],optimize="optimal") #vvvovoov
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("iabcjkde->abcijdek",dic["ovvvoovv"],optimize="optimal") #vvvoovvo	
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] =  np.einsum("iabcjkde->abcidjek",dic["ovvvoovv"],optimize="optimal") #vvvovovo	
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] = -np.einsum("iabcjkde->abcidejk",dic["ovvvoovv"],optimize="optimal") #vvvovvoo
			case "ovvvovvv":
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["ovvvovvv"]
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iabcjdef->iabcdjef",dic["ovvvovvv"],optimize="optimal") #ovvvvovv
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("iabcjdef->iabcdejf",dic["ovvvovvv"],optimize="optimal") #ovvvvvov			
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("iabcjdef->iabcdefj",dic["ovvvovvv"],optimize="optimal") #ovvvvvvo
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iabcjdef->aibcjdef",dic["ovvvovvv"],optimize="optimal") #vovvvovv
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("iabcjdef->aibcdjef",dic["ovvvovvv"],optimize="optimal") #vovvvovv
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("iabcjdef->aibcdejf",dic["ovvvovvv"],optimize="optimal") #vovvvvov			
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("iabcjdef->aibcdefj",dic["ovvvovvv"],optimize="optimal") #vovvvvvo
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("iabcjdef->abicjdef",dic["ovvvovvv"],optimize="optimal") #vvovvovv
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iabcjdef->abicdjef",dic["ovvvovvv"],optimize="optimal") #vvovvovv
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("iabcjdef->abicdejf",dic["ovvvovvv"],optimize="optimal") #vvovvvov			
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("iabcjdef->abicdefj",dic["ovvvovvv"],optimize="optimal") #vvovvvvo
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iabcjdef->abcijdef",dic["ovvvovvv"],optimize="optimal") #vvvovovv
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("iabcjdef->abcidjef",dic["ovvvovvv"],optimize="optimal") #vvvovovv
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("iabcjdef->abcidejf",dic["ovvvovvv"],optimize="optimal") #vvvovvov			
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("iabcjdef->abcidefj",dic["ovvvovvv"],optimize="optimal") #vvvovvvo
			case "ovvvvvvv":
				ten[0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["ovvvvvvv"]	
				ten[n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iabcdefg->aibcdefg",dic["ovvvvvvv"],optimize="optimal") #vovvvvvv				
				ten[n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  np.einsum("iabcdefg->aibcdefg",dic["ovvvvvvv"],optimize="optimal") #vovvvvvv				
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("iabcdefg->aibcdefg",dic["ovvvvvvv"],optimize="optimal") #vovvvvvv	
			case "vvvvoooo":
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,0:n_occ] =  dic["vvvvoooo"]
			case "vvvvooov":			
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ,n_occ:2*n_act] =  dic["vvvvooov"]
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("abcdijke->abcdijek",dic["vvvvooov"],optimize="optimal") #vvvvoovo
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("abcdijke->abcdiejk",dic["vvvvooov"],optimize="optimal") #vvvvovoo
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,0:n_occ] = -np.einsum("abcdijke->abcdeijk",dic["vvvvooov"],optimize="optimal") #vvvvvooo
			case "vvvvoovv":
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] =  dic["vvvvoovv"]
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] = -np.einsum("abcdijef->abcdiejf",dic["vvvvoovv"],optimize="optimal") #vvvvovov
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ,n_occ:2*n_act] =  np.einsum("abcdijef->abcdeijf",dic["vvvvoovv"],optimize="optimal") #vvvvvoov
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] =  np.einsum("abcdijef->abcdiefj",dic["vvvvoovv"],optimize="optimal") #vvvvovov
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,0:n_occ] = -np.einsum("abcdijef->abcdeifj",dic["vvvvoovv"],optimize="optimal") #vvvvvovo
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,0:n_occ] =  np.einsum("abcdijef->abcdefij",dic["vvvvoovv"],optimize="optimal") #vvvvvvoo
			case "vvvvovvv":
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["vvvvovvv"]
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act,n_occ:2*n_act] = -np.einsum("abcdiefg->abcdeifg",dic["vvvvovvv"],optimize="optimal") #vvvvvovv
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ,n_occ:2*n_act] =  np.einsum("abcdiefg->abcdefig",dic["vvvvovvv"],optimize="optimal") #vvvvvvov
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,0:n_occ] = -np.einsum("abcdiefg->abcdefgi",dic["vvvvovvv"],optimize="optimal") #vvvvvvvo
			case "vvvvvvvv":
				ten[n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act,n_occ:2*n_act] =  dic["vvvvvvvv"]
	return ten 

def t1_mat2dic(t1_amps,n_act):
	n_a = t1_amps[0].shape[0]
	n_orb = t1_amps[0].shape[1] + t1_amps[0].shape[0]
	n_virt_int_a = n_act - n_a 
	n_virt_ext_a = n_orb - n_act  
	n_occ = 2*n_a 
	n_virt_ext = 2*n_virt_ext_a
	t1 = {
		"oV": np.zeros((n_occ,n_virt_ext)),
		"Vo": np.zeros((n_virt_ext,n_occ))
	}
	for i in range(0,n_a):
		ia = 2*i 
		ib = 2*i+1 
		for A in range(0,n_virt_ext_a):
			Aa = 2*A 
			Ab = 2*A+1 
			t1["oV"][ia,Aa] = t1_amps[0][i,A+n_virt_int_a]
			t1["Vo"][Aa,ia] = t1_amps[0][i,A+n_virt_int_a]
			t1["oV"][ib,Ab] = t1_amps[1][i,A+n_virt_int_a]
			t1["Vo"][Ab,ib] = t1_amps[1][i,A+n_virt_int_a] 
	return t1  

def t2_ten2dic(t2_amps,n_act):
	n_a = t2_amps[0].shape[0]
	n_orb = t2_amps[0].shape[2] + t2_amps[0].shape[0] 
	n_virt_int_a = n_act - n_a 
	n_virt_ext_a = n_orb - n_act  
	n_occ = 2*n_a 
	n_virt_int = 2*n_virt_int_a 
	n_virt_ext = 2*n_virt_ext_a		

	t2 = {
		"oovV": np.zeros((n_occ,n_occ,n_virt_int,n_virt_ext)),
		"vVoo": np.zeros((n_virt_int,n_virt_ext,n_occ,n_occ)),
		"ooVV": np.zeros((n_occ,n_occ,n_virt_ext,n_virt_ext)),
		"VVoo": np.zeros((n_virt_ext,n_virt_ext,n_occ,n_occ))
	}

	# t_{ia,ja}^{aa,Ba}/t_{ib,jb}^{ab,Bb}
	for i in range(0,n_a):
		ia = 2*i
		ib = 2*i+1 
		for j in range(i+1,n_a):
			ja = 2*j
			jb = 2*j+1  
			for a in range(0,n_virt_int_a):
				aa = 2*a 
				ab = 2*a+1
				for B in range(0,n_virt_ext_a):
					Ba = 2*B
					Bb = 2*B+1  
					tijaB = t2_amps[0][i,j,a,B+n_virt_int_a]
					t2["oovV"][ia,ja,aa,Ba] =  tijaB
					t2["oovV"][ja,ia,aa,Ba] = -tijaB
					t2["vVoo"][aa,Ba,ja,ia] = -tijaB
					t2["vVoo"][aa,Ba,ia,ja] =  tijaB
					tijaB = t2_amps[2][i,j,a,B+n_virt_int_a]
					t2["oovV"][ib,jb,ab,Bb] =  tijaB
					t2["oovV"][jb,ib,ab,Bb] = -tijaB
					t2["vVoo"][ab,Bb,jb,ib] = -tijaB
					t2["vVoo"][ab,Bb,ib,jb] =  tijaB
	# t_{ia,ja}^{Aa,Ba}/t_{ib,jb}^{Ab,Bb}
	for i in range(0,n_a):
		ia = 2*i 
		ib = 2*i+1 
		for j in range(i+1,n_a):
			ja = 2*j
			jb = 2*j+1  
			for A in range(0,n_virt_ext_a):
				Aa = 2*A
				Ab = 2*A+1  
				for B in range(A+1,n_virt_ext_a):
					Ba = 2*B
					Bb = 2*B+1  
					tijAB = t2_amps[0][i,j,A+n_virt_int_a,B+n_virt_int_a]
					t2["ooVV"][ia,ja,Aa,Ba] =  tijAB
					t2["ooVV"][ja,ia,Aa,Ba] = -tijAB
					t2["ooVV"][ia,ja,Ba,Aa] = -tijAB
					t2["ooVV"][ja,ia,Ba,Aa] =  tijAB
					t2["VVoo"][Ba,Aa,ja,ia] =  tijAB
					t2["VVoo"][Aa,Ba,ja,ia] = -tijAB
					t2["VVoo"][Ba,Aa,ia,ja] = -tijAB
					t2["VVoo"][Aa,Ba,ia,ja] =  tijAB
					tijAB = t2_amps[2][i,j,A+n_virt_int_a,B+n_virt_int_a]
					t2["ooVV"][ib,jb,Ab,Bb] =  tijAB
					t2["ooVV"][jb,ib,Ab,Bb] = -tijAB
					t2["ooVV"][ib,jb,Bb,Ab] = -tijAB
					t2["ooVV"][jb,ib,Bb,Ab] =  tijAB
					t2["VVoo"][Bb,Ab,jb,ib] =  tijAB
					t2["VVoo"][Ab,Bb,jb,ib] = -tijAB
					t2["VVoo"][Bb,Ab,ib,jb] = -tijAB
					t2["VVoo"][Ab,Bb,ib,jb] =  tijAB
	# t_{ia,jb}^{aa,Bb}
	for i in range(0,n_a):
		ia = 2*i 
		for j in range(0,n_a):
			jb = 2*j+1 
			for a in range(0,n_virt_int_a):
				aa = 2*a 
				ab = 2*a+1 
				for B in range(0,n_virt_ext_a):
					Ba = 2*B 
					Bb = 2*B+1 
					tijaB = t2_amps[1][i,j,a,B+n_virt_int_a]
					t2["oovV"][ia,jb,aa,Bb] =  tijaB
					t2["oovV"][jb,ia,aa,Bb] = -tijaB
					t2["vVoo"][aa,Bb,ia,jb] =  tijaB
					t2["vVoo"][aa,Bb,jb,ia] = -tijaB
					tijBa = t2_amps[1][i,j,B+n_virt_int_a,a]
					t2["oovV"][ia,jb,ab,Ba] = -tijBa
					t2["oovV"][jb,ia,ab,Ba] =  tijBa
					t2["vVoo"][ab,Ba,ia,jb] = -tijBa
					t2["vVoo"][ab,Ba,jb,ia] =  tijBa
	# t_{ia,jb}^{Aa,Bb}
	for i in range(0,n_a):
		ia = 2*i 
		for j in range(0,n_a):
			jb = 2*j+1 
			for A in range(0,n_virt_ext_a):
				Aa = 2*A 
				for B in range(0,n_virt_ext_a):
					Bb = 2*B+1 
					tijAB = t2_amps[1][i,j,A+n_virt_int_a,B+n_virt_int_a]
					t2["ooVV"][ia,jb,Aa,Bb] =  tijAB
					t2["ooVV"][jb,ia,Aa,Bb] = -tijAB
					t2["ooVV"][ia,jb,Bb,Aa] = -tijAB
					t2["ooVV"][jb,ia,Bb,Aa] =  tijAB
					t2["VVoo"][Aa,Bb,ia,jb] =  tijAB
					t2["VVoo"][Aa,Bb,jb,ia] = -tijAB
					t2["VVoo"][Bb,Aa,ia,jb] = -tijAB
					t2["VVoo"][Bb,Aa,jb,ia] =  tijAB
	return t2  

def get_many_body_terms(operator):
	constant = of.FermionOperator()
	one_body = of.FermionOperator()
	two_body = of.FermionOperator()
	three_body = of.FermionOperator()
	four_body = of.FermionOperator()
	terms = operator.terms 
	for term in terms:
		if(len(term) == 0):
			constant += of.FermionOperator(term,terms.get(term))
		elif(len(term) == 2):
			one_body += of.FermionOperator(term,terms.get(term))
		elif(len(term) == 4):
			two_body += of.FermionOperator(term,terms.get(term))
		elif(len(term) == 6):
			three_body += of.FermionOperator(term,terms.get(term))
		elif(len(term) == 8):
			four_body += of.FermionOperator(term,terms.get(term))
		else:
			print("Unexpected number of terms: %d"%len(term))
	return(constant,one_body,two_body,three_body,four_body)

def as_proj(operator,act_max):
	proj_op = of.FermionOperator()
	const, one_body, two_body, three_body, four_body = get_many_body_terms(operator)
	# constant terms
	proj_op += const 
	# one-body terms
	terms1 = one_body.terms
	for term in terms1:
		if(term[0][0] < act_max):
			if(term[1][0] < act_max):
				proj_op += of.FermionOperator(term,terms1.get(term))
	# two-body terms
	terms2 = two_body.terms
	for term in terms2:
		if(term[0][0] < act_max):
			if(term[1][0] < act_max):
				if(term[2][0] < act_max):
					if (term[3][0] < act_max):
						proj_op += of.FermionOperator(term,terms2.get(term))
	# three-body terms
	terms3 = three_body.terms
	for term in terms3:
		if(term[0][0] < act_max):
			if(term[1][0] < act_max):
				if(term[2][0] < act_max):
					if (term[3][0] < act_max):
						if(term[4][0] < act_max):
							if(term[5][0] < act_max):
								proj_op += of.FermionOperator(term,terms3.get(term))
	# four-body terms
	terms4 = four_body.terms  
	for term in terms4:
		if(term[0][0] < act_max):
			if(term[1][0] < act_max):
				if(term[2][0] < act_max):
					if (term[3][0] < act_max):
						if(term[4][0] < act_max):
							if(term[5][0] < act_max):
								if(term[6][0] < act_max):
									if(term[7][0] < act_max):
										proj_op += of.FermionOperator(term,terms4.get(term))
	return proj_op

def t1_to_op(t1_amps):
	n_a = t1_amps[0].shape[0]
	n_virt_a = t1_amps[0].shape[1]
	s1_op = of.FermionOperator()
	for i in range(0,n_a):
		ia = 2*i 
		ib = 2*i+1 
		for a in range(0,n_virt_a):
			aa = 2*a + 2*n_a
			ab = 2*a+1 + 2*n_a
			s1_op += of.FermionOperator(((aa,1),(ia,0)),  t1_amps[0][i,a])
			s1_op += of.FermionOperator(((ia,1),(aa,0)), -t1_amps[0][i,a])
			s1_op += of.FermionOperator(((ab,1),(ib,0)),  t1_amps[1][i,a])
			s1_op += of.FermionOperator(((ib,1),(ab,0)), -t1_amps[1][i,a])
	return s1_op

def t2_to_op(t2_amps):
	n_a = t2_amps[0].shape[0]
	n_virt_a = t2_amps[0].shape[2]
	s2_op = of.FermionOperator()
	# aaaa/bbbb
	for i in range(0,n_a):
		ia = 2*i 
		ib = 2*i+1 
		for j in range(i+1,n_a):
			ja = 2*j 
			jb = 2*j+1 
			for a in range(0,n_virt_a):
				aa = 2*a + 2*n_a
				ab = 2*a+1 + 2*n_a
				for b in range(a+1,n_virt_a):
					ba = 2*b + 2*n_a
					bb = 2*b+1 + 2*n_a 
					s2_op += of.FermionOperator(((aa,1),(ba,1),(ja,0),(ia,0)),  t2_amps[0][i,j,a,b])
					s2_op += of.FermionOperator(((ia,1),(ja,1),(ba,0),(aa,0)), -t2_amps[0][i,j,a,b])
					s2_op += of.FermionOperator(((ab,1),(bb,1),(jb,0),(ib,0)),  t2_amps[2][i,j,a,b])
					s2_op += of.FermionOperator(((ib,1),(jb,1),(bb,0),(ab,0)), -t2_amps[2][i,j,a,b])
	# abab
	for i in range(0,n_a):
		ia = 2*i 
		for j in range(0,n_a):
			jb = 2*j+1 
			for a in range(0,n_virt_a):
				aa = 2*a + 2*n_a 
				for b in range(0,n_virt_a):
					bb = 2*b+1 + 2*n_a 
					s2_op += of.FermionOperator(((aa,1),(bb,1),(jb,0),(ia,0)),  t2_amps[1][i,j,a,b])
					s2_op += of.FermionOperator(((ia,1),(jb,1),(bb,0),(aa,0)), -t2_amps[1][i,j,a,b])
	return s2_op 

def t1_to_ext(t1,n_act):
	n_a = t1[0].shape[0]
	n_virt_a = t1[0].shape[1]
	n_virt_int_a = n_act - n_a 
	for i in range(0,n_a):
		for a in range(0,n_virt_int_a):
			t1[0][i,a] = 0
			t1[1][i,a] = 0
	return t1 

def t2_to_ext(t2,n_act):
	n_a = t2[0].shape[0]
	n_virt_a = t2[0].shape[2]
	n_virt_int_a = n_act-n_a 
	for i in range(0,n_a):
		for j in range(0,n_a):
			for a in range(0,n_virt_int_a):
				for b in range(0,n_virt_int_a):
					t2[0][i,j,a,b] = 0
					t2[1][i,j,a,b] = 0
					t2[2][i,j,a,b] = 0
	return t2 



