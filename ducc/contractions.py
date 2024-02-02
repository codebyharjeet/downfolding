import ducc
import scipy
#import vqe_methods
#import pyscf_helper

import pyscf
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo, molden, cc
from pyscf.cc import ccsd

import openfermion as of
from openfermion import *
#from tVQE import *

import numpy as np


def compute_ft1(f, t, n_a, n_b, n_orb):
    # FT1
    # t_ov = t1 and t_vo = transpose(t1)
    # Initializing the ft1_mat_1 one body
    ft1_mat = np.zeros((2 * n_orb, 2 * n_orb))

    # f_oo and t1
    ft1oo = 0
    ft1oo += 1.000000000 * np.einsum("ai,ja->ji", f["vo"], t["ov"], optimize="optimal")
    ft1oo += 1.000000000 * np.einsum("ia,aj->ij", f["ov"], t["vo"], optimize="optimal")
    # print("ft1oo = ",ft1oo.shape)

    # f_ov and t1
    ft1ov = 0
    ft1ov += -1.000000000 * np.einsum("ij,ja->ia", f["oo"], t["ov"], optimize="optimal")
    ft1ov += 1.000000000 * np.einsum("ba,ib->ia", f["vv"], t["ov"], optimize="optimal")
    # print("ft1ov = ",ft1ov.shape)

    # f_vo and t1
    ft1vo = 0
    ft1vo += -1.000000000 * np.einsum("ji,aj->ai", f["oo"], t["vo"], optimize="optimal")
    ft1vo += 1.000000000 * np.einsum("ab,bi->ai", f["vv"], t["vo"], optimize="optimal")
    # print("ft1vo = ",ft1vo.shape)

    # f_vv and t1
    ft1vv = 0
    ft1vv += -1.000000000 * np.einsum("ai,ib->ab", f["vo"], t["ov"], optimize="optimal")
    ft1vv += -1.000000000 * np.einsum("ia,bi->ba", f["ov"], t["vo"], optimize="optimal")
    # print("ft1vv = ",ft1vv.shape)

    # f and t1
    ft1 = 0
    ft1 += 1.000000000 * np.einsum("ai,ia->", f["vo"], t["ov"], optimize="optimal")
    ft1 += 1.000000000 * np.einsum("ia,ai->", f["ov"], t["vo"], optimize="optimal")
    # print("ft1 = ",ft1)

    ft1_mat = ducc.make_full_one(ft1_mat, ft1oo, n_a, n_b, n_orb)
    ft1_mat = ducc.make_full_one(ft1_mat, ft1ov, n_a, n_b, n_orb)
    ft1_mat = ducc.make_full_one(ft1_mat, ft1vo, n_a, n_b, n_orb)
    ft1_mat = ducc.make_full_one(ft1_mat, ft1vv, n_a, n_b, n_orb)

    return ft1_mat, ft1


def compute_ft2(f, t, n_a, n_b, n_orb):
    # FT2
    # t_oovv = t2 and t_vvoo = transpose(t2)

    # Initializing the ft2_mat_1 one body
    ft2_mat_1 = np.zeros((2 * n_orb, 2 * n_orb))
    # print(ft2_mat_1.shape)

    # Initializing the ft2_mat_2 two body
    ft2_mat_2 = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    # print(ft2_mat_2.shape)

    ft2ooov = 0
    ft2ooov += -0.500000000 * np.einsum(
        "bi,jkab->jkia", f["vo"], t["oovv"], optimize="optimal"
    )
    # print(ft2ooov.shape)
    ft2_mat_2 = ducc.make_full_two(ft2_mat_2, ft2ooov, n_a, n_b, n_orb)

    ft2oovv = 0
    ft2oovv += 0.500000000 * np.einsum(
        "ik,jkab->ijab", f["oo"], t["oovv"], optimize="optimal"
    )
    ft2oovv += -0.500000000 * np.einsum(
        "ca,ijbc->ijab", f["vv"], t["oovv"], optimize="optimal"
    )
    # print(ft2oovv.shape)
    ft2_mat_2 = ducc.make_full_two(ft2_mat_2, ft2oovv, n_a, n_b, n_orb)

    ft2ovoo = 0
    ft2ovoo += -0.500000000 * np.einsum(
        "ib,abjk->iajk", f["ov"], t["vvoo"], optimize="optimal"
    )
    # print(ft2ovoo.shape)
    ft2_mat_2 = ducc.make_full_two(ft2_mat_2, ft2ovoo, n_a, n_b, n_orb)

    ft2ovvv = 0
    ft2ovvv += -0.500000000 * np.einsum(
        "aj,ijbc->iabc", f["vo"], t["oovv"], optimize="optimal"
    )
    # print(ft2ovvv.shape)
    ft2_mat_2 = ducc.make_full_two(ft2_mat_2, ft2ovvv, n_a, n_b, n_orb)

    ft2vvoo = 0
    ft2vvoo += 0.500000000 * np.einsum(
        "ki,abjk->abij", f["oo"], t["vvoo"], optimize="optimal"
    )
    ft2vvoo += -0.500000000 * np.einsum(
        "ac,bcij->abij", f["vv"], t["vvoo"], optimize="optimal"
    )
    # print(ft2vvoo.shape)
    ft2_mat_2 = ducc.make_full_two(ft2_mat_2, ft2vvoo, n_a, n_b, n_orb)

    ft2vvov = 0
    ft2vvov += -0.500000000 * np.einsum(
        "ja,bcij->bcia", f["ov"], t["vvoo"], optimize="optimal"
    )
    # print(ft2vvov.shape)
    ft2_mat_2 = ducc.make_full_two(ft2_mat_2, ft2vvov, n_a, n_b, n_orb)

    ft2ov = 0
    ft2ov += 1.000000000 * np.einsum(
        "bj,ijab->ia", f["vo"], t["oovv"], optimize="optimal"
    )
    # print(ft2ov.shape)
    ft2_mat_1 = ducc.make_full_one(ft2_mat_1, ft2ov, n_a, n_b, n_orb)

    ft2vo = 0
    ft2vo += 1.000000000 * np.einsum(
        "jb,abij->ai", f["ov"], t["vvoo"], optimize="optimal"
    )
    # print(ft2vo.shape)
    ft2_mat_1 = ducc.make_full_one(ft2_mat_1, ft2vo, n_a, n_b, n_orb)

    return ft2_mat_1, ft2_mat_2


def compute_wt1(v, t, n_a, n_b, n_orb):
    # WT1
    # t_ov = t1 and t_vo = transpose(t1)

    # Initializing the wt1_mat_1 one body
    wt1_mat_1 = np.zeros((2 * n_orb, 2 * n_orb))
    # print("wt1_mat_1 = ",wt1_mat_1.shape)

    # Initializing the wt1_mat_2 two body
    wt1_mat_2 = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    # print("wt1_mat_2 = ",wt1_mat_2.shape)

    wt1oooo = 0
    wt1oooo += 0.500000000 * np.einsum(
        "kaij,la->klij", v["ovoo"], t["ov"], optimize="optimal"
    )
    wt1oooo += 0.500000000 * np.einsum(
        "jkia,al->jkil", v["ooov"], t["vo"], optimize="optimal"
    )
    # print("wt1oooo = ",wt1oooo.shape)
    wt1_mat_2 = ducc.make_full_two(wt1_mat_2, wt1oooo, n_a, n_b, n_orb)

    wt1ooov = 0
    wt1ooov += -0.500000000 * np.einsum(
        "jkil,la->jkia", v["oooo"], t["ov"], optimize="optimal"
    )
    wt1ooov += 1.000000000 * np.einsum(
        "jbia,kb->jkia", v["ovov"], t["ov"], optimize="optimal"
    )
    wt1ooov += -0.500000000 * np.einsum(
        "ijab,bk->ijka", v["oovv"], t["vo"], optimize="optimal"
    )
    # print("wt1ooov = ",wt1ooov.shape)
    wt1_mat_2 = ducc.make_full_two(wt1_mat_2, wt1ooov, n_a, n_b, n_orb)

    wt1oovv = 0
    wt1oovv += 0.500000000 * np.einsum(
        "ijka,kb->ijab", v["ooov"], t["ov"], optimize="optimal"
    )
    wt1oovv += 0.500000000 * np.einsum(
        "icab,jc->ijab", v["ovvv"], t["ov"], optimize="optimal"
    )
    # print("wt1oovv = ",wt1oovv.shape)
    wt1_mat_2 = ducc.make_full_two(wt1_mat_2, wt1oovv, n_a, n_b, n_orb)

    wt1ovoo = 0
    wt1ovoo += -0.500000000 * np.einsum(
        "klij,al->kaij", v["oooo"], t["vo"], optimize="optimal"
    )
    wt1ovoo += -0.500000000 * np.einsum(
        "abij,kb->kaij", v["vvoo"], t["ov"], optimize="optimal"
    )
    wt1ovoo += 1.000000000 * np.einsum(
        "jaib,bk->jaik", v["ovov"], t["vo"], optimize="optimal"
    )
    # print("wt1ovoo = ",wt1ovoo.shape)
    wt1_mat_2 = ducc.make_full_two(wt1_mat_2, wt1ovoo, n_a, n_b, n_orb)

    wt1ovov = 0
    wt1ovov += -1.000000000 * np.einsum(
        "jaik,kb->jaib", v["ovoo"], t["ov"], optimize="optimal"
    )
    wt1ovov += -1.000000000 * np.einsum(
        "jkia,bk->jbia", v["ooov"], t["vo"], optimize="optimal"
    )
    wt1ovov += -1.000000000 * np.einsum(
        "bcia,jc->jbia", v["vvov"], t["ov"], optimize="optimal"
    )
    wt1ovov += -1.000000000 * np.einsum(
        "ibac,cj->ibja", v["ovvv"], t["vo"], optimize="optimal"
    )
    # print("wt1ovov = ",wt1ovov.shape)
    wt1_mat_2 = ducc.make_full_two(wt1_mat_2, wt1ovov, n_a, n_b, n_orb)

    wt1ovvv = 0
    wt1ovvv += 1.000000000 * np.einsum(
        "ibja,jc->ibac", v["ovov"], t["ov"], optimize="optimal"
    )
    wt1ovvv += -0.500000000 * np.einsum(
        "ijab,cj->icab", v["oovv"], t["vo"], optimize="optimal"
    )
    wt1ovvv += -0.500000000 * np.einsum(
        "cdab,id->icab", v["vvvv"], t["ov"], optimize="optimal"
    )
    # print("wt1ovvv = ",wt1ovvv.shape)
    wt1_mat_2 = ducc.make_full_two(wt1_mat_2, wt1ovvv, n_a, n_b, n_orb)

    wt1vvoo = 0
    wt1vvoo += 0.500000000 * np.einsum(
        "kaij,bk->abij", v["ovoo"], t["vo"], optimize="optimal"
    )
    wt1vvoo += 0.500000000 * np.einsum(
        "abic,cj->abij", v["vvov"], t["vo"], optimize="optimal"
    )
    # print("wt1vvoo = ",wt1vvoo.shape)
    wt1_mat_2 = ducc.make_full_two(wt1_mat_2, wt1vvoo, n_a, n_b, n_orb)

    wt1vvov = 0
    wt1vvov += -0.500000000 * np.einsum(
        "abij,jc->abic", v["vvoo"], t["ov"], optimize="optimal"
    )
    wt1vvov += 1.000000000 * np.einsum(
        "jbia,cj->bcia", v["ovov"], t["vo"], optimize="optimal"
    )
    wt1vvov += -0.500000000 * np.einsum(
        "bcad,di->bcia", v["vvvv"], t["vo"], optimize="optimal"
    )
    # print("wt1vvov = ",wt1vvov.shape)
    wt1_mat_2 = ducc.make_full_two(wt1_mat_2, wt1vvov, n_a, n_b, n_orb)

    wt1vvvv = 0
    wt1vvvv += 0.500000000 * np.einsum(
        "bcia,id->bcad", v["vvov"], t["ov"], optimize="optimal"
    )
    wt1vvvv += 0.500000000 * np.einsum(
        "icab,di->cdab", v["ovvv"], t["vo"], optimize="optimal"
    )
    # print("wt1vvvv = ",wt1vvvv.shape)
    wt1_mat_2 = ducc.make_full_two(wt1_mat_2, wt1vvvv, n_a, n_b, n_orb)

    wt1oo = 0
    wt1oo += 1.000000000 * np.einsum(
        "jaik,ka->ji", v["ovoo"], t["ov"], optimize="optimal"
    )
    wt1oo += 1.000000000 * np.einsum(
        "jkia,ak->ji", v["ooov"], t["vo"], optimize="optimal"
    )
    # print("wt1oo = ",wt1oo.shape)
    wt1_mat_1 = ducc.make_full_one(wt1_mat_1, wt1oo, n_a, n_b, n_orb)

    wt1ov = 0
    wt1ov += -1.000000000 * np.einsum(
        "ibja,jb->ia", v["ovov"], t["ov"], optimize="optimal"
    )
    wt1ov += 1.000000000 * np.einsum(
        "ijab,bj->ia", v["oovv"], t["vo"], optimize="optimal"
    )
    # print("wt1ov = ",wt1ov.shape)
    wt1_mat_1 = ducc.make_full_one(wt1_mat_1, wt1ov, n_a, n_b, n_orb)

    wt1vo = 0
    wt1vo += 1.000000000 * np.einsum(
        "abij,jb->ai", v["vvoo"], t["ov"], optimize="optimal"
    )
    wt1vo += -1.000000000 * np.einsum(
        "jaib,bj->ai", v["ovov"], t["vo"], optimize="optimal"
    )
    # print("wt1vo = ",wt1vo.shape)
    wt1_mat_1 = ducc.make_full_one(wt1_mat_1, wt1vo, n_a, n_b, n_orb)

    wt1vv = 0
    wt1vv += -1.000000000 * np.einsum(
        "bcia,ic->ba", v["vvov"], t["ov"], optimize="optimal"
    )
    wt1vv += -1.000000000 * np.einsum(
        "ibac,ci->ba", v["ovvv"], t["vo"], optimize="optimal"
    )
    # print("wt1vv = ",wt1vv.shape)
    wt1_mat_1 = ducc.make_full_one(wt1_mat_1, wt1vv, n_a, n_b, n_orb)

    return wt1_mat_1, wt1_mat_2


def compute_wt2(v, t, n_a, n_b, n_orb, compute_three_body=False):
    # WT2
    # t_oovv = t2 and t_vvoo = transpose(t2)

    # Initializing the wt2_mat_1 one body
    wt2_mat_1 = np.zeros((2 * n_orb, 2 * n_orb))
    # print("wt2_mat_1 = ",wt2_mat_1.shape)

    # Initializing the wt2_mat_2 two body
    wt2_mat_2 = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    # print("wt2_mat_2 = ",wt2_mat_2.shape)

    # Initializing the wt2_mat_3 three body
    wt2_mat_3 = np.zeros(
        (2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb)
    )
    # print("wt2_mat_3 = ",wt2_mat_3.shape)

    if compute_three_body == True:
        wt2ooooov = 0
        wt2ooooov += -0.250000000 * np.einsum(
            "kbij,lmab->klmija", v["ovoo"], t["oovv"], optimize="optimal"
        )
        # print("wt2ooooov = ",wt2ooooov.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2ooooov, n_a, n_b, n_orb)

        wt2oooovv = 0
        wt2oooovv += 0.250000000 * np.einsum(
            "jkim,lmab->jkliab", v["oooo"], t["oovv"], optimize="optimal"
        )
        wt2oooovv += -0.500000000 * np.einsum(
            "jcia,klbc->jkliab", v["ovov"], t["oovv"], optimize="optimal"
        )
        # print("wt2oooovv = ",wt2oooovv.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2oooovv, n_a, n_b, n_orb)

        wt2ooovvv = 0
        wt2ooovvv += -0.250000000 * np.einsum(
            "ijla,klbc->ijkabc", v["ooov"], t["oovv"], optimize="optimal"
        )
        wt2ooovvv += -0.250000000 * np.einsum(
            "idab,jkcd->ijkabc", v["ovvv"], t["oovv"], optimize="optimal"
        )
        # print("wt2ooovvv = ",wt2ooovvv.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2ooovvv, n_a, n_b, n_orb)

        wt2oovooo = 0
        wt2oovooo += -0.250000000 * np.einsum(
            "jkib,ablm->jkailm", v["ooov"], t["vvoo"], optimize="optimal"
        )
        # print("wt2oovooo = ",wt2oovooo.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2oovooo, n_a, n_b, n_orb)

        wt2oovoov = 0
        wt2oovoov += -0.250000000 * np.einsum(
            "acij,klbc->klaijb", v["vvoo"], t["oovv"], optimize="optimal"
        )
        wt2oovoov += -0.250000000 * np.einsum(
            "ijac,bckl->ijbkla", v["oovv"], t["vvoo"], optimize="optimal"
        )
        # print("wt2oovoov = ",wt2oovoov.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2oovoov, n_a, n_b, n_orb)

        wt2oovovv = 0
        wt2oovovv += -0.500000000 * np.einsum(
            "jail,klbc->jkaibc", v["ovoo"], t["oovv"], optimize="optimal"
        )
        wt2oovovv += -0.500000000 * np.einsum(
            "bdia,jkcd->jkbiac", v["vvov"], t["oovv"], optimize="optimal"
        )
        # print("wt2oovovv = ",wt2oovovv.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2oovovv, n_a, n_b, n_orb)

        wt2oovvvv = 0
        wt2oovvvv += 0.500000000 * np.einsum(
            "ibka,jkcd->ijbacd", v["ovov"], t["oovv"], optimize="optimal"
        )
        wt2oovvvv += -0.250000000 * np.einsum(
            "ceab,ijde->ijcabd", v["vvvv"], t["oovv"], optimize="optimal"
        )
        # print("wt2oovvvv = ",wt2oovvvv.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2oovvvv, n_a, n_b, n_orb)

        wt2ovvooo = 0
        wt2ovvooo += 0.250000000 * np.einsum(
            "kmij,ablm->kabijl", v["oooo"], t["vvoo"], optimize="optimal"
        )
        wt2ovvooo += -0.500000000 * np.einsum(
            "jaic,bckl->jabikl", v["ovov"], t["vvoo"], optimize="optimal"
        )
        # print("wt2ovvooo = ",wt2ovvooo.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2ovvooo, n_a, n_b, n_orb)

        wt2ovvoov = 0
        wt2ovvoov += -0.500000000 * np.einsum(
            "jlia,bckl->jbcika", v["ooov"], t["vvoo"], optimize="optimal"
        )
        wt2ovvoov += -0.500000000 * np.einsum(
            "ibad,cdjk->ibcjka", v["ovvv"], t["vvoo"], optimize="optimal"
        )
        # print("wt2ovvoov = ",wt2ovvoov.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2ovvoov, n_a, n_b, n_orb)

        wt2ovvovv = 0
        wt2ovvovv += 0.250000000 * np.einsum(
            "abik,jkcd->jabicd", v["vvoo"], t["oovv"], optimize="optimal"
        )
        wt2ovvovv += 0.250000000 * np.einsum(
            "ikab,cdjk->icdjab", v["oovv"], t["vvoo"], optimize="optimal"
        )
        # print("wt2ovvovv = ",wt2ovvovv.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2ovvovv, n_a, n_b, n_orb)

        wt2ovvvvv = 0
        wt2ovvvvv += -0.250000000 * np.einsum(
            "bcja,ijde->ibcade", v["vvov"], t["oovv"], optimize="optimal"
        )
        # print("wt2ovvvvv = ",wt2ovvvvv.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2ovvvvv, n_a, n_b, n_orb)

        wt2vvvooo = 0
        wt2vvvooo += -0.250000000 * np.einsum(
            "laij,bckl->abcijk", v["ovoo"], t["vvoo"], optimize="optimal"
        )
        wt2vvvooo += -0.250000000 * np.einsum(
            "abid,cdjk->abcijk", v["vvov"], t["vvoo"], optimize="optimal"
        )
        # print("wt2vvvooo = ",wt2vvvooo.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2vvvooo, n_a, n_b, n_orb)

        wt2vvvoov = 0
        wt2vvvoov += 0.500000000 * np.einsum(
            "kbia,cdjk->bcdija", v["ovov"], t["vvoo"], optimize="optimal"
        )
        wt2vvvoov += -0.250000000 * np.einsum(
            "bcae,deij->bcdija", v["vvvv"], t["vvoo"], optimize="optimal"
        )
        # print("wt2vvvoov = ",wt2vvvoov.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2vvvoov, n_a, n_b, n_orb)

        wt2vvvovv = 0
        wt2vvvovv += -0.250000000 * np.einsum(
            "jcab,deij->cdeiab", v["ovvv"], t["vvoo"], optimize="optimal"
        )
        # print("wt2vvvovv = ",wt2vvvovv.shape)
        wt2_mat_3 = ducc.make_full_three(wt2_mat_3, wt2vvvovv, n_a, n_b, n_orb)

    wt2oooo = 0
    wt2oooo += 0.125000000 * np.einsum(
        "abij,klab->klij", v["vvoo"], t["oovv"], optimize="optimal"
    )
    wt2oooo += 0.125000000 * np.einsum(
        "ijab,abkl->ijkl", v["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("wt2oooo = ",wt2oooo.shape)
    wt2_mat_2 = ducc.make_full_two(wt2_mat_2, wt2oooo, n_a, n_b, n_orb)

    wt2ooov = 0
    wt2ooov += 1.000000000 * np.einsum(
        "jbil,klab->jkia", v["ovoo"], t["oovv"], optimize="optimal"
    )
    wt2ooov += 0.250000000 * np.einsum(
        "bcia,jkbc->jkia", v["vvov"], t["oovv"], optimize="optimal"
    )
    # print("wt2ooov = ",wt2ooov.shape)
    wt2_mat_2 = ducc.make_full_two(wt2_mat_2, wt2ooov, n_a, n_b, n_orb)

    wt2oovv = 0
    wt2oovv += 0.125000000 * np.einsum(
        "ijkl,klab->ijab", v["oooo"], t["oovv"], optimize="optimal"
    )
    wt2oovv += -1.000000000 * np.einsum(
        "icka,jkbc->ijab", v["ovov"], t["oovv"], optimize="optimal"
    )
    wt2oovv += 0.125000000 * np.einsum(
        "cdab,ijcd->ijab", v["vvvv"], t["oovv"], optimize="optimal"
    )
    # print("wt2oovv = ",wt2oovv.shape)
    wt2_mat_2 = ducc.make_full_two(wt2_mat_2, wt2oovv, n_a, n_b, n_orb)

    wt2ovoo = 0
    wt2ovoo += 1.000000000 * np.einsum(
        "jlib,abkl->jaik", v["ooov"], t["vvoo"], optimize="optimal"
    )
    wt2ovoo += 0.250000000 * np.einsum(
        "iabc,bcjk->iajk", v["ovvv"], t["vvoo"], optimize="optimal"
    )
    # print("wt2ovoo = ",wt2ovoo.shape)
    wt2_mat_2 = ducc.make_full_two(wt2_mat_2, wt2ovoo, n_a, n_b, n_orb)

    wt2ovov = 0
    wt2ovov += -1.000000000 * np.einsum(
        "acik,jkbc->jaib", v["vvoo"], t["oovv"], optimize="optimal"
    )
    wt2ovov += -1.000000000 * np.einsum(
        "ikac,bcjk->ibja", v["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("wt2ovov = ",wt2ovov.shape)
    wt2_mat_2 = ducc.make_full_two(wt2_mat_2, wt2ovov, n_a, n_b, n_orb)

    wt2ovvv = 0
    wt2ovvv += 0.250000000 * np.einsum(
        "iajk,jkbc->iabc", v["ovoo"], t["oovv"], optimize="optimal"
    )
    wt2ovvv += 1.000000000 * np.einsum(
        "bdja,ijcd->ibac", v["vvov"], t["oovv"], optimize="optimal"
    )
    # print("wt2ovvv = ",wt2ovvv.shape)
    wt2_mat_2 = ducc.make_full_two(wt2_mat_2, wt2ovvv, n_a, n_b, n_orb)

    wt2vvoo = 0
    wt2vvoo += 0.125000000 * np.einsum(
        "klij,abkl->abij", v["oooo"], t["vvoo"], optimize="optimal"
    )
    wt2vvoo += -1.000000000 * np.einsum(
        "kaic,bcjk->abij", v["ovov"], t["vvoo"], optimize="optimal"
    )
    wt2vvoo += 0.125000000 * np.einsum(
        "abcd,cdij->abij", v["vvvv"], t["vvoo"], optimize="optimal"
    )
    # print("wt2vvoo = ",wt2vvoo.shape)
    wt2_mat_2 = ducc.make_full_two(wt2_mat_2, wt2vvoo, n_a, n_b, n_orb)

    wt2vvov = 0
    wt2vvov += 0.250000000 * np.einsum(
        "jkia,bcjk->bcia", v["ooov"], t["vvoo"], optimize="optimal"
    )
    wt2vvov += 1.000000000 * np.einsum(
        "jbad,cdij->bcia", v["ovvv"], t["vvoo"], optimize="optimal"
    )
    # print("wt2vvov = ",wt2vvov.shape)
    wt2_mat_2 = ducc.make_full_two(wt2_mat_2, wt2vvov, n_a, n_b, n_orb)

    wt2vvvv = 0
    wt2vvvv += 0.125000000 * np.einsum(
        "abij,ijcd->abcd", v["vvoo"], t["oovv"], optimize="optimal"
    )
    wt2vvvv += 0.125000000 * np.einsum(
        "ijab,cdij->cdab", v["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("wt2vvvv = ",wt2vvvv.shape)
    wt2_mat_2 = ducc.make_full_two(wt2_mat_2, wt2vvvv, n_a, n_b, n_orb)

    wt2oo = 0
    wt2oo += 0.500000000 * np.einsum(
        "abik,jkab->ji", v["vvoo"], t["oovv"], optimize="optimal"
    )
    wt2oo += 0.500000000 * np.einsum(
        "ikab,abjk->ij", v["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("wt2oo = ",wt2oo.shape)
    wt2_mat_1 = ducc.make_full_one(wt2_mat_1, wt2oo, n_a, n_b, n_orb)

    wt2ov = 0
    wt2ov += -0.500000000 * np.einsum(
        "ibjk,jkab->ia", v["ovoo"], t["oovv"], optimize="optimal"
    )
    wt2ov += -0.500000000 * np.einsum(
        "bcja,ijbc->ia", v["vvov"], t["oovv"], optimize="optimal"
    )
    # print("wt2ov = ",wt2ov.shape)
    wt2_mat_1 = ducc.make_full_one(wt2_mat_1, wt2ov, n_a, n_b, n_orb)

    wt2vo = 0
    wt2vo += -0.500000000 * np.einsum(
        "jkib,abjk->ai", v["ooov"], t["vvoo"], optimize="optimal"
    )
    wt2vo += -0.500000000 * np.einsum(
        "jabc,bcij->ai", v["ovvv"], t["vvoo"], optimize="optimal"
    )
    # print("wt2vo = ",wt2vo.shape)
    wt2_mat_1 = ducc.make_full_one(wt2_mat_1, wt2vo, n_a, n_b, n_orb)

    wt2vv = 0
    wt2vv += -0.500000000 * np.einsum(
        "acij,ijbc->ab", v["vvoo"], t["oovv"], optimize="optimal"
    )
    wt2vv += -0.500000000 * np.einsum(
        "ijac,bcij->ba", v["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("wt2vv = ",wt2vv.shape)
    wt2_mat_1 = ducc.make_full_one(wt2_mat_1, wt2vv, n_a, n_b, n_orb)

    wt2 = 0
    wt2 += 0.250000000 * np.einsum(
        "abij,ijab->", v["vvoo"], t["oovv"], optimize="optimal"
    )
    wt2 += 0.250000000 * np.einsum(
        "ijab,abij->", v["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("wt2 = ",wt2)

    return wt2_mat_1, wt2_mat_2, wt2_mat_3, wt2


def compute_ft1t1(f, t, n_a, n_b, n_orb):
    # [[F,T1]T1]
    # t_ov = t1 and t_vo = transpose(t1)

    # Initializing the ft1t1_mat_1 one body
    ft1t1_mat_1 = np.zeros((2 * n_orb, 2 * n_orb))
    # print("ft1t1_mat_1 = ",ft1t1_mat_1.shape)

    ft1t1oo = 0
    ft1t1oo += -1.000000000 * np.einsum(
        "ki,ja,ak->ji", f["oo"], t["ov"], t["vo"], optimize="optimal"
    )
    ft1t1oo += -1.000000000 * np.einsum(
        "ik,ka,aj->ij", f["oo"], t["ov"], t["vo"], optimize="optimal"
    )
    ft1t1oo += 2.000000000 * np.einsum(
        "ba,ib,aj->ij", f["vv"], t["ov"], t["vo"], optimize="optimal"
    )
    # print("ft1t1oo = ",ft1t1oo.shape)
    ft1t1_mat_1 = ducc.make_full_one(ft1t1_mat_1, ft1t1oo, n_a, n_b, n_orb)

    ft1t1ov = 0
    ft1t1ov += -2.000000000 * np.einsum(
        "bj,ja,ib->ia", f["vo"], t["ov"], t["ov"], optimize="optimal"
    )
    ft1t1ov += -1.000000000 * np.einsum(
        "ja,ib,bj->ia", f["ov"], t["ov"], t["vo"], optimize="optimal"
    )
    ft1t1ov += -1.000000000 * np.einsum(
        "ib,ja,bj->ia", f["ov"], t["ov"], t["vo"], optimize="optimal"
    )
    # print("ft1t1ov = ",ft1t1ov.shape)
    ft1t1_mat_1 = ducc.make_full_one(ft1t1_mat_1, ft1t1ov, n_a, n_b, n_orb)

    ft1t1vo = 0
    ft1t1vo += -1.000000000 * np.einsum(
        "bi,jb,aj->ai", f["vo"], t["ov"], t["vo"], optimize="optimal"
    )
    ft1t1vo += -1.000000000 * np.einsum(
        "aj,jb,bi->ai", f["vo"], t["ov"], t["vo"], optimize="optimal"
    )
    ft1t1vo += -2.000000000 * np.einsum(
        "jb,bi,aj->ai", f["ov"], t["vo"], t["vo"], optimize="optimal"
    )
    # print("ft1t1vo = ",ft1t1vo.shape)
    ft1t1_mat_1 = ducc.make_full_one(ft1t1_mat_1, ft1t1vo, n_a, n_b, n_orb)

    ft1t1vv = 0
    ft1t1vv += 2.000000000 * np.einsum(
        "ji,ia,bj->ba", f["oo"], t["ov"], t["vo"], optimize="optimal"
    )
    ft1t1vv += -1.000000000 * np.einsum(
        "ca,ic,bi->ba", f["vv"], t["ov"], t["vo"], optimize="optimal"
    )
    ft1t1vv += -1.000000000 * np.einsum(
        "ac,ib,ci->ab", f["vv"], t["ov"], t["vo"], optimize="optimal"
    )
    # print("ft1t1vv = ",ft1t1vv.shape)
    ft1t1_mat_1 = ducc.make_full_one(ft1t1_mat_1, ft1t1vv, n_a, n_b, n_orb)

    ft1t1 = 0
    ft1t1 += -2.000000000 * np.einsum(
        "ji,ia,aj->", f["oo"], t["ov"], t["vo"], optimize="optimal"
    )
    ft1t1 += 2.000000000 * np.einsum(
        "ba,ib,ai->", f["vv"], t["ov"], t["vo"], optimize="optimal"
    )
    # print("ft1t1 = ",ft1t1)

    return ft1t1_mat_1, ft1t1


def compute_ft2t1(f, t, n_a, n_b, n_orb):
    # [[F,T2]T1]
    # t_ov = t1 and t_vo = transpose(t1)
    # t_oovv = t2 and t_vvoo = transpose(t2)

    # Initializing the ft2t1_mat_1 one body
    ft2t1_mat_1 = np.zeros((2 * n_orb, 2 * n_orb))
    # print("ft2t1_mat_1 = ",ft2t1_mat_1.shape)

    # Initializing the ft2t1_mat_2 two body
    ft2t1_mat_2 = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    # print("ft2t1_mat_2 = ",ft2t1_mat_2.shape)

    ft2t1oooo = 0
    ft2t1oooo += 0.500000000 * np.einsum(
        "ai,bj,klab->klij", f["vo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1oooo += 0.500000000 * np.einsum(
        "ia,jb,abkl->ijkl", f["ov"], t["ov"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t1oooo = ",ft2t1oooo.shape)
    ft2t1_mat_2 = ducc.make_full_two(ft2t1_mat_2, ft2t1oooo, n_a, n_b, n_orb)

    ft2t1ooov = 0
    ft2t1ooov += -1.000000000 * np.einsum(
        "il,bj,klab->ikja", f["oo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1ooov += -0.500000000 * np.einsum(
        "ba,ci,jkbc->jkia", f["vv"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1ooov += -0.500000000 * np.einsum(
        "cb,bi,jkac->jkia", f["vv"], t["vo"], t["oovv"], optimize="optimal"
    )
    # print("ft2t1ooov = ",ft2t1ooov.shape)
    ft2t1_mat_2 = ducc.make_full_two(ft2t1_mat_2, ft2t1ooov, n_a, n_b, n_orb)

    ft2t1oovv = 0
    ft2t1oovv += 0.500000000 * np.einsum(
        "ck,ka,ijbc->ijab", f["vo"], t["ov"], t["oovv"], optimize="optimal"
    )
    ft2t1oovv += 0.500000000 * np.einsum(
        "ck,ic,jkab->ijab", f["vo"], t["ov"], t["oovv"], optimize="optimal"
    )
    # print("ft2t1oovv = ",ft2t1oovv.shape)
    ft2t1_mat_2 = ducc.make_full_two(ft2t1_mat_2, ft2t1oovv, n_a, n_b, n_orb)

    ft2t1ovoo = 0
    ft2t1ovoo += -1.000000000 * np.einsum(
        "li,jb,abkl->jaik", f["oo"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft2t1ovoo += -0.500000000 * np.einsum(
        "ab,ic,bcjk->iajk", f["vv"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft2t1ovoo += -0.500000000 * np.einsum(
        "cb,ic,abjk->iajk", f["vv"], t["ov"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t1ovoo = ",ft2t1ovoo.shape)
    ft2t1_mat_2 = ducc.make_full_two(ft2t1_mat_2, ft2t1ovoo, n_a, n_b, n_orb)

    ft2t1ovov = 0
    ft2t1ovov += 1.000000000 * np.einsum(
        "ci,ak,jkbc->jaib", f["vo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1ovov += 1.000000000 * np.einsum(
        "ak,ci,jkbc->jaib", f["vo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1ovov += 1.000000000 * np.einsum(
        "ka,ic,bcjk->ibja", f["ov"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft2t1ovov += 1.000000000 * np.einsum(
        "ic,ka,bcjk->ibja", f["ov"], t["ov"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t1ovov = ",ft2t1ovov.shape)
    ft2t1_mat_2 = ducc.make_full_two(ft2t1_mat_2, ft2t1ovov, n_a, n_b, n_orb)

    ft2t1ovvv = 0
    ft2t1ovvv += 0.500000000 * np.einsum(
        "ij,ak,jkbc->iabc", f["oo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1ovvv += 0.500000000 * np.einsum(
        "kj,ak,ijbc->iabc", f["oo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1ovvv += 1.000000000 * np.einsum(
        "da,bj,ijcd->ibac", f["vv"], t["vo"], t["oovv"], optimize="optimal"
    )
    # print("ft2t1ovvv = ",ft2t1ovvv.shape)
    ft2t1_mat_2 = ducc.make_full_two(ft2t1_mat_2, ft2t1ovvv, n_a, n_b, n_orb)

    ft2t1vvoo = 0
    ft2t1vvoo += 0.500000000 * np.einsum(
        "kc,ci,abjk->abij", f["ov"], t["vo"], t["vvoo"], optimize="optimal"
    )
    ft2t1vvoo += 0.500000000 * np.einsum(
        "kc,ak,bcij->abij", f["ov"], t["vo"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t1vvoo = ",ft2t1vvoo.shape)
    ft2t1_mat_2 = ducc.make_full_two(ft2t1_mat_2, ft2t1vvoo, n_a, n_b, n_orb)

    ft2t1vvov = 0
    ft2t1vvov += 0.500000000 * np.einsum(
        "ji,ka,bcjk->bcia", f["oo"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft2t1vvov += 0.500000000 * np.einsum(
        "kj,ja,bcik->bcia", f["oo"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft2t1vvov += 1.000000000 * np.einsum(
        "ad,jb,cdij->acib", f["vv"], t["ov"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t1vvov = ",ft2t1vvov.shape)
    ft2t1_mat_2 = ducc.make_full_two(ft2t1_mat_2, ft2t1vvov, n_a, n_b, n_orb)

    ft2t1vvvv = 0
    ft2t1vvvv += 0.500000000 * np.einsum(
        "ai,bj,ijcd->abcd", f["vo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1vvvv += 0.500000000 * np.einsum(
        "ia,jb,cdij->cdab", f["ov"], t["ov"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t1vvvv = ",ft2t1vvvv.shape)
    ft2t1_mat_2 = ducc.make_full_two(ft2t1_mat_2, ft2t1vvvv, n_a, n_b, n_orb)

    ft2t1oo = 0
    ft2t1oo += 1.000000000 * np.einsum(
        "ai,bk,jkab->ji", f["vo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1oo += -1.000000000 * np.einsum(
        "ak,bi,jkab->ji", f["vo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1oo += 1.000000000 * np.einsum(
        "ia,kb,abjk->ij", f["ov"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft2t1oo += -1.000000000 * np.einsum(
        "ka,ib,abjk->ij", f["ov"], t["ov"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t1oo = ",ft2t1oo.shape)
    ft2t1_mat_1 = ducc.make_full_one(ft2t1_mat_1, ft2t1oo, n_a, n_b, n_orb)

    ft2t1ov = 0
    ft2t1ov += -1.000000000 * np.einsum(
        "ij,bk,jkab->ia", f["oo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1ov += -1.000000000 * np.einsum(
        "kj,bk,ijab->ia", f["oo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1ov += 1.000000000 * np.einsum(
        "ba,cj,ijbc->ia", f["vv"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1ov += 1.000000000 * np.einsum(
        "cb,bj,ijac->ia", f["vv"], t["vo"], t["oovv"], optimize="optimal"
    )
    # print("ft2t1ov = ",ft2t1ov.shape)
    ft2t1_mat_1 = ducc.make_full_one(ft2t1_mat_1, ft2t1ov, n_a, n_b, n_orb)

    ft2t1vo = 0
    ft2t1vo += -1.000000000 * np.einsum(
        "ji,kb,abjk->ai", f["oo"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft2t1vo += -1.000000000 * np.einsum(
        "kj,jb,abik->ai", f["oo"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft2t1vo += 1.000000000 * np.einsum(
        "ab,jc,bcij->ai", f["vv"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft2t1vo += 1.000000000 * np.einsum(
        "cb,jc,abij->ai", f["vv"], t["ov"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t1vo = ",ft2t1vo.shape)
    ft2t1_mat_1 = ducc.make_full_one(ft2t1_mat_1, ft2t1vo, n_a, n_b, n_orb)

    ft2t1vv = 0
    ft2t1vv += -1.000000000 * np.einsum(
        "ai,cj,ijbc->ab", f["vo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1vv += 1.000000000 * np.einsum(
        "ci,aj,ijbc->ab", f["vo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1vv += -1.000000000 * np.einsum(
        "ia,jc,bcij->ba", f["ov"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft2t1vv += 1.000000000 * np.einsum(
        "ic,ja,bcij->ba", f["ov"], t["ov"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t1vv = ",ft2t1vv.shape)
    ft2t1_mat_1 = ducc.make_full_one(ft2t1_mat_1, ft2t1vv, n_a, n_b, n_orb)

    ft2t1 = 0
    ft2t1 += 1.000000000 * np.einsum(
        "ai,bj,ijab->", f["vo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft2t1 += 1.000000000 * np.einsum(
        "ia,jb,abij->", f["ov"], t["ov"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t1 = ",ft2t1)

    return ft2t1_mat_1, ft2t1_mat_2, ft2t1


def compute_ft1t2(f, t, n_a, n_b, n_orb):
    # [[F,T1]T2]
    # t_ov = t1 and t_vo = transpose(t1)
    # t_oovv = t2 and t_vvoo = transpose(t2)

    # Initializing the ft1t2_mat_1 one body
    ft1t2_mat_1 = np.zeros((2 * n_orb, 2 * n_orb))
    # print("ft1t2_mat_1 = ",ft1t2_mat_1.shape)

    # Initializing the ft1t2_mat_2 two body
    ft1t2_mat_2 = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    # print("ft1t2_mat_2 = ",ft1t2_mat_2.shape)

    ft1t2ooov = 0
    ft1t2ooov += 0.500000000 * np.einsum(
        "li,bl,jkab->jkia", f["oo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft1t2ooov += -0.500000000 * np.einsum(
        "cb,bi,jkac->jkia", f["vv"], t["vo"], t["oovv"], optimize="optimal"
    )
    # print("ft1t2ooov = ",ft1t2ooov.shape)
    ft1t2_mat_2 = ducc.make_full_two(ft1t2_mat_2, ft1t2ooov, n_a, n_b, n_orb)

    ft1t2oovv = 0
    ft1t2oovv += 0.500000000 * np.einsum(
        "ck,ka,ijbc->ijab", f["vo"], t["ov"], t["oovv"], optimize="optimal"
    )
    ft1t2oovv += 0.500000000 * np.einsum(
        "ck,ic,jkab->ijab", f["vo"], t["ov"], t["oovv"], optimize="optimal"
    )
    ft1t2oovv += 0.500000000 * np.einsum(
        "ka,ck,ijbc->ijab", f["ov"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft1t2oovv += 0.500000000 * np.einsum(
        "ic,ck,jkab->ijab", f["ov"], t["vo"], t["oovv"], optimize="optimal"
    )
    # print("ft1t2oovv = ",ft1t2oovv.shape)
    ft1t2_mat_2 = ducc.make_full_two(ft1t2_mat_2, ft1t2oovv, n_a, n_b, n_orb)

    ft1t2ovoo = 0
    ft1t2ovoo += 0.500000000 * np.einsum(
        "il,lb,abjk->iajk", f["oo"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft1t2ovoo += -0.500000000 * np.einsum(
        "cb,ic,abjk->iajk", f["vv"], t["ov"], t["vvoo"], optimize="optimal"
    )
    # print("ft1t2ovoo = ",ft1t2ovoo.shape)
    ft1t2_mat_2 = ducc.make_full_two(ft1t2_mat_2, ft1t2ovoo, n_a, n_b, n_orb)

    ft1t2ovvv = 0
    ft1t2ovvv += 0.500000000 * np.einsum(
        "kj,ak,ijbc->iabc", f["oo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft1t2ovvv += -0.500000000 * np.einsum(
        "ad,dj,ijbc->iabc", f["vv"], t["vo"], t["oovv"], optimize="optimal"
    )
    # print("ft1t2ovvv = ",ft1t2ovvv.shape)
    ft1t2_mat_2 = ducc.make_full_two(ft1t2_mat_2, ft1t2ovvv, n_a, n_b, n_orb)

    ft1t2vvoo = 0
    ft1t2vvoo += 0.500000000 * np.einsum(
        "ci,kc,abjk->abij", f["vo"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft1t2vvoo += 0.500000000 * np.einsum(
        "ak,kc,bcij->abij", f["vo"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft1t2vvoo += 0.500000000 * np.einsum(
        "kc,ci,abjk->abij", f["ov"], t["vo"], t["vvoo"], optimize="optimal"
    )
    ft1t2vvoo += 0.500000000 * np.einsum(
        "kc,ak,bcij->abij", f["ov"], t["vo"], t["vvoo"], optimize="optimal"
    )
    # print("ft1t2vvoo = ",ft1t2vvoo.shape)
    ft1t2_mat_2 = ducc.make_full_two(ft1t2_mat_2, ft1t2vvoo, n_a, n_b, n_orb)

    ft1t2vvov = 0
    ft1t2vvov += 0.500000000 * np.einsum(
        "kj,ja,bcik->bcia", f["oo"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft1t2vvov += -0.500000000 * np.einsum(
        "da,jd,bcij->bcia", f["vv"], t["ov"], t["vvoo"], optimize="optimal"
    )
    # print("ft1t2vvov = ",ft1t2vvov.shape)
    ft1t2_mat_2 = ducc.make_full_two(ft1t2_mat_2, ft1t2vvov, n_a, n_b, n_orb)

    ft1t2ov = 0
    ft1t2ov += -1.000000000 * np.einsum(
        "kj,bk,ijab->ia", f["oo"], t["vo"], t["oovv"], optimize="optimal"
    )
    ft1t2ov += 1.000000000 * np.einsum(
        "cb,bj,ijac->ia", f["vv"], t["vo"], t["oovv"], optimize="optimal"
    )
    # print("ft1t2ov = ",ft1t2ov.shape)
    ft1t2_mat_1 = ducc.make_full_one(ft1t2_mat_1, ft1t2ov, n_a, n_b, n_orb)

    ft1t2vo = 0
    ft1t2vo += -1.000000000 * np.einsum(
        "kj,jb,abik->ai", f["oo"], t["ov"], t["vvoo"], optimize="optimal"
    )
    ft1t2vo += 1.000000000 * np.einsum(
        "cb,jc,abij->ai", f["vv"], t["ov"], t["vvoo"], optimize="optimal"
    )
    # print("ft1t2vo = ",ft1t2vo.shape)
    ft1t2_mat_1 = ducc.make_full_one(ft1t2_mat_1, ft1t2vo, n_a, n_b, n_orb)

    return ft1t2_mat_1, ft1t2_mat_2


def compute_ft2t2(f, t, n_a, n_b, n_orb, compute_three_body=False):
    # [[F,T2]T2]
    # t_oovv = t2 and t_vvoo = transpose(t2)

    # Initializing the ft2t2_mat_1 one body
    ft2t2_mat_1 = np.zeros((2 * n_orb, 2 * n_orb))
    # print("ft2t2_mat_1 = ",ft2t2_mat_1.shape)

    # Initializing the ft2t2_mat_2 two body
    ft2t2_mat_2 = np.zeros((2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb))
    # print("ft2t2_mat_2 = ",ft2t2_mat_2.shape)

    # Initializing the ft2t2_mat_3 three body
    ft2t2_mat_3 = np.zeros(
        (2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb, 2 * n_orb)
    )
    # print("ft2t2_mat_3 = ",ft2t2_mat_3.shape)

    if compute_three_body == True:
        ft2t2ooooov = 0
        ft2t2ooooov += -0.250000000 * np.einsum(
            "ib,jkac,bclm->ijklma", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        # print("ft2t2ooooov = ",ft2t2ooooov.shape)
        ft2t2_mat_3 = ducc.make_full_three(ft2t2_mat_3, ft2t2ooooov, n_a, n_b, n_orb)

        ft2t2ooovvv = 0
        ft2t2ooovvv += 0.500000000 * np.einsum(
            "dl,ilab,jkcd->ijkabc", f["vo"], t["oovv"], t["oovv"], optimize="optimal"
        )
        # print("ft2t2ooovvv = ",ft2t2ooovvv.shape)
        ft2t2_mat_3 = ducc.make_full_three(ft2t2_mat_3, ft2t2ooovvv, n_a, n_b, n_orb)

        ft2t2oovooo = 0
        ft2t2oovooo += -0.250000000 * np.einsum(
            "bi,jkbc,aclm->jkailm", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        # print("ft2t2oovooo = ",ft2t2oovooo.shape)
        ft2t2_mat_3 = ducc.make_full_three(ft2t2_mat_3, ft2t2oovooo, n_a, n_b, n_orb)

        ft2t2oovoov = 0
        ft2t2oovoov += -0.500000000 * np.einsum(
            "mi,jkac,bclm->jkbila", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        ft2t2oovoov += -0.500000000 * np.einsum(
            "im,jmac,bckl->ijbkla", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        ft2t2oovoov += -0.250000000 * np.einsum(
            "ca,ijcd,bdkl->ijbkla", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        ft2t2oovoov += -0.250000000 * np.einsum(
            "ac,ijbd,cdkl->ijaklb", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        ft2t2oovoov += -0.500000000 * np.einsum(
            "dc,ijad,bckl->ijbkla", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        # print("ft2t2oovoov = ",ft2t2oovoov.shape)
        ft2t2_mat_3 = ducc.make_full_three(ft2t2_mat_3, ft2t2oovoov, n_a, n_b, n_orb)

        ft2t2oovovv = 0
        ft2t2oovovv += 0.500000000 * np.einsum(
            "la,ijbd,cdkl->ijckab", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        ft2t2oovovv += 0.500000000 * np.einsum(
            "id,jlab,cdkl->ijckab", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        # print("ft2t2oovovv = ",ft2t2oovovv.shape)
        ft2t2_mat_3 = ducc.make_full_three(ft2t2_mat_3, ft2t2oovovv, n_a, n_b, n_orb)

        ft2t2ovvoov = 0
        ft2t2ovvoov += 0.500000000 * np.einsum(
            "di,jlad,bckl->jbcika", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        ft2t2ovvoov += 0.500000000 * np.einsum(
            "al,ilbd,cdjk->iacjkb", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        # print("ft2t2ovvoov = ",ft2t2ovvoov.shape)
        ft2t2_mat_3 = ducc.make_full_three(ft2t2_mat_3, ft2t2ovvoov, n_a, n_b, n_orb)

        ft2t2ovvovv = 0
        ft2t2ovvovv += -0.250000000 * np.einsum(
            "ki,jlab,cdkl->jcdiab", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        ft2t2ovvovv += -0.250000000 * np.einsum(
            "ik,klab,cdjl->icdjab", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        ft2t2ovvovv += -0.500000000 * np.einsum(
            "lk,ikab,cdjl->icdjab", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        ft2t2ovvovv += -0.500000000 * np.einsum(
            "ea,ikbe,cdjk->icdjab", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        ft2t2ovvovv += -0.500000000 * np.einsum(
            "ae,ikbc,dejk->iadjbc", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        # print("ft2t2ovvovv = ",ft2t2ovvovv.shape)
        ft2t2_mat_3 = ducc.make_full_three(ft2t2_mat_3, ft2t2ovvovv, n_a, n_b, n_orb)

        ft2t2ovvvvv = 0
        ft2t2ovvvvv += -0.250000000 * np.einsum(
            "ja,ikbc,dejk->ideabc", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        # print("ft2t2ovvvvv = ",ft2t2ovvvvv.shape)
        ft2t2_mat_3 = ducc.make_full_three(ft2t2_mat_3, ft2t2ovvvvv, n_a, n_b, n_orb)

        ft2t2vvvooo = 0
        ft2t2vvvooo += 0.500000000 * np.einsum(
            "ld,adij,bckl->abcijk", f["ov"], t["vvoo"], t["vvoo"], optimize="optimal"
        )
        # print("ft2t2vvvooo = ",ft2t2vvvooo.shape)
        ft2t2_mat_3 = ducc.make_full_three(ft2t2_mat_3, ft2t2vvvooo, n_a, n_b, n_orb)

        ft2t2vvvovv = 0
        ft2t2vvvovv += -0.250000000 * np.einsum(
            "aj,jkbc,deik->adeibc", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
        )
        # print("ft2t2vvvovv = ",ft2t2vvvovv.shape)
        ft2t2_mat_3 = ducc.make_full_three(ft2t2_mat_3, ft2t2vvvovv, n_a, n_b, n_orb)

    ft2t2oooo = 0
    ft2t2oooo += 0.250000000 * np.einsum(
        "mi,jkab,ablm->jkil", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2oooo += 0.250000000 * np.einsum(
        "im,jmab,abkl->ijkl", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2oooo += 0.500000000 * np.einsum(
        "ba,ijbc,ackl->ijkl", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t2oooo = ",ft2t2oooo.shape)
    ft2t2_mat_2 = ducc.make_full_two(ft2t2_mat_2, ft2t2oooo, n_a, n_b, n_orb)

    ft2t2ooov = 0
    ft2t2ooov += -0.250000000 * np.einsum(
        "la,ijbc,bckl->ijka", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ooov += 1.000000000 * np.einsum(
        "ib,jlac,bckl->ijka", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ooov += 0.500000000 * np.einsum(
        "lb,ijac,bckl->ijka", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t2ooov = ",ft2t2ooov.shape)
    ft2t2_mat_2 = ducc.make_full_two(ft2t2_mat_2, ft2t2ooov, n_a, n_b, n_orb)

    ft2t2ovov = 0
    ft2t2ovov += 1.000000000 * np.einsum(
        "ki,jlac,bckl->jbia", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ovov += 1.000000000 * np.einsum(
        "ik,klac,bcjl->ibja", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ovov += 2.000000000 * np.einsum(
        "lk,ikac,bcjl->ibja", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ovov += -1.000000000 * np.einsum(
        "ca,ikcd,bdjk->ibja", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ovov += -1.000000000 * np.einsum(
        "ac,ikbd,cdjk->iajb", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ovov += -2.000000000 * np.einsum(
        "dc,ikad,bcjk->ibja", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t2ovov = ",ft2t2ovov.shape)
    ft2t2_mat_2 = ducc.make_full_two(ft2t2_mat_2, ft2t2ovov, n_a, n_b, n_orb)

    ft2t2ovoo = 0
    ft2t2ovoo += 1.000000000 * np.einsum(
        "bi,jlbc,ackl->jaik", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ovoo += -0.250000000 * np.einsum(
        "al,ilbc,bcjk->iajk", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ovoo += 0.500000000 * np.einsum(
        "bl,ilbc,acjk->iajk", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t2ovoo = ",ft2t2ovoo.shape)
    ft2t2_mat_2 = ducc.make_full_two(ft2t2_mat_2, ft2t2ovoo, n_a, n_b, n_orb)

    ft2t2ovvv = 0
    ft2t2ovvv += 1.000000000 * np.einsum(
        "ja,ikbd,cdjk->icab", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ovvv += -0.250000000 * np.einsum(
        "id,jkab,cdjk->icab", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ovvv += 0.500000000 * np.einsum(
        "jd,ikab,cdjk->icab", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t2ovvv = ",ft2t2ovvv.shape)
    ft2t2_mat_2 = ducc.make_full_two(ft2t2_mat_2, ft2t2ovvv, n_a, n_b, n_orb)

    ft2t2vvov = 0
    ft2t2vvov += -0.250000000 * np.einsum(
        "di,jkad,bcjk->bcia", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2vvov += 1.000000000 * np.einsum(
        "aj,jkbd,cdik->acib", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2vvov += 0.500000000 * np.einsum(
        "dj,jkad,bcik->bcia", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t2vvov = ",ft2t2vvov.shape)
    ft2t2_mat_2 = ducc.make_full_two(ft2t2_mat_2, ft2t2vvov, n_a, n_b, n_orb)

    ft2t2vvvv = 0
    ft2t2vvvv += -0.500000000 * np.einsum(
        "ji,ikab,cdjk->cdab", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2vvvv += -0.250000000 * np.einsum(
        "ea,ijbe,cdij->cdab", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2vvvv += -0.250000000 * np.einsum(
        "ae,ijbc,deij->adbc", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t2vvvv = ",ft2t2vvvv.shape)
    ft2t2_mat_2 = ducc.make_full_two(ft2t2_mat_2, ft2t2vvvv, n_a, n_b, n_orb)

    ft2t2oo = 0
    ft2t2oo += -0.500000000 * np.einsum(
        "ki,jlab,abkl->ji", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2oo += -0.500000000 * np.einsum(
        "ik,klab,abjl->ij", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2oo += -1.000000000 * np.einsum(
        "lk,ikab,abjl->ij", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2oo += 2.000000000 * np.einsum(
        "ba,ikbc,acjk->ij", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t2oo = ",ft2t2oo.shape)
    ft2t2_mat_1 = ducc.make_full_one(ft2t2_mat_1, ft2t2oo, n_a, n_b, n_orb)

    ft2t2ov = 0
    ft2t2ov += -0.500000000 * np.einsum(
        "ja,ikbc,bcjk->ia", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ov += -0.500000000 * np.einsum(
        "ib,jkac,bcjk->ia", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2ov += 1.000000000 * np.einsum(
        "jb,ikac,bcjk->ia", f["ov"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t2ov = ",ft2t2ov.shape)
    ft2t2_mat_1 = ducc.make_full_one(ft2t2_mat_1, ft2t2ov, n_a, n_b, n_orb)

    ft2t2vo = 0
    ft2t2vo += -0.500000000 * np.einsum(
        "bi,jkbc,acjk->ai", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2vo += -0.500000000 * np.einsum(
        "aj,jkbc,bcik->ai", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2vo += 1.000000000 * np.einsum(
        "bj,jkbc,acik->ai", f["vo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t2vo = ",ft2t2vo.shape)
    ft2t2_mat_1 = ducc.make_full_one(ft2t2_mat_1, ft2t2vo, n_a, n_b, n_orb)

    ft2t2vv = 0
    ft2t2vv += 2.000000000 * np.einsum(
        "ji,ikac,bcjk->ba", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2vv += -0.500000000 * np.einsum(
        "ca,ijcd,bdij->ba", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2vv += -0.500000000 * np.einsum(
        "ac,ijbd,cdij->ab", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2vv += -1.000000000 * np.einsum(
        "dc,ijad,bcij->ba", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t2vv = ",ft2t2vv.shape)
    ft2t2_mat_1 = ducc.make_full_one(ft2t2_mat_1, ft2t2vv, n_a, n_b, n_orb)

    ft2t2 = 0
    ft2t2 += -1.000000000 * np.einsum(
        "ji,ikab,abjk->", f["oo"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    ft2t2 += 1.000000000 * np.einsum(
        "ba,ijbc,acij->", f["vv"], t["oovv"], t["vvoo"], optimize="optimal"
    )
    # print("ft2t2 = ",ft2t2)

    return ft2t2_mat_1, ft2t2_mat_2, ft2t2_mat_3, ft2t2
