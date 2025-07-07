import time
import numpy as np
from downfolding.diis import DIIS 
from itertools import product
from opt_einsum import contract
from downfolding.helper import one_body_mat2dic, two_body_ten2dic, asym_term


def calc_ccsd(fmat, vten, t1_amps, t2_amps, verbose=0):
    """
    CCSd closed-shell energy (restricted spin-orbitals).

    Parameters
    ----------
    fmat, vten, t1_amps, t2_amps

    Returns
    -------
    float
        Total CCSD energy
    """
    t0 = time.perf_counter()

    n_a = t1_amps[0].shape[0]
    n_orb = n_a + t1_amps[0].shape[1]
    vten = 4 * vten  

    # Define slices 
    occ_alpha = slice(0, 2*n_a, 2)     # Alpha occupied indices
    occ_beta = slice(1, 2*n_a, 2)      # Beta occupied indices
    virt_alpha = slice(2*n_a, 2*n_orb, 2)  # Alpha virtual indices
    virt_beta = slice(2*n_a + 1, 2*n_orb, 2)  # Beta virtual indices

    # Initialize correlation energy
    ECCSD = 0.0

    # Singles (t1) contributions
    ECCSD += contract('ia,ia->', fmat[occ_alpha, virt_alpha], t1_amps[0])
    ECCSD += contract('ia,ia->', fmat[occ_beta, virt_beta], t1_amps[1])

    # Doubles (t2) contributions
    ECCSD += 0.25 * contract('ijab,ijab->', t2_amps[0], vten[occ_alpha, occ_alpha, virt_alpha, virt_alpha])
    ECCSD += 0.25 * contract('ijab,ijab->', t2_amps[2], vten[occ_beta, occ_beta, virt_beta, virt_beta])
    ECCSD += 1.0 * contract('ijab,ijab->', t2_amps[1], vten[occ_alpha, occ_beta, virt_alpha, virt_beta])

    # Mixed singles (t1) contributions
    ECCSD += 0.5 * contract('ia,jb,ijab->', t1_amps[0], t1_amps[0], vten[occ_alpha, occ_alpha, virt_alpha, virt_alpha])
    ECCSD += 0.5 * contract('ia,jb,ijab->', t1_amps[1], t1_amps[1], vten[occ_beta, occ_beta, virt_beta, virt_beta])
    ECCSD += 1.0 * contract('ia,jb,ijab->', t1_amps[0], t1_amps[1], vten[occ_alpha, occ_beta, virt_alpha, virt_beta])

    dt = time.perf_counter() - t0
    m, s = divmod(dt, 60)

    if verbose != 0:
        print("\n   CCSD Calculation Summary")
        print("   -------------------------------------")
        print(f"   Total wall time: {m:0.2f} m  {s:0.2f} s")
        print(f"   CCSD Correction = {ECCSD:16.10f}")

    return ECCSD


def compute_inverse_denominators(H: dict, nocc: list[int], nvir: list[int], rank: int):
    """
    A function to compute the inverse of Møller-Plesset denominators
    """
    fo = np.diag(H["oo"])
    fv = np.diag(H["vv"])

    D = {}

    if rank >= 1:
        D["ov"] = 1.0 / (fo.reshape(-1, 1) - fv)

    if rank >= 2:
        D["oovv"] = 1.0 / (
            fo.reshape(-1, 1, 1, 1) + fo.reshape(-1, 1, 1) - fv.reshape(-1, 1) - fv
        )

    if rank >= 3:
        D["ooovvv"] = 1.0 / (
            fo.reshape(-1, 1, 1, 1, 1, 1)
            + fo.reshape(-1, 1, 1, 1, 1)
            + fo.reshape(-1, 1, 1, 1)
            - fv.reshape(-1, 1, 1)
            - fv.reshape(-1, 1)
            - fv
        )
    if rank > 3:
        raise ValueError(
            f"compute_inverse_denominators() supports rank up to 3, but was called with rank = {rank}"
        )
    return D

def update_cc_amplitudes(T, R, invD, rank: int):
    """
    A function to update the CCSD amplitudes

    Parameters
    ----------
    T : dict[np.ndarray]
        The cluster amplitudes
    R : dict[np.ndarray]
        The CC residual
    invD : dict[np.ndarray]
        The inverse MP denominators
    rank : int
        The rank of the CC equations (e.g., CCSD : rank = 2)
    """
    if rank >= 1:
        T["ov"] += contract("ia,ia->ia", R["ov"], invD["ov"])
    if rank >= 2:
        T["oovv"] += contract("ijab,ijab->ijab", R["oovv"], invD["oovv"])
    if rank >= 3:
        T["ooovvv"] += contract("ijkabc,ijkabc->ijkabc", R["ooovvv"], invD["ooovvv"])

    return T 

def evaluate_residual_0_0(F,V,T):
    # contributions to the residual
    R = 0.0
    R += 1.000000000 * contract("ai,ia->",F["vo"],T["ov"])
    R += 0.500000000 * contract("ia,jb,abij->",T["ov"],T["ov"],V["vvoo"])
    R += 0.250000000 * contract("ijab,abij->",T["oovv"],V["vvoo"])
    return R

def evaluate_residual_1_1(F,V,T,nocc,nvir):
    # contributions to the residual
    Rov = np.zeros((nocc,nvir))
    Rov += -1.000000000 * contract("ij,ja->ia",F["oo"],T["ov"])
    Rov += -1.000000000 * contract("bj,ja,ib->ia",F["vo"],T["ov"],T["ov"])
    Rov += 1.000000000 * contract("bj,ijab->ia",F["vo"],T["oovv"])
    Rov += 1.000000000 * contract("ia->ia",F["ov"])
    Rov += 1.000000000 * contract("ba,ib->ia",F["vv"],T["ov"])
    Rov += -1.000000000 * contract("ja,ib,kc,bcjk->ia",T["ov"],T["ov"],T["ov"],V["vvoo"])
    Rov += -1.000000000 * contract("ja,kb,ibjk->ia",T["ov"],T["ov"],V["ovoo"])
    Rov += -0.500000000 * contract("ja,ikbc,bcjk->ia",T["ov"],T["oovv"],V["vvoo"])
    Rov += -0.500000000 * contract("jkab,ibjk->ia",T["oovv"],V["ovoo"])
    Rov += -0.500000000 * contract("ib,jkac,bcjk->ia",T["ov"],T["oovv"],V["vvoo"])
    Rov += -1.000000000 * contract("ib,jc,bcja->ia",T["ov"],T["ov"],V["vvov"])
    Rov += 1.000000000 * contract("jb,ikac,bcjk->ia",T["ov"],T["oovv"],V["vvoo"])
    Rov += -1.000000000 * contract("jb,ibja->ia",T["ov"],V["ovov"])
    Rov += -0.500000000 * contract("ijbc,bcja->ia",T["oovv"],V["vvov"])
    return Rov

def evaluate_residual_2_2(F,V,T,nocc,nvir):
    # contributions to the residual
    Roovv = np.zeros((nocc,nocc,nvir,nvir))
    Roovv += 0.500000000 * contract("ik,jkab->ijab",F["oo"],T["oovv"])
    Roovv += 0.500000000 * contract("ck,ka,ijbc->ijab",F["vo"],T["ov"],T["oovv"])
    Roovv += 0.500000000 * contract("ck,ic,jkab->ijab",F["vo"],T["ov"],T["oovv"])
    Roovv += -0.500000000 * contract("ca,ijbc->ijab",F["vv"],T["oovv"])
    Roovv += 0.250000000 * contract("ka,lb,ic,jd,cdkl->ijab",T["ov"],T["ov"],T["ov"],T["ov"],V["vvoo"])
    Roovv += -0.500000000 * contract("ka,lb,ic,jckl->ijab",T["ov"],T["ov"],T["ov"],V["ovoo"])
    Roovv += 0.125000000 * contract("ka,lb,ijcd,cdkl->ijab",T["ov"],T["ov"],T["oovv"],V["vvoo"])
    Roovv += 0.250000000 * contract("ka,lb,ijkl->ijab",T["ov"],T["ov"],V["oooo"])
    Roovv += 1.000000000 * contract("ka,ilbc,jckl->ijab",T["ov"],T["oovv"],V["ovoo"])
    Roovv += -1.000000000 * contract("ka,ic,jlbd,cdkl->ijab",T["ov"],T["ov"],T["oovv"],V["vvoo"])
    Roovv += -0.500000000 * contract("ka,ic,jd,cdkb->ijab",T["ov"],T["ov"],T["ov"],V["vvov"])
    Roovv += 1.000000000 * contract("ka,ic,jckb->ijab",T["ov"],T["ov"],V["ovov"])
    Roovv += -0.500000000 * contract("ka,lc,ijbd,cdkl->ijab",T["ov"],T["ov"],T["oovv"],V["vvoo"])
    Roovv += -0.250000000 * contract("ka,ijcd,cdkb->ijab",T["ov"],T["oovv"],V["vvov"])
    Roovv += -0.500000000 * contract("ka,ijkb->ijab",T["ov"],V["ooov"])
    Roovv += -0.250000000 * contract("ikab,jlcd,cdkl->ijab",T["oovv"],T["oovv"],V["vvoo"])
    Roovv += 0.062500000 * contract("klab,ijcd,cdkl->ijab",T["oovv"],T["oovv"],V["vvoo"])
    Roovv += 0.125000000 * contract("klab,ijkl->ijab",T["oovv"],V["oooo"])
    Roovv += -0.250000000 * contract("ijac,klbd,cdkl->ijab",T["oovv"],T["oovv"],V["vvoo"])
    Roovv += 0.500000000 * contract("ikac,jlbd,cdkl->ijab",T["oovv"],T["oovv"],V["vvoo"])
    Roovv += -1.000000000 * contract("ikac,jckb->ijab",T["oovv"],V["ovov"])
    Roovv += -0.250000000 * contract("ic,klab,jckl->ijab",T["ov"],T["oovv"],V["ovoo"])
    Roovv += 1.000000000 * contract("ic,jkad,cdkb->ijab",T["ov"],T["oovv"],V["vvov"])
    Roovv += 0.125000000 * contract("ic,jd,klab,cdkl->ijab",T["ov"],T["ov"],T["oovv"],V["vvoo"])
    Roovv += 0.250000000 * contract("ic,jd,cdab->ijab",T["ov"],T["ov"],V["vvvv"])
    Roovv += -0.500000000 * contract("ic,kd,jlab,cdkl->ijab",T["ov"],T["ov"],T["oovv"],V["vvoo"])
    Roovv += -0.500000000 * contract("ic,jcab->ijab",T["ov"],V["ovvv"])
    Roovv += 0.500000000 * contract("kc,ilab,jckl->ijab",T["ov"],T["oovv"],V["ovoo"])
    Roovv += 0.500000000 * contract("kc,ijad,cdkb->ijab",T["ov"],T["oovv"],V["vvov"])
    Roovv += 0.125000000 * contract("ijcd,cdab->ijab",T["oovv"],V["vvvv"])
    Roovv += 0.250000000 * contract("ijab->ijab",V["oovv"])

    Roovv = 4*asym_term(Roovv,"oovv")
    return Roovv

def kernel_w(F, V, T, invD, nocc, nvirt, hf_energy, diis_size=None, diis_start_cycle=4, max_iter=150, conv_normr=1.0E-9):
    if diis_size is not None:
        print(f"\nRunning CCSD (fast) with DIIS (diis_size={diis_size}, diis_start_cycle={diis_start_cycle}) ")
        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        t1_dim = T["ov"].size
        old_vec = np.hstack((T["ov"].flatten(), T["oovv"].flatten()))
    else:
        print("\nRunning CCSD (fast) without DIIS")

    # Emp2 = 0.0
    # for i in range(nocc):
    #     for j in range(nocc):
    #         for a in range(nvirt):
    #             for b in range(nvirt):
    #                 Emp2 += 0.25 * V["oovv"][i][j][a][b] ** 2 * invD["oovv"][i][j][a][b]
    # print(f"MP2 correlation energy: {Emp2:.12f} Eh")

    header = "Iter.     Energy [Eh]    Corr. energy [Eh]       |R|       "
    print(header)

    for i in range(max_iter):
        # 1. compute energy and residuals
        R = {}
        Ecorr_w = evaluate_residual_0_0(F, V, T)
        Etot_w = hf_energy + Ecorr_w
        R["ov"] = evaluate_residual_1_1(F, V, T, nocc, nvirt)
        R["oovv"] = evaluate_residual_2_2(F, V, T, nocc, nvirt)

        # 2. amplitude update
        T = update_cc_amplitudes(T, R, invD, 2)

        # 3. check diis
        if diis_size is not None:
            vectorized_iterate = np.hstack(
                (T["ov"].flatten(), T["oovv"].flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                error_vec)
            T["ov"] = new_vectorized_iterate[:t1_dim].reshape(T["ov"].shape)
            T["oovv"] = new_vectorized_iterate[t1_dim:].reshape(T["oovv"].shape)
            old_vec = new_vectorized_iterate

        # 3. check for convergence
        norm_R = np.sqrt(np.linalg.norm(R["ov"]) ** 2 + np.linalg.norm(R["oovv"]) ** 2)
        print(f"{i:3d}    {Etot_w:+.12f}    {Ecorr_w:+.12f}    {norm_R:e}")
        if norm_R < conv_normr:
            break

    return Etot_w, T 

def ccsd_energy(t1, t2, f, g, o, v):
    """
    < 0 | e(-T) H e(T) | 0> :
    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    #	  1.0000 f(i,i)
    energy = 1.0 * contract('ii', f[o, o])

    #	  1.0000 f(i,a)*t1(a,i)
    energy += 1.0 * contract('ia,ai', f[o, v], t1)

    #	 -0.5000 <j,i||j,i>
    energy += -0.5 * contract('jiji', g[o, o, o, o])

    #	  0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy += 0.25 * contract('jiab,abji', g[o, o, v, v], t2)

    #	 -0.5000 <j,i||a,b>*t1(a,i)*t1(b,j)
    energy += -0.5 * contract('jiab,ai,bj', g[o, o, v, v], t1, t1)

    return energy

def singles_residual(t1, t2, f, g, o, v):
    """
    < 0 | m* e e(-T) H e(T) | 0>
    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    #	  1.0000 f(e,m)
    singles_res = 1.0 * contract('em->em', f[v, o])

    #	 -1.0000 f(i,m)*t1(e,i)
    singles_res += -1.0 * contract('im,ei->em', f[o, o], t1)

    #	  1.0000 f(e,a)*t1(a,m)
    singles_res += 1.0 * contract('ea,am->em', f[v, v], t1)

    #	 -1.0000 f(i,a)*t2(a,e,m,i)
    singles_res += -1.0 * contract('ia,aemi->em', f[o, v], t2)

    #	 -1.0000 f(i,a)*t1(a,m)*t1(e,i)
    singles_res += -1.0 * contract('ia,am,ei->em', f[o, v], t1, t1)

    #	  1.0000 <i,e||a,m>*t1(a,i)
    singles_res += 1.0 * contract('ieam,ai->em', g[o, v, v, o], t1)

    #	 -0.5000 <j,i||a,m>*t2(a,e,j,i)
    singles_res += -0.5 * contract('jiam,aeji->em', g[o, o, v, o], t2)

    #	 -0.5000 <i,e||a,b>*t2(a,b,m,i)
    singles_res += -0.5 * contract('ieab,abmi->em', g[o, v, v, v], t2)

    #	  1.0000 <j,i||a,m>*t1(a,i)*t1(e,j)
    singles_res += 1.0 * contract('jiam,ai,ej->em', g[o, o, v, o], t1, t1)

    #	  1.0000 <i,e||a,b>*t1(a,i)*t1(b,m)
    singles_res += 1.0 * contract('ieab,ai,bm->em', g[o, v, v, v], t1, t1)

    #	  1.0000 <j,i||a,b>*t1(a,i)*t2(b,e,m,j)
    singles_res += 1.0 * contract('jiab,ai,bemj->em', g[o, o, v, v], t1, t2)

    #	  0.5000 <j,i||a,b>*t1(a,m)*t2(b,e,j,i)
    singles_res += 0.5 * contract('jiab,am,beji->em', g[o, o, v, v], t1, t2)

    #	  0.5000 <j,i||a,b>*t1(e,i)*t2(a,b,m,j)
    singles_res += 0.5 * contract('jiab,ei,abmj->em', g[o, o, v, v], t1, t2)

    #	  1.0000 <j,i||a,b>*t1(a,i)*t1(b,m)*t1(e,j)
    singles_res += 1.0 * contract('jiab,ai,bm,ej->em', g[o, o, v, v], t1, t1, t1)
    return singles_res

def doubles_residual(t1, t2, f, g, o, v):
    """
     < 0 | m* n* f e e(-T) H e(T) | 0>

    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    #	 -1.0000 P(m,n)f(i,n)*t2(e,f,m,i)
    contracted_intermediate = -1.0 * contract('in,efmi->efmn', f[o, o], t2)
    doubles_res = 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate)

    #	  1.0000 P(e,f)f(e,a)*t2(a,f,m,n)
    contracted_intermediate = 1.0 * contract('ea,afmn->efmn', f[v, v], t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->femn', contracted_intermediate)

    #	 -1.0000 P(m,n)f(i,a)*t1(a,n)*t2(e,f,m,i)
    contracted_intermediate = -1.0 * contract('ia,an,efmi->efmn', f[o, v], t1, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate)

    #	 -1.0000 P(e,f)f(i,a)*t1(e,i)*t2(a,f,m,n)
    contracted_intermediate = -1.0 * contract('ia,ei,afmn->efmn', f[o, v], t1, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->femn', contracted_intermediate)

    #	  1.0000 <e,f||m,n>
    doubles_res += 1.0 * contract('efmn->efmn', g[v, v, o, o])

    #	  1.0000 P(e,f)<i,e||m,n>*t1(f,i)
    contracted_intermediate = 1.0 * contract('iemn,fi->efmn', g[o, v, o, o], t1)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->femn', contracted_intermediate)

    #	  1.0000 P(m,n)<e,f||a,n>*t1(a,m)
    contracted_intermediate = 1.0 * contract('efan,am->efmn', g[v, v, v, o], t1)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate)

    #	  0.5000 <j,i||m,n>*t2(e,f,j,i)
    doubles_res += 0.5 * contract('jimn,efji->efmn', g[o, o, o, o], t2)

    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t2(a,f,m,i)
    contracted_intermediate = 1.0 * contract('iean,afmi->efmn', g[o, v, v, o], t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate) + -1.00000 * contract('efmn->femn',
                                                                   contracted_intermediate) + 1.00000 * contract(
        'efmn->fenm', contracted_intermediate)

    #	  0.5000 <e,f||a,b>*t2(a,b,m,n)
    doubles_res += 0.5 * contract('efab,abmn->efmn', g[v, v, v, v], t2)

    #	 -1.0000 <j,i||m,n>*t1(e,i)*t1(f,j)
    doubles_res += -1.0 * contract('jimn,ei,fj->efmn', g[o, o, o, o], t1, t1)

    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t1(a,m)*t1(f,i)
    contracted_intermediate = 1.0 * contract('iean,am,fi->efmn', g[o, v, v, o],
                                           t1, t1)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate) + -1.00000 * contract('efmn->femn',
                                                                   contracted_intermediate) + 1.00000 * contract(
        'efmn->fenm', contracted_intermediate)

    #	 -1.0000 <e,f||a,b>*t1(a,n)*t1(b,m)
    doubles_res += -1.0 * contract('efab,an,bm->efmn', g[v, v, v, v], t1, t1)

    #	  1.0000 P(m,n)<j,i||a,n>*t1(a,i)*t2(e,f,m,j)
    contracted_intermediate = 1.0 * contract('jian,ai,efmj->efmn', g[o, o, v, o],
                                           t1, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate)

    #	  0.5000 P(m,n)<j,i||a,n>*t1(a,m)*t2(e,f,j,i)
    contracted_intermediate = 0.5 * contract('jian,am,efji->efmn', g[o, o, v, o],
                                           t1, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate)

    #	 -1.0000 P(m,n)*P(e,f)<j,i||a,n>*t1(e,i)*t2(a,f,m,j)
    contracted_intermediate = -1.0 * contract('jian,ei,afmj->efmn', g[o, o, v, o],
                                            t1, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate) + -1.00000 * contract('efmn->femn',
                                                                   contracted_intermediate) + 1.00000 * contract(
        'efmn->fenm', contracted_intermediate)

    #	  1.0000 P(e,f)<i,e||a,b>*t1(a,i)*t2(b,f,m,n)
    contracted_intermediate = 1.0 * contract('ieab,ai,bfmn->efmn', g[o, v, v, v],
                                           t1, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->femn', contracted_intermediate)

    #	 -1.0000 P(m,n)*P(e,f)<i,e||a,b>*t1(a,n)*t2(b,f,m,i)
    contracted_intermediate = -1.0 * contract('ieab,an,bfmi->efmn', g[o, v, v, v],
                                            t1, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate) + -1.00000 * contract('efmn->femn',
                                                                   contracted_intermediate) + 1.00000 * contract(
        'efmn->fenm', contracted_intermediate)

    #	  0.5000 P(e,f)<i,e||a,b>*t1(f,i)*t2(a,b,m,n)
    contracted_intermediate = 0.5 * contract('ieab,fi,abmn->efmn', g[o, v, v, v],
                                           t1, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->femn', contracted_intermediate)

    #	 -0.5000 P(m,n)<j,i||a,b>*t2(a,b,n,i)*t2(e,f,m,j)
    contracted_intermediate = -0.5 * contract('jiab,abni,efmj->efmn',
                                            g[o, o, v, v], t2, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate)

    #	  0.2500 <j,i||a,b>*t2(a,b,m,n)*t2(e,f,j,i)
    doubles_res += 0.25 * contract('jiab,abmn,efji->efmn', g[o, o, v, v], t2, t2)

    #	 -0.5000 <j,i||a,b>*t2(a,e,j,i)*t2(b,f,m,n)
    doubles_res += -0.5 * contract('jiab,aeji,bfmn->efmn', g[o, o, v, v], t2, t2)

    #	  1.0000 P(m,n)<j,i||a,b>*t2(a,e,n,i)*t2(b,f,m,j)
    contracted_intermediate = 1.0 * contract('jiab,aeni,bfmj->efmn',
                                           g[o, o, v, v], t2, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate)

    #	 -0.5000 <j,i||a,b>*t2(a,e,m,n)*t2(b,f,j,i)
    doubles_res += -0.5 * contract('jiab,aemn,bfji->efmn', g[o, o, v, v], t2, t2)

    #	 -1.0000 P(m,n)<j,i||a,n>*t1(a,m)*t1(e,i)*t1(f,j)
    contracted_intermediate = -1.0 * contract('jian,am,ei,fj->efmn',
                                            g[o, o, v, o], t1, t1, t1)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate)

    #	 -1.0000 P(e,f)<i,e||a,b>*t1(a,n)*t1(b,m)*t1(f,i)
    contracted_intermediate = -1.0 * contract('ieab,an,bm,fi->efmn',
                                            g[o, v, v, v], t1, t1, t1)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->femn', contracted_intermediate)

    #	  1.0000 P(m,n)<j,i||a,b>*t1(a,i)*t1(b,n)*t2(e,f,m,j)
    contracted_intermediate = 1.0 * contract('jiab,ai,bn,efmj->efmn',
                                           g[o, o, v, v], t1, t1, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate)

    #	  1.0000 P(e,f)<j,i||a,b>*t1(a,i)*t1(e,j)*t2(b,f,m,n)
    contracted_intermediate = 1.0 * contract('jiab,ai,ej,bfmn->efmn',
                                           g[o, o, v, v], t1, t1, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->femn', contracted_intermediate)

    #	 -0.5000 <j,i||a,b>*t1(a,n)*t1(b,m)*t2(e,f,j,i)
    doubles_res += -0.5 * contract('jiab,an,bm,efji->efmn', g[o, o, v, v], t1, t1,
                                 t2)

    #	  1.0000 P(m,n)*P(e,f)<j,i||a,b>*t1(a,n)*t1(e,i)*t2(b,f,m,j)
    contracted_intermediate = 1.0 * contract('jiab,an,ei,bfmj->efmn',
                                           g[o, o, v, v], t1, t1, t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * contract(
        'efmn->efnm', contracted_intermediate) + -1.00000 * contract('efmn->femn',
                                                                   contracted_intermediate) + 1.00000 * contract(
        'efmn->fenm', contracted_intermediate)

    #	 -0.5000 <j,i||a,b>*t1(e,i)*t1(f,j)*t2(a,b,m,n)
    doubles_res += -0.5 * contract('jiab,ei,fj,abmn->efmn', g[o, o, v, v], t1, t1,
                                 t2)

    #	  1.0000 <j,i||a,b>*t1(a,n)*t1(b,m)*t1(e,i)*t1(f,j)
    doubles_res += 1.0 * contract('jiab,an,bm,ei,fj->efmn', g[o, o, v, v], t1, t1,
                                t1, t1)

    return doubles_res

def kernel(t1, t2, fock, g, o, v, e_ai, e_abij, nuclear_repulsion, hf_energy, max_iter=150, stopping_eps=1.0E-9, diis_size=None, diis_start_cycle=4):
    if diis_size is not None:
        print(f"\nRunning CCSD (slow) with DIIS (diis_size={diis_size}, diis_start_cycle={diis_start_cycle}) ")
        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        t1_dim = t1.size
        old_vec = np.hstack((t1.flatten(), t2.flatten()))
    else:
        print("\nRunning CCSD (slow) without DIIS")

    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    old_energy = ccsd_energy(t1, t2, fock, g, o, v) + nuclear_repulsion
    header = "Iter.     Energy [Eh]    Corr. energy [Eh]       Delta_E   "
    print(header)
    for idx in range(max_iter):

        singles_res = singles_residual(t1, t2, fock, g, o, v) + fock_e_ai * t1
        doubles_res = doubles_residual(t1, t2, fock, g, o, v) + fock_e_abij * t2

        new_singles = singles_res * e_ai
        new_doubles = doubles_res * e_abij

        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack(
                (new_singles.flatten(), new_doubles.flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                 error_vec)
            new_singles = new_vectorized_iterate[:t1_dim].reshape(t1.shape)
            new_doubles = new_vectorized_iterate[t1_dim:].reshape(t2.shape)
            old_vec = new_vectorized_iterate

        current_energy = ccsd_energy(new_singles, new_doubles, fock, g, o, v) + nuclear_repulsion
        delta_e = np.abs(old_energy - current_energy)
        corr_energy = current_energy - hf_energy

        if delta_e < stopping_eps:
            return new_singles, new_doubles
        else:
            t1 = new_singles
            t2 = new_doubles
            old_energy = current_energy
            print(f"{idx:3d}    {old_energy:+.12f}    {corr_energy:+.12f}    {delta_e:e}")
    else:
        print("Did not converge")
        return new_singles, new_doubles

def ccsd_main(system, ham, diis_size, diis_start_cycle, optimized):
    if ham.n_act is None:
        n_act = ham.n_orb
    else:
        n_act = ham.n_act
        if ham._f.shape[0] != 2*n_act:
            ham.extract_local_hamiltonian()

    nocc = ham.n_a + ham.n_b
    nvirt = 2 * n_act - nocc
    
    start = time.perf_counter()
    if optimized:
        F = one_body_mat2dic(ham._f,nocc,n_act,n_act)
        V = two_body_ten2dic(4*ham._v,nocc,n_act,n_act)
        invD = compute_inverse_denominators(F, nocc, nvirt, 2)
        T = {"ov": np.zeros((nocc, nvirt)), "oovv": np.zeros((nocc, nocc, nvirt, nvirt))}
        Etot_w, T = kernel_w(F, V, T, invD, nocc, nvirt, hf_energy=system.meanfield.e_tot, diis_size=diis_size, diis_start_cycle=diis_start_cycle)

    else:
        fock = ham._f 
        g = 4*ham._v   
        eps = np.diag(ham._f)

        n = np.newaxis
        o = slice(None, nocc)
        v = slice(nocc, None)
        e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
            n, n, n, o])
        e_ai = 1 / (-eps[v, n] + eps[n, o])

        t1f, t2f = kernel(np.zeros((nvirt, nocc)), np.zeros((nvirt, nvirt, nocc, nocc)), fock, g, o, v, e_ai, e_abij, system.nuclear_repulsion, hf_energy=system.meanfield.e_tot,
                        diis_size=diis_size, diis_start_cycle=diis_start_cycle)
        
        Etot_w = ccsd_energy(t1f, t2f, fock, g, o, v) + system.nuclear_repulsion
    
    dt = time.perf_counter() - start
    print("")
    m, s = divmod(dt, 60)
    print("CCSD wall time                                 :%8.2f m  %3.2f s" % (m, s))

    return (Etot_w)

