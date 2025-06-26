import time
import numpy as np


def calc_hf(system, H):
    """
    Hartree-Fock closed-shell energy (restricted spin-orbitals).

    Parameters
    ----------
    system : object
           n_a, n_b        - number of alpha / beta electrons
           nuclear_repulsion - nuclear repulsion energy (float)
    H : object
           _h  - one-electron integrals, shape (n_so, n_so)
           _v  - two-electron integrals, shape (n_so, n_so, n_so, n_so)

    Returns
    -------
    float
        Total Hartree-Fock energy E = E_one + E_two + E_nuc
    """
    t0 = time.perf_counter()

    n_a, n_b = system.n_a, system.n_b
    E_nuc    = system.nuclear_repulsion
    f        = H._f       
    v        = H._v        

    idx_a = 2 * np.arange(n_a)        # alpha: 0,2,4,…
    idx_b = 2 * np.arange(n_b) + 1    # beta: 1,3,5,…

    e1 = f[idx_a, idx_a].sum() + f[idx_b, idx_b].sum()

    ia, ja = np.triu_indices(n_a, k=1)   
    ib, jb = np.triu_indices(n_b, k=1)   

    J_aa = v[idx_a[ia], idx_a[ia], idx_a[ja], idx_a[ja]]
    K_aa = v[idx_a[ia], idx_a[ja], idx_a[ja], idx_a[ia]]

    J_bb = v[idx_b[ib], idx_b[ib], idx_b[jb], idx_b[jb]]
    K_bb = v[idx_b[ib], idx_b[jb], idx_b[jb], idx_b[ib]]

    # idx_a[:, None] → shape (n_a, 1) so broadcasting builds (n_a, n_b)
    J_ab = v[idx_a[:, None], idx_a[:, None], idx_b, idx_b].sum()

    e2 = (J_aa - K_aa).sum() + (J_bb - K_bb).sum() + J_ab

    E_tot = e1 + e2 + E_nuc

    dt = time.perf_counter() - t0
    m, s = divmod(dt, 60)

    print("\n   HF Calculation Summary")
    print("   -------------------------------------")
    print(f"   Total wall time: {m:0.2f} m  {s:0.2f} s")
    print(f"   One-body  = {e1:16.10f}")
    print(f"   Two-body  = {e2:16.10f}")
    print(f"   HF Energy = {E_tot:16.10f}")

    return E_tot


"""
e1 = 0
e2 = 0
for i in range(n_a):
    e1 += h[2*i,2*i]
for i in range(n_b):
    e1 += h[2*i+1,2*i+1]
for i in range(n_a):
    for j in range(n_a):
        if i>=j:
            continue
        e2 += v[2*i,2*i,2*j,2*j]
        e2 -= v[2*i,2*j,2*j,2*i]
for i in range(n_b):             
    for j in range(n_b):         
        if i>=j:                
            continue           
        e2 += v[2*i+1,2*i+1,2*j+1,2*j+1]
        e2 -= v[2*i+1,2*j+1,2*j+1,2*i+1]
for i in range(n_a):             
    for j in range(n_b):         
        e2 += v[2*i,2*i,2*j+1,2*j+1]
"""