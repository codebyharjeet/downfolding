import time
import numpy as np

from downfolding.hamiltonian import HamFormat


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

    n_a, n_b = H.n_a, H.n_b
    E_nuc    = system.nuclear_repulsion

    # constant, h, g = H(HamFormat.SPATORB_PV)
    constant, h, g = H.get_MO_integrals()
    print(f"\nNuclear Repulsion Energy: {E_nuc:.10f}")
    print(f"\nConstant Energy: {constant:.10f}")

    e1 = 0
    e2 = 0

    # Extract the main diagonal of h to compute e1
    h_diag = np.diag(h)
    e1 = np.sum(h_diag[:n_a]) + np.sum(h_diag[:n_b])

    # Pre-slice the g tensor to avoid redundant slicing overhead
    g_aa = g[:n_a, :n_a, :n_a, :n_a]
    g_bb = g[:n_b, :n_b, :n_b, :n_b]
    g_ab = g[:n_a, :n_a, :n_b, :n_b]

    # Use einsum to compute the 2-electron integrals
    e2_aa = 0.5 * (np.einsum('iijj->', g_aa) - np.einsum('ijji->', g_aa))
    e2_bb = 0.5 * (np.einsum('iijj->', g_bb) - np.einsum('ijji->', g_bb))
    e2_ab = np.einsum('iijj->', g_ab)

    e2 = e2_aa + e2_bb + e2_ab
    E_tot = e1 + e2 + constant

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


    e1 = 0
    e2 = 0
    config_a = range(n_a)
    config_b = range(n_b)
    
    for i in config_a:
        e1 += h[i,i]
    for i in config_b:
        e1 += h[i,i]
    for i in config_a:
        for j in config_a:
            if i>=j:
                continue
            e2 += g[i,i,j,j]
            e2 -= g[i,j,j,i]
    for i in config_b:             
        for j in config_b:         
            if i>=j:                
                continue           
            e2 += g[i,i,j,j]
            e2 -= g[i,j,j,i]
    for i in config_a:             
        for j in config_b:         
            e2 += g[i,i,j,j]
    
    E_tot = e1 + e2 + E_nuc 

"""