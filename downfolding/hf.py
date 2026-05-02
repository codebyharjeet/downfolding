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

"""


def compute_hf_energy_mo(H_mo, G_mo, n_occ):
    """
    Computes the restricted Hartree-Fock energy from integrals in the MO basis.
    
    Parameters:
    H_mo (ndarray): 1-electron integrals in MO basis (n_mo, n_mo)
    G_mo (ndarray): 2-electron integrals in MO basis, chemist's notation (n_mo, n_mo, n_mo, n_mo)
    n_occ (int): Number of occupied spatial orbitals (N_electrons // 2)
    """
    # 1. One-Electron Energy: 2 * sum(H_ii)
    # We take the trace of the occupied block and multiply by 2 for spin-up and spin-down
    e_1e = 2.0 * np.trace(H_mo[:n_occ, :n_occ])
    
    # 2. Slice the 2-electron integrals to only include the occupied block
    G_occ = G_mo[:n_occ, :n_occ, :n_occ, :n_occ]
    
    # 3. Two-Electron Energy
    # Calculate the Coulomb (J) and Exchange (K) sums over occupied orbitals
    # J_sum = sum_{ij} (ii|jj)
    # K_sum = sum_{ij} (ij|ji)
    J_sum = np.einsum('iijj->', G_occ)
    K_sum = np.einsum('ijji->', G_occ)
    
    e_2e = 2.0 * J_sum - K_sum
    
    # 4. Total Energy
    e_total = e_1e + e_2e
    
    return e_total