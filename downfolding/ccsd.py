import time
import numpy as np


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
    ECCSD += np.einsum('ia,ia->', fmat[occ_alpha, virt_alpha], t1_amps[0])
    ECCSD += np.einsum('ia,ia->', fmat[occ_beta, virt_beta], t1_amps[1])

    # Doubles (t2) contributions
    ECCSD += 0.25 * np.einsum('ijab,ijab->', t2_amps[0], vten[occ_alpha, occ_alpha, virt_alpha, virt_alpha])
    ECCSD += 0.25 * np.einsum('ijab,ijab->', t2_amps[2], vten[occ_beta, occ_beta, virt_beta, virt_beta])
    ECCSD += 1.0 * np.einsum('ijab,ijab->', t2_amps[1], vten[occ_alpha, occ_beta, virt_alpha, virt_beta])

    # Mixed singles (t1) contributions
    ECCSD += 0.5 * np.einsum('ia,jb,ijab->', t1_amps[0], t1_amps[0], vten[occ_alpha, occ_alpha, virt_alpha, virt_alpha])
    ECCSD += 0.5 * np.einsum('ia,jb,ijab->', t1_amps[1], t1_amps[1], vten[occ_beta, occ_beta, virt_beta, virt_beta])
    ECCSD += 1.0 * np.einsum('ia,jb,ijab->', t1_amps[0], t1_amps[1], vten[occ_alpha, occ_beta, virt_alpha, virt_beta])

    dt = time.perf_counter() - t0
    m, s = divmod(dt, 60)

    if verbose != 0:
        print("\n   CCSD Calculation Summary")
        print("   -------------------------------------")
        print(f"   Total wall time: {m:0.2f} m  {s:0.2f} s")
        print(f"   CCSD Correction = {ECCSD:16.10f}")

    return ECCSD

