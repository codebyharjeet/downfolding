import numpy as np
# import torch
import opt_einsum


class cc_contract(object):
    """
    A wrapper for opt_einsum.contract with tensors stored on CPU and/or GPU.
    """
    def __init__(self, device='CPU'):
        """
        Parameters
        ----------
        device: string
            initiated in ccwfn object, default: 'CPU'
        
        Returns
        -------
        None
        """
        self.device = device
        if self.device == 'GPU':
            # torch.device is an object representing the device on which torch.Tensor is or will be allocated.
            self.device0 = torch.device('cpu')
            self.device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    def __call__(self, subscripts, *operands): 
        """
        Parameters
        ----------
        subscripts: string
            specify the subscripts for summation (same format as numpy.einsum)
        *operands: list of array_like
            the arrays/tensors for the operation
   
        Returns
        -------
        An ndarray/torch.Tensor that is calculated based on Einstein summation convention.   
        """       
        if self.device == 'CPU':
            return opt_einsum.contract(subscripts, *operands)
        elif self.device == 'GPU':
            # Check the type and allocation of the tensors 
            # Transfer the copy from CPU to GPU if needed (for ERI)
            input_list = list(operands)
            for i in range(len(input_list)):
                if (not input_list[i].is_cuda):
                    input_list[i] = input_list[i].to(self.device1)               
            #print(len(input_list), type(input_list[0]), type(input_list[1]))    
            output = opt_einsum.contract(subscripts, *input_list)
            del input_list
            return output


def get_memory_usage():
    """Displays the percentage of used RAM and available memory. Useful for
    investigating the memory usages of various routines.
    Usage:
    >> # gives a single float value
    >> psutil.cpu_percent()
    >> # gives an object with many fields
    >> psutil.virtual_memory()
    >> # you can convert that object to a dictionary
    >> dict(psutil.virtual_memory()._asdict())
    >> # you can have the percentage of used RAM
    >> psutil.virtual_memory().percent
    >> 79.2
    >> # you can calculate percentage of available memory
    >> psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    >> 20.
    print(eri_so.shape, " %12.8f Mb" %(eri_so.nbytes*1e-6))
    """
    import psutil
    current_process = psutil.Process(os.getpid())
    memory = current_process.memory_info().rss # RSS (e.g., RAM usage) memory in bytes
    return memory / (1024 * 1024)            


def analyze_tensor_physics(h2, tol=1e-10):
    """
    Analyzes the physical and mathematical symmetries of a 2-electron 
    tensor in chemist's notation (pq|rs).
    """
    # 1. Core Physical Checks
    # Hermiticity: (pq|rs) = (qp|sr)* -> For real numbers, (pq|rs) = (qp|sr)
    is_hermitian = np.allclose(h2, np.transpose(h2, (1, 0, 3, 2)), atol=tol)
    
    # Particle Indistinguishability: Electron 1 <-> Electron 2
    is_indistinguishable = np.allclose(h2, np.transpose(h2, (2, 3, 0, 1)), atol=tol)
    
    # Bare Spatial Symmetry: p <-> q (Broken by DUCC downfolding)
    has_spatial_sym = np.allclose(h2, np.transpose(h2, (1, 0, 2, 3)), atol=tol)

    print("--- Core Physical Properties ---")
    print(f"Hermitian (pq|rs == qp|sr)              : {is_hermitian}")
    print(f"Particle Indistinguishable (pq|rs == rs|pq): {is_indistinguishable}")
    print(f"Intra-pair Spatial Sym (pq|rs == qp|rs) : {has_spatial_sym}")
    print("--------------------------------\n")

    # 2. Mathematical Fold Count
    permutations = {
        "(pq|rs) [Original]":         (0, 1, 2, 3),
        "(qp|rs) [p<->q]":            (1, 0, 2, 3),
        "(pq|sr) [r<->s]":            (0, 1, 3, 2),
        "(qp|sr) [Both intra]":       (1, 0, 3, 2), # Hermiticity
        "(rs|pq) [Inter-pair]":       (2, 3, 0, 1), # Indistinguishability
        "(rs|qp) [Inter + p<->q]":    (2, 3, 1, 0),
        "(sr|pq) [Inter + r<->s]":    (3, 2, 0, 1),
        "(sr|qp) [Inter + both]":     (3, 2, 1, 0)
    }
    
    symmetry_fold = 0
    
    print("--- 8-Fold Permutation Breakdown ---")
    for name, axes in permutations.items():
        is_sym = np.allclose(h2, np.transpose(h2, axes), atol=tol)
        print(f"{name:<25}: {is_sym}")
        if is_sym:
            symmetry_fold += 1
            
    print("------------------------------------")
    print(f"Conclusion: Tensor has {symmetry_fold}-fold symmetry.")
    
    # Actionable advice
    if symmetry_fold == 8:
        print("Action: Standard Hamiltonian. Safe to use fci.direct_spin0.FCISolver().")
    elif symmetry_fold == 4 and is_hermitian:
        print("Action: Effective Hermitian operator. MUST use fci.direct_nosym.FCISolver().")
    else:
        print("Action: Asymmetric/Non-Hermitian operator. MUST use fci.direct_nosym.FCISolver().")
        
    return symmetry_fold, is_hermitian

        