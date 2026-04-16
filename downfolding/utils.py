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

def check_8fold_symmetry(h2, tol=1e-10):
    """
    Checks if a 2-electron tensor in chemist's notation (pq|rs) 
    obeys strict 8-fold spatial symmetry.
    """
    # 1. Check p <-> q exchange: (pq|rs) == (qp|rs)
    sym_pq = np.allclose(h2, np.transpose(h2, (1, 0, 2, 3)), atol=tol)
    
    # 2. Check r <-> s exchange: (pq|rs) == (pq|sr)
    sym_rs = np.allclose(h2, np.transpose(h2, (0, 1, 3, 2)), atol=tol)
    
    # 3. Check (pq) <-> (rs) pair exchange: (pq|rs) == (rs|pq)
    sym_pq_rs = np.allclose(h2, np.transpose(h2, (2, 3, 0, 1)), atol=tol)
    
    print("--- Symmetry Check Results ---")
    print(f"(pq|rs) = (qp|rs) [p<->q] : {sym_pq}")
    print(f"(pq|rs) = (pq|sr) [r<->s] : {sym_rs}")
    print(f"(pq|rs) = (rs|pq) [pairs] : {sym_pq_rs}")
    print("------------------------------")
    
    if sym_pq and sym_rs and sym_pq_rs:
        print("Conclusion: Tensor has FULL 8-fold symmetry.")
        print("Action: Safe to use fci.direct_spin0.FCISolver().")
    else:
        print("Conclusion: Tensor LACKS 8-fold symmetry.")
        print("Action: MUST use fci.direct_nosym.FCISolver().")


        