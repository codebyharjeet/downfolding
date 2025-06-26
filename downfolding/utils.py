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
    """
    import psutil
    current_process = psutil.Process(os.getpid())
    memory = current_process.memory_info().rss # RSS (e.g., RAM usage) memory in bytes
    return memory / (1024 * 1024)            