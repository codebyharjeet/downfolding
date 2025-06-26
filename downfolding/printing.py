import datetime
from typing import Literal

def get_timestamp():
    return datetime.datetime.strptime(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")

def tprint(tens,thresh=1e-15):
	"""Print indices and value of non-zero tensor elements

	Parameters
	----------
	tens : np.ndarray
		tensor to be printed
	thresh : float
		threshold for element printing

	Returns
	-------
	NULL
	"""
	if(np.ndim(tens) == 0):
		print(tens) 
	elif(np.ndim(tens) == 1):
		for i in range(0,len(tens)):
			if(abs(tens[i]) > thresh):
				print("[%d] : %e"%(i,tens[i]))
	elif(np.ndim(tens) == 2):
		for i in range(0,tens.shape[0]):
			for j in range(0,tens.shape[1]):
				if(abs(tens[i,j]) > thresh):
					print("[%d,%d] : %e"%(i,j,tens[i,j]))
	elif(np.ndim(tens) == 4): 
		for i in range(0,tens.shape[0]):
			for j in range(0,tens.shape[1]):
				for k in range(0,tens.shape[2]):
					for l in range(0,tens.shape[3]):
						if(abs(tens[i,j,k,l]) > thresh):
							print("[%d,%d,%d,%d] : %e"%(i,j,k,l,tens[i,j,k,l]))
	elif(np.ndim(tens) == 6): 
		for i in range(0,tens.shape[0]):
			for j in range(0,tens.shape[1]):
				for k in range(0,tens.shape[2]):
					for l in range(0,tens.shape[3]):
						for m in range(0,tens.shape[4]):
							for n in range(0,tens.shape[5]):
								if(abs(tens[i,j,k,l,m,n]) > thresh):
									print("[%d,%d,%d,%d,%d,%d] : %e"%(i,j,k,l,m,n,tens[i,j,k,l,m,n]))
	elif(np.ndim(tens) == 8):
		for i in range(0,tens.shape[0]):
			for j in range(0,tens.shape[1]):
				for k in range(0,tens.shape[2]):
					for l in range(0,tens.shape[3]):
						for m in range(0,tens.shape[4]):
							for n in range(0,tens.shape[5]):
								for o in range(0,tens.shape[6]):
									for p in range(0,tens.shape[7]):
										if(abs(tens[i,j,k,l,m,n,o,p]) > thresh):
											print("[%d,%d,%d,%d,%d,%d,%d,%d] : %e"%(i,j,k,l,m,n,o,p,tens[i,j,k,l,m,n,o,p]))
	else:
		print("TODO: implement a printing for a %dD tensor."%(np.ndim(tens)))
		exit()

def ccsd_summary(ccsd_tot, ccsd_corr):
    print("\n   CCSD Calculation Summary")
    print("   -------------------------------------")
    print(f"CCSD Total Energy                              :%18.12f"%(ccsd_tot))
    print(f"CCSD Correlation Energy                        :%18.12f"%(ccsd_corr))    

def ducc_summary(energy: float, backend: Literal["pyscf", "openfermion"]) -> None:
    """
    Print a formatted DUCC Full-CI energy.

    Parameters
    ----------
    energy : float
        The computed ground-state energy.
    backend : {"pyscf","openfermion"}
        Which backend computed energy.
    """
    labels = {
        "pyscf":       "DUCC Full CI (PySCF)",
        "openfermion": "DUCC Full CI (OpenFermion)",
    }
    label = labels.get(backend, backend)
    print(f"{label:46s} : {energy:17.12f}")