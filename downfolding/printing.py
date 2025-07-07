from datetime import datetime 
from typing import Literal
import subprocess
import platform
import psutil

def get_timestamp():
    width = 60
    border = "=" * width
    label  = "Downfolding Package Initialized"
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_line = f"Timestamp: {ts}"

    print("\n" + border)
    print(f"{label:^{width}}")
    print(f"{timestamp_line:^{width}}")
    print(border + "\n")

def device_information():
    """
    Detects and prints the CPU model and total physical memory (GiB)
    on macOS, Linux, or within an MPI/Slurm environment.
    """
    system = platform.system()
    if system == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL
            )
            cpu = out.decode().strip()
        except Exception:
            cpu = platform.processor() or platform.machine()
    elif system == "Linux":
        try:
            # Attempt to parse lscpu output for model name
            out = subprocess.check_output(["lscpu"], universal_newlines=True)
            for line in out.splitlines():
                if "Model name:" in line:
                    cpu = line.split(":", 1)[1].strip()
                    break
            else:
                cpu = platform.processor() or platform.machine()
        except Exception:
            cpu = platform.processor() or platform.machine()
    else:
        cpu = platform.processor() or platform.machine()

    mem_bytes = psutil.virtual_memory().total
    mem_gib = round(mem_bytes / (1024 ** 3), 2)

    print("\n   Device Information")
    print("   -------------------------------------")
    print(f"CPU 					       :%15s"  %(cpu))   
    print(f"Available memory                               :%11.2f GB"%(mem_gib))

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
    print(f"CCSD Correlation Energy                        :%18.12f"%(ccsd_corr))   
    print(f"CCSD Total Energy                              :%18.12f"%(ccsd_tot))
 

def ccsd_t_summary(ccsd_tot, ccsd_t_corr):
	print("\n   CCSD(T) Calculation Summary")
	print("   -------------------------------------")
	print(f"CCSD Total Energy                              :%18.12f"%(ccsd_tot))
	print(f"CCSD(T) Correlation Energy                     :%18.12f"%(ccsd_t_corr))
	print(f"CCSD(T) Total Energy                           :%18.12f"%(ccsd_tot+ccsd_t_corr))  

def ducc_summary(energy: float, backend: Literal["pyscf", "openfermion"], hf_energy: float) -> None:
	"""
	Print a formatted DUCC Full-CI energy.

	Parameters
	----------
	energy : float
		The computed ground-state energy.
	backend : {"pyscf","openfermion"}
		Which backend computed energy.
	"""
	print(f"\n   Exact Diagonalization Summary (using {backend})")
	print("   -------------------------------------")
	print(f"DUCC Full CI Correlation Energy                :%18.12f"%(energy))  
	print(f"DUCC Full CI Total Energy                      :%18.12f"%(energy+hf_energy))  
