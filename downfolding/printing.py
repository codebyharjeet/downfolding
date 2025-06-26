import datetime

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

# def ducc_summary():
#     print("\n   DUCC Calculation Summary")
#     print("   -------------------------------------")
#     print("Size of the active space                       :%10i" %(n_act))
#     print(f"CCSD Total Energy                              :%21.12f"%(mccsd.e_tot))
#     print(f"CCSD Correlation Energy                        :%21.12f"%(ccsd_energy))  