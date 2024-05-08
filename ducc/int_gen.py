import scipy
import numpy as np




def rand_1_body(n_orb):
	test_1_body = np.zeros((2*n_orb,2*n_orb))
	for p in range(0,n_orb):
		pa = 2*p 
		pb = 2*p+1 
		for q in range(p,n_orb):
			qa = 2*q 
			qb = 2*q+1 
			aa = 2*np.random.rand()-1
			bb = 2*np.random.rand()-1 
			test_1_body[pa,qa] = aa
			test_1_body[qa,pa] = aa
			test_1_body[pb,qb] = bb
			test_1_body[qb,pb] = bb 
	return test_1_body

def rand_1_body_r(n_orb):
	test_1_body = np.zeros((2*n_orb,2*n_orb))
	for p in range(0,n_orb):
		pa = 2*p 
		pb = 2*p+1 
		for q in range(p,n_orb):
			qa = 2*q 
			qb = 2*q+1 
			aa = 2*np.random.rand()-1
			test_1_body[pa,qa] = aa
			test_1_body[qa,pa] = aa
			test_1_body[pb,qb] = aa
			test_1_body[qb,pb] = aa 
	return test_1_body

def rand_2_body(n_orb):
	test_2_body = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
	for p in range(0,n_orb):
		pa = 2*p 
		pb = 2*p+1 
		for q in range(p+1,n_orb):
			qa = 2*q 
			qb = 2*q+1 
			for r in range(0,n_orb):
				ra = 2*r 
				rb = 2*r+1 
				for s in range(r+1,n_orb):
					sa = 2*s 
					sb = 2*s+1 
					aaaa = 2*np.random.rand()-1 
					bbbb = 2*np.random.rand()-1
					test_2_body[pa,qa,ra,sa] =  aaaa
					test_2_body[pa,qa,sa,ra] = -aaaa 
					test_2_body[qa,pa,ra,sa] = -aaaa
					test_2_body[qa,pa,sa,ra] =  aaaa 
					test_2_body[ra,sa,pa,qa] =  aaaa
					test_2_body[sa,ra,pa,qa] = -aaaa 
					test_2_body[ra,sa,qa,pa] = -aaaa
					test_2_body[sa,ra,qa,pa] =  aaaa

					test_2_body[pb,qb,rb,sb] =  bbbb
					test_2_body[pb,qb,sb,rb] = -bbbb 
					test_2_body[qb,pb,rb,sb] = -bbbb
					test_2_body[qb,pb,sb,rb] =  bbbb 
					test_2_body[rb,sb,pb,qb] =  bbbb
					test_2_body[sb,rb,pb,qb] = -bbbb 
					test_2_body[rb,sb,qb,pb] = -bbbb
					test_2_body[sb,rb,qb,pb] =  bbbb
	for p in range(0,n_orb):
		pa = 2*p 
		for q in range(0,n_orb):
			qb = 2*q+1 
			for r in range(0,n_orb):
				ra = 2*r 
				for s in range(0,n_orb):
					sb = 2*s+1 
					abab = 2*np.random.rand()-1
					test_2_body[pa,qb,ra,sb] =  abab 
					test_2_body[pa,qb,sb,ra] = -abab
					test_2_body[qb,pa,ra,sb] = -abab 
					test_2_body[qb,pa,sb,ra] =  abab 
					test_2_body[ra,sb,pa,qb] =  abab 
					test_2_body[ra,sb,qb,pa] = -abab 
					test_2_body[sb,ra,pa,qb] = -abab 
					test_2_body[sb,ra,qb,pa] =  abab 
	return test_2_body
comment = """
print("2-Body Tensor Tests")
test_2_body = rand_2_body(n_act)
print("Hermitian:")
tprint(test_2_body - np.einsum("pqrs->rspq",test_2_body,optimize="optimal"))
print("Antisymmetric")
print("pqrs->pqsr")
tprint(test_2_body + np.einsum("pqrs->pqsr",test_2_body,optimize="optimal"))
print("pqrs->qprs")
tprint(test_2_body + np.einsum("pqrs->qprs",test_2_body,optimize="optimal"))
print("pqrs->qpsr")
tprint(test_2_body - np.einsum("pqrs->qpsr",test_2_body,optimize="optimal"))
"""

def rand_2_body_r(n_orb):
	test_2_body = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb))
	for p in range(0,n_orb):
		pa = 2*p 
		pb = 2*p+1 
		for q in range(0,n_orb):
			qa = 2*q
			qb = 2*q+1 
			for r in range(0,n_orb):
				ra = 2*r 
				rb = 2*r+1 
				for s in range(0,n_orb):
					sa = 2*s
					sb = 2*s+1 
					# <pa,qb||ra,sb> = <p,q|r,s>
					abab = 2*np.random.rand()-1
					test_2_body[pa,qb,ra,sb] =  abab
					test_2_body[pa,qb,sb,ra] = -abab
					test_2_body[qb,pa,ra,sb] = -abab
					test_2_body[qb,pa,sb,ra] =  abab
					test_2_body[ra,sb,pa,qb] =  abab
					test_2_body[ra,sb,qb,pa] = -abab
					test_2_body[sb,ra,pa,qb] = -abab
					test_2_body[sb,ra,qb,pa] =  abab
					# <pb,qa||rb,sa> = <p,q|r,s>
					test_2_body[pb,qa,rb,sa] =  abab
					test_2_body[pb,qa,sa,rb] = -abab
					test_2_body[qa,pb,rb,sa] = -abab
					test_2_body[qa,pb,sa,rb] =  abab
					test_2_body[rb,sa,pb,qa] =  abab
					test_2_body[rb,sa,qa,pb] = -abab
					test_2_body[sa,rb,pb,qa] = -abab
					test_2_body[sa,rb,qa,pb] =  abab
	for p in range(0,n_orb):
		pa = 2*p 
		pb = 2*p+1 
		for q in range(p+1,n_orb):
			qa = 2*q 
			qb = 2*q+1 
			for r in range(0,n_orb):
				ra = 2*r 
				rb = 2*r+1 
				for s in range(r+1,n_orb):
					sa = 2*s 
					sb = 2*s+1 
					# <pa,qa||ra,sa> = <p,q|r,s> - <p,q|s,r> = <pa,qb||ra,sb> - <pa,qb||sa,rb>
					aaaa = test_2_body[pa,qb,ra,sb] - test_2_body[pa,qb,sa,rb]
					test_2_body[pa,qa,ra,sa] =  aaaa
					test_2_body[pa,qa,sa,ra] = -aaaa
					test_2_body[qa,pa,ra,sa] = -aaaa
					test_2_body[qa,pa,sa,ra] =  aaaa
					test_2_body[ra,sa,pa,qa] =  aaaa
					test_2_body[ra,sa,qa,pa] = -aaaa 
					test_2_body[sa,ra,pa,qa] = -aaaa 
					test_2_body[sa,ra,qa,pa] =  aaaa 
					# <pb,qb||rb,sb> = <p,q|r,s> - <p,q|s,r> = <pa,qb||ra,sb> - <pa,qb||sa,rb>
					test_2_body[pb,qb,rb,sb] =  aaaa
					test_2_body[pb,qb,sb,rb] = -aaaa
					test_2_body[qb,pb,rb,sb] = -aaaa
					test_2_body[qb,pb,sb,rb] =  aaaa
					test_2_body[rb,sb,pb,qb] =  aaaa
					test_2_body[rb,sb,qb,pb] = -aaaa 
					test_2_body[sb,rb,pb,qb] = -aaaa 
					test_2_body[sb,rb,qb,pb] =  aaaa 
	return test_2_body
comment = """
print("2-Body Restricted Tensor Tests")
test_2_body = rand_2_body_r(n_act)
print("Hermitian:")
tprint(test_2_body - np.einsum("pqrs->rspq",test_2_body,optimize="optimal"))
print("a<->b")
test_2_body_sf = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act))
for p in range(0,n_act):
	pa = 2*p 
	pb = 2*p+1 
	for q in range(0,n_act):
		qa = 2*q 
		qb = 2*q+1 
		for r in range(0,n_act):
			ra = 2*r 
			rb = 2*r+1 
			for s in range(0,n_act):
				sa = 2*s 
				sb = 2*s+1 
				test_2_body_sf[pb,qb,rb,sb] = test_2_body[pa,qa,ra,sa]
				test_2_body_sf[pa,qa,ra,sa] = test_2_body[pb,qb,rb,sb]
				test_2_body_sf[pa,qb,ra,sb] = test_2_body[pb,qa,rb,sa]
				test_2_body_sf[pa,qb,sb,ra] = test_2_body[pb,qa,sa,rb]
				test_2_body_sf[qb,pa,ra,sb] = test_2_body[qa,pb,rb,sa]
				test_2_body_sf[qb,pa,sb,ra] = test_2_body[qa,pb,sa,rb]
tprint(test_2_body - test_2_body_sf)
print("Antisymmetric")
print("pqrs->pqsr")
tprint(test_2_body + np.einsum("pqrs->pqsr",test_2_body,optimize="optimal"))
print("pqrs->qprs")
tprint(test_2_body + np.einsum("pqrs->qprs",test_2_body,optimize="optimal"))
print("pqrs->qpsr")
tprint(test_2_body - np.einsum("pqrs->qpsr",test_2_body,optimize="optimal"))
exit()
"""
def rand_3_body(n_orb):
	test_3_body = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb))
	for p in range(0,n_orb):
		pa = 2*p 
		for q in range(p+1,n_orb):
			qa = 2*q 
			for r in range(q+1,n_orb):
				ra = 2*r 
				for s in range(0,n_orb):
					sa = 2*s 
					for t in range(s+1,n_orb):
						ta = 2*t  
						for u in range(t+1,n_orb):
							ua = 2*u 
							aaaaaa = 2*np.random.rand()-1
							test_3_body[pa,qa,ra,sa,ta,ua] =  aaaaaa
							test_3_body[pa,ra,qa,sa,ta,ua] = -aaaaaa
							test_3_body[qa,pa,ra,sa,ta,ua] = -aaaaaa
							test_3_body[qa,ra,pa,sa,ta,ua] =  aaaaaa
							test_3_body[ra,pa,qa,sa,ta,ua] =  aaaaaa
							test_3_body[ra,qa,pa,sa,ta,ua] = -aaaaaa
							test_3_body[sa,ta,ua,pa,qa,ra] =  aaaaaa
							test_3_body[sa,ta,ua,pa,ra,qa] = -aaaaaa
							test_3_body[sa,ta,ua,qa,pa,ra] = -aaaaaa
							test_3_body[sa,ta,ua,qa,ra,pa] =  aaaaaa
							test_3_body[sa,ta,ua,ra,pa,qa] =  aaaaaa
							test_3_body[sa,ta,ua,ra,qa,pa] = -aaaaaa

							test_3_body[pa,qa,ra,sa,ua,ta] = -aaaaaa
							test_3_body[pa,ra,qa,sa,ua,ta] =  aaaaaa
							test_3_body[qa,pa,ra,sa,ua,ta] =  aaaaaa
							test_3_body[qa,ra,pa,sa,ua,ta] = -aaaaaa
							test_3_body[ra,pa,qa,sa,ua,ta] = -aaaaaa
							test_3_body[ra,qa,pa,sa,ua,ta] =  aaaaaa
							test_3_body[sa,ua,ta,pa,qa,ra] = -aaaaaa
							test_3_body[sa,ua,ta,pa,ra,qa] =  aaaaaa
							test_3_body[sa,ua,ta,qa,pa,ra] =  aaaaaa
							test_3_body[sa,ua,ta,qa,ra,pa] = -aaaaaa
							test_3_body[sa,ua,ta,ra,pa,qa] = -aaaaaa
							test_3_body[sa,ua,ta,ra,qa,pa] =  aaaaaa

							test_3_body[pa,qa,ra,ta,sa,ua] = -aaaaaa
							test_3_body[pa,ra,qa,ta,sa,ua] =  aaaaaa
							test_3_body[qa,pa,ra,ta,sa,ua] =  aaaaaa
							test_3_body[qa,ra,pa,ta,sa,ua] = -aaaaaa
							test_3_body[ra,pa,qa,ta,sa,ua] = -aaaaaa
							test_3_body[ra,qa,pa,ta,sa,ua] =  aaaaaa
							test_3_body[ta,sa,ua,pa,qa,ra] = -aaaaaa
							test_3_body[ta,sa,ua,pa,ra,qa] =  aaaaaa
							test_3_body[ta,sa,ua,qa,pa,ra] =  aaaaaa
							test_3_body[ta,sa,ua,qa,ra,pa] = -aaaaaa
							test_3_body[ta,sa,ua,ra,pa,qa] = -aaaaaa
							test_3_body[ta,sa,ua,ra,qa,pa] =  aaaaaa

							test_3_body[pa,qa,ra,ta,ua,sa] =  aaaaaa
							test_3_body[pa,ra,qa,ta,ua,sa] = -aaaaaa
							test_3_body[qa,pa,ra,ta,ua,sa] = -aaaaaa
							test_3_body[qa,ra,pa,ta,ua,sa] =  aaaaaa
							test_3_body[ra,pa,qa,ta,ua,sa] =  aaaaaa
							test_3_body[ra,qa,pa,ta,ua,sa] = -aaaaaa
							test_3_body[ta,ua,sa,pa,qa,ra] =  aaaaaa
							test_3_body[ta,ua,sa,pa,ra,qa] = -aaaaaa
							test_3_body[ta,ua,sa,qa,pa,ra] = -aaaaaa
							test_3_body[ta,ua,sa,qa,ra,pa] =  aaaaaa
							test_3_body[ta,ua,sa,ra,pa,qa] =  aaaaaa
							test_3_body[ta,ua,sa,ra,qa,pa] = -aaaaaa

							test_3_body[pa,qa,ra,ua,sa,ta] =  aaaaaa
							test_3_body[pa,ra,qa,ua,sa,ta] = -aaaaaa
							test_3_body[qa,pa,ra,ua,sa,ta] = -aaaaaa
							test_3_body[qa,ra,pa,ua,sa,ta] =  aaaaaa
							test_3_body[ra,pa,qa,ua,sa,ta] =  aaaaaa
							test_3_body[ra,qa,pa,ua,sa,ta] = -aaaaaa
							test_3_body[ua,sa,ta,pa,qa,ra] =  aaaaaa
							test_3_body[ua,sa,ta,pa,ra,qa] = -aaaaaa
							test_3_body[ua,sa,ta,qa,pa,ra] = -aaaaaa
							test_3_body[ua,sa,ta,qa,ra,pa] =  aaaaaa
							test_3_body[ua,sa,ta,ra,pa,qa] =  aaaaaa
							test_3_body[ua,sa,ta,ra,qa,pa] = -aaaaaa

							test_3_body[pa,qa,ra,ua,ta,sa] = -aaaaaa
							test_3_body[pa,ra,qa,ua,ta,sa] =  aaaaaa
							test_3_body[qa,pa,ra,ua,ta,sa] =  aaaaaa
							test_3_body[qa,ra,pa,ua,ta,sa] = -aaaaaa
							test_3_body[ra,pa,qa,ua,ta,sa] = -aaaaaa
							test_3_body[ra,qa,pa,ua,ta,sa] =  aaaaaa
							test_3_body[ua,ta,sa,pa,qa,ra] = -aaaaaa
							test_3_body[ua,ta,sa,pa,ra,qa] =  aaaaaa
							test_3_body[ua,ta,sa,qa,pa,ra] =  aaaaaa
							test_3_body[ua,ta,sa,qa,ra,pa] = -aaaaaa
							test_3_body[ua,ta,sa,ra,pa,qa] = -aaaaaa
							test_3_body[ua,ta,sa,ra,qa,pa] =  aaaaaa

	for p in range(0,n_orb):
		pa = 2*p 
		for q in range(p+1,n_orb):
			qa = 2*q 
			for r in range(0,n_orb):
				rb = 2*r+1 
				for s in range(0,n_orb):
					sa = 2*s 
					for t in range(s+1,n_orb):
						ta = 2*t  
						for u in range(0,n_orb):
							ub = 2*u+1 
							aabaab = 2*np.random.rand()-1
							test_3_body[pa,qa,rb,sa,ta,ub] =  aabaab
							test_3_body[pa,rb,qa,sa,ta,ub] = -aabaab
							test_3_body[qa,pa,rb,sa,ta,ub] = -aabaab
							test_3_body[qa,rb,pa,sa,ta,ub] =  aabaab
							test_3_body[rb,pa,qa,sa,ta,ub] =  aabaab
							test_3_body[rb,qa,pa,sa,ta,ub] = -aabaab
							test_3_body[sa,ta,ub,pa,qa,rb] =  aabaab
							test_3_body[sa,ta,ub,pa,rb,qa] = -aabaab
							test_3_body[sa,ta,ub,qa,pa,rb] = -aabaab
							test_3_body[sa,ta,ub,qa,rb,pa] =  aabaab
							test_3_body[sa,ta,ub,rb,pa,qa] =  aabaab
							test_3_body[sa,ta,ub,rb,qa,pa] = -aabaab

							test_3_body[pa,qa,rb,sa,ub,ta] = -aabaab
							test_3_body[pa,rb,qa,sa,ub,ta] =  aabaab
							test_3_body[qa,pa,rb,sa,ub,ta] =  aabaab
							test_3_body[qa,rb,pa,sa,ub,ta] = -aabaab
							test_3_body[rb,pa,qa,sa,ub,ta] = -aabaab
							test_3_body[rb,qa,pa,sa,ub,ta] =  aabaab
							test_3_body[sa,ub,ta,pa,qa,rb] = -aabaab
							test_3_body[sa,ub,ta,pa,rb,qa] =  aabaab
							test_3_body[sa,ub,ta,qa,pa,rb] =  aabaab
							test_3_body[sa,ub,ta,qa,rb,pa] = -aabaab
							test_3_body[sa,ub,ta,rb,pa,qa] = -aabaab
							test_3_body[sa,ub,ta,rb,qa,pa] =  aabaab

							test_3_body[pa,qa,rb,ta,sa,ub] = -aabaab
							test_3_body[pa,rb,qa,ta,sa,ub] =  aabaab
							test_3_body[qa,pa,rb,ta,sa,ub] =  aabaab
							test_3_body[qa,rb,pa,ta,sa,ub] = -aabaab
							test_3_body[rb,pa,qa,ta,sa,ub] = -aabaab
							test_3_body[rb,qa,pa,ta,sa,ub] =  aabaab
							test_3_body[ta,sa,ub,pa,qa,rb] = -aabaab
							test_3_body[ta,sa,ub,pa,rb,qa] =  aabaab
							test_3_body[ta,sa,ub,qa,pa,rb] =  aabaab
							test_3_body[ta,sa,ub,qa,rb,pa] = -aabaab
							test_3_body[ta,sa,ub,rb,pa,qa] = -aabaab
							test_3_body[ta,sa,ub,rb,qa,pa] =  aabaab

							test_3_body[pa,qa,rb,ta,ub,sa] =  aabaab
							test_3_body[pa,rb,qa,ta,ub,sa] = -aabaab
							test_3_body[qa,pa,rb,ta,ub,sa] = -aabaab
							test_3_body[qa,rb,pa,ta,ub,sa] =  aabaab
							test_3_body[rb,pa,qa,ta,ub,sa] =  aabaab
							test_3_body[rb,qa,pa,ta,ub,sa] = -aabaab
							test_3_body[ta,ub,sa,pa,qa,rb] =  aabaab
							test_3_body[ta,ub,sa,pa,rb,qa] = -aabaab
							test_3_body[ta,ub,sa,qa,pa,rb] = -aabaab
							test_3_body[ta,ub,sa,qa,rb,pa] =  aabaab
							test_3_body[ta,ub,sa,rb,pa,qa] =  aabaab
							test_3_body[ta,ub,sa,rb,qa,pa] = -aabaab

							test_3_body[pa,qa,rb,ub,sa,ta] =  aabaab
							test_3_body[pa,rb,qa,ub,sa,ta] = -aabaab
							test_3_body[qa,pa,rb,ub,sa,ta] = -aabaab
							test_3_body[qa,rb,pa,ub,sa,ta] =  aabaab
							test_3_body[rb,pa,qa,ub,sa,ta] =  aabaab
							test_3_body[rb,qa,pa,ub,sa,ta] = -aabaab
							test_3_body[ub,sa,ta,pa,qa,rb] =  aabaab
							test_3_body[ub,sa,ta,pa,rb,qa] = -aabaab
							test_3_body[ub,sa,ta,qa,pa,rb] = -aabaab
							test_3_body[ub,sa,ta,qa,rb,pa] =  aabaab
							test_3_body[ub,sa,ta,rb,pa,qa] =  aabaab
							test_3_body[ub,sa,ta,rb,qa,pa] = -aabaab

							test_3_body[pa,qa,rb,ub,ta,sa] = -aabaab
							test_3_body[pa,rb,qa,ub,ta,sa] =  aabaab
							test_3_body[qa,pa,rb,ub,ta,sa] =  aabaab
							test_3_body[qa,rb,pa,ub,ta,sa] = -aabaab
							test_3_body[rb,pa,qa,ub,ta,sa] = -aabaab
							test_3_body[rb,qa,pa,ub,ta,sa] =  aabaab
							test_3_body[ub,ta,sa,pa,qa,rb] = -aabaab
							test_3_body[ub,ta,sa,pa,rb,qa] =  aabaab
							test_3_body[ub,ta,sa,qa,pa,rb] =  aabaab
							test_3_body[ub,ta,sa,qa,rb,pa] = -aabaab
							test_3_body[ub,ta,sa,rb,pa,qa] = -aabaab
							test_3_body[ub,ta,sa,rb,qa,pa] =  aabaab
	for p in range(0,n_orb):
		pa = 2*p 
		for q in range(0,n_orb):
			qb = 2*q+1 
			for r in range(q+1,n_orb):
				rb = 2*r+1 
				for s in range(0,n_orb):
					sa = 2*s 
					for t in range(0,n_orb):
						tb = 2*t+1  
						for u in range(t+1,n_orb):
							ub = 2*u+1 
							abbabb = 2*np.random.rand()-1
							test_3_body[pa,qb,rb,sa,tb,ub] =  abbabb
							test_3_body[pa,rb,qb,sa,tb,ub] = -abbabb
							test_3_body[qb,pa,rb,sa,tb,ub] = -abbabb
							test_3_body[qb,rb,pa,sa,tb,ub] =  abbabb
							test_3_body[rb,pa,qb,sa,tb,ub] =  abbabb
							test_3_body[rb,qb,pa,sa,tb,ub] = -abbabb
							test_3_body[sa,tb,ub,pa,qb,rb] =  abbabb
							test_3_body[sa,tb,ub,pa,rb,qb] = -abbabb
							test_3_body[sa,tb,ub,qb,pa,rb] = -abbabb
							test_3_body[sa,tb,ub,qb,rb,pa] =  abbabb
							test_3_body[sa,tb,ub,rb,pa,qb] =  abbabb
							test_3_body[sa,tb,ub,rb,qb,pa] = -abbabb

							test_3_body[pa,qb,rb,sa,ub,tb] = -abbabb
							test_3_body[pa,rb,qb,sa,ub,tb] =  abbabb
							test_3_body[qb,pa,rb,sa,ub,tb] =  abbabb
							test_3_body[qb,rb,pa,sa,ub,tb] = -abbabb
							test_3_body[rb,pa,qb,sa,ub,tb] = -abbabb
							test_3_body[rb,qb,pa,sa,ub,tb] =  abbabb
							test_3_body[sa,ub,tb,pa,qb,rb] = -abbabb
							test_3_body[sa,ub,tb,pa,rb,qb] =  abbabb
							test_3_body[sa,ub,tb,qb,pa,rb] =  abbabb
							test_3_body[sa,ub,tb,qb,rb,pa] = -abbabb
							test_3_body[sa,ub,tb,rb,pa,qb] = -abbabb
							test_3_body[sa,ub,tb,rb,qb,pa] =  abbabb

							test_3_body[pa,qb,rb,tb,sa,ub] = -abbabb
							test_3_body[pa,rb,qb,tb,sa,ub] =  abbabb
							test_3_body[qb,pa,rb,tb,sa,ub] =  abbabb
							test_3_body[qb,rb,pa,tb,sa,ub] = -abbabb
							test_3_body[rb,pa,qb,tb,sa,ub] = -abbabb
							test_3_body[rb,qb,pa,tb,sa,ub] =  abbabb
							test_3_body[tb,sa,ub,pa,qb,rb] = -abbabb
							test_3_body[tb,sa,ub,pa,rb,qb] =  abbabb
							test_3_body[tb,sa,ub,qb,pa,rb] =  abbabb
							test_3_body[tb,sa,ub,qb,rb,pa] = -abbabb
							test_3_body[tb,sa,ub,rb,pa,qb] = -abbabb
							test_3_body[tb,sa,ub,rb,qb,pa] =  abbabb

							test_3_body[pa,qb,rb,tb,ub,sa] =  abbabb
							test_3_body[pa,rb,qb,tb,ub,sa] = -abbabb
							test_3_body[qb,pa,rb,tb,ub,sa] = -abbabb
							test_3_body[qb,rb,pa,tb,ub,sa] =  abbabb
							test_3_body[rb,pa,qb,tb,ub,sa] =  abbabb
							test_3_body[rb,qb,pa,tb,ub,sa] = -abbabb
							test_3_body[tb,ub,sa,pa,qb,rb] =  abbabb
							test_3_body[tb,ub,sa,pa,rb,qb] = -abbabb
							test_3_body[tb,ub,sa,qb,pa,rb] = -abbabb
							test_3_body[tb,ub,sa,qb,rb,pa] =  abbabb
							test_3_body[tb,ub,sa,rb,pa,qb] =  abbabb
							test_3_body[tb,ub,sa,rb,qb,pa] = -abbabb

							test_3_body[pa,qb,rb,ub,sa,tb] =  abbabb
							test_3_body[pa,rb,qb,ub,sa,tb] = -abbabb
							test_3_body[qb,pa,rb,ub,sa,tb] = -abbabb
							test_3_body[qb,rb,pa,ub,sa,tb] =  abbabb
							test_3_body[rb,pa,qb,ub,sa,tb] =  abbabb
							test_3_body[rb,qb,pa,ub,sa,tb] = -abbabb
							test_3_body[ub,sa,tb,pa,qb,rb] =  abbabb
							test_3_body[ub,sa,tb,pa,rb,qb] = -abbabb
							test_3_body[ub,sa,tb,qb,pa,rb] = -abbabb
							test_3_body[ub,sa,tb,qb,rb,pa] =  abbabb
							test_3_body[ub,sa,tb,rb,pa,qb] =  abbabb
							test_3_body[ub,sa,tb,rb,qb,pa] = -abbabb

							test_3_body[pa,qb,rb,ub,tb,sa] = -abbabb
							test_3_body[pa,rb,qb,ub,tb,sa] =  abbabb
							test_3_body[qb,pa,rb,ub,tb,sa] =  abbabb
							test_3_body[qb,rb,pa,ub,tb,sa] = -abbabb
							test_3_body[rb,pa,qb,ub,tb,sa] = -abbabb
							test_3_body[rb,qb,pa,ub,tb,sa] =  abbabb
							test_3_body[ub,tb,sa,pa,qb,rb] = -abbabb
							test_3_body[ub,tb,sa,pa,rb,qb] =  abbabb
							test_3_body[ub,tb,sa,qb,pa,rb] =  abbabb
							test_3_body[ub,tb,sa,qb,rb,pa] = -abbabb
							test_3_body[ub,tb,sa,rb,pa,qb] = -abbabb
							test_3_body[ub,tb,sa,rb,qb,pa] =  abbabb

	for p in range(0,n_orb):
		pb = 2*p+1 
		for q in range(p+1,n_orb):
			qb = 2*q+1 
			for r in range(q+1,n_orb):
				rb = 2*r+1 
				for s in range(0,n_orb):
					sb = 2*s+1 
					for t in range(s+1,n_orb):
						tb = 2*t+1  
						for u in range(t+1,n_orb):
							ub = 2*u+1 
							bbbbbb = 2*np.random.rand()-1
							test_3_body[pb,qb,rb,sb,tb,ub] =  bbbbbb
							test_3_body[pb,rb,qb,sb,tb,ub] = -bbbbbb
							test_3_body[qb,pb,rb,sb,tb,ub] = -bbbbbb
							test_3_body[qb,rb,pb,sb,tb,ub] =  bbbbbb
							test_3_body[rb,pb,qb,sb,tb,ub] =  bbbbbb
							test_3_body[rb,qb,pb,sb,tb,ub] = -bbbbbb
							test_3_body[sb,tb,ub,pb,qb,rb] =  bbbbbb
							test_3_body[sb,tb,ub,pb,rb,qb] = -bbbbbb
							test_3_body[sb,tb,ub,qb,pb,rb] = -bbbbbb
							test_3_body[sb,tb,ub,qb,rb,pb] =  bbbbbb
							test_3_body[sb,tb,ub,rb,pb,qb] =  bbbbbb
							test_3_body[sb,tb,ub,rb,qb,pb] = -bbbbbb

							test_3_body[pb,qb,rb,sb,ub,tb] = -bbbbbb
							test_3_body[pb,rb,qb,sb,ub,tb] =  bbbbbb
							test_3_body[qb,pb,rb,sb,ub,tb] =  bbbbbb
							test_3_body[qb,rb,pb,sb,ub,tb] = -bbbbbb
							test_3_body[rb,pb,qb,sb,ub,tb] = -bbbbbb
							test_3_body[rb,qb,pb,sb,ub,tb] =  bbbbbb
							test_3_body[sb,ub,tb,pb,qb,rb] = -bbbbbb
							test_3_body[sb,ub,tb,pb,rb,qb] =  bbbbbb
							test_3_body[sb,ub,tb,qb,pb,rb] =  bbbbbb
							test_3_body[sb,ub,tb,qb,rb,pb] = -bbbbbb
							test_3_body[sb,ub,tb,rb,pb,qb] = -bbbbbb
							test_3_body[sb,ub,tb,rb,qb,pb] =  bbbbbb

							test_3_body[pb,qb,rb,tb,sb,ub] = -bbbbbb
							test_3_body[pb,rb,qb,tb,sb,ub] =  bbbbbb
							test_3_body[qb,pb,rb,tb,sb,ub] =  bbbbbb
							test_3_body[qb,rb,pb,tb,sb,ub] = -bbbbbb
							test_3_body[rb,pb,qb,tb,sb,ub] = -bbbbbb
							test_3_body[rb,qb,pb,tb,sb,ub] =  bbbbbb
							test_3_body[tb,sb,ub,pb,qb,rb] = -bbbbbb
							test_3_body[tb,sb,ub,pb,rb,qb] =  bbbbbb
							test_3_body[tb,sb,ub,qb,pb,rb] =  bbbbbb
							test_3_body[tb,sb,ub,qb,rb,pb] = -bbbbbb
							test_3_body[tb,sb,ub,rb,pb,qb] = -bbbbbb
							test_3_body[tb,sb,ub,rb,qb,pb] =  bbbbbb

							test_3_body[pb,qb,rb,tb,ub,sb] =  bbbbbb
							test_3_body[pb,rb,qb,tb,ub,sb] = -bbbbbb
							test_3_body[qb,pb,rb,tb,ub,sb] = -bbbbbb
							test_3_body[qb,rb,pb,tb,ub,sb] =  bbbbbb
							test_3_body[rb,pb,qb,tb,ub,sb] =  bbbbbb
							test_3_body[rb,qb,pb,tb,ub,sb] = -bbbbbb
							test_3_body[tb,ub,sb,pb,qb,rb] =  bbbbbb
							test_3_body[tb,ub,sb,pb,rb,qb] = -bbbbbb
							test_3_body[tb,ub,sb,qb,pb,rb] = -bbbbbb
							test_3_body[tb,ub,sb,qb,rb,pb] =  bbbbbb
							test_3_body[tb,ub,sb,rb,pb,qb] =  bbbbbb
							test_3_body[tb,ub,sb,rb,qb,pb] = -bbbbbb

							test_3_body[pb,qb,rb,ub,sb,tb] =  bbbbbb
							test_3_body[pb,rb,qb,ub,sb,tb] = -bbbbbb
							test_3_body[qb,pb,rb,ub,sb,tb] = -bbbbbb
							test_3_body[qb,rb,pb,ub,sb,tb] =  bbbbbb
							test_3_body[rb,pb,qb,ub,sb,tb] =  bbbbbb
							test_3_body[rb,qb,pb,ub,sb,tb] = -bbbbbb
							test_3_body[ub,sb,tb,pb,qb,rb] =  bbbbbb
							test_3_body[ub,sb,tb,pb,rb,qb] = -bbbbbb
							test_3_body[ub,sb,tb,qb,pb,rb] = -bbbbbb
							test_3_body[ub,sb,tb,qb,rb,pb] =  bbbbbb
							test_3_body[ub,sb,tb,rb,pb,qb] =  bbbbbb
							test_3_body[ub,sb,tb,rb,qb,pb] = -bbbbbb

							test_3_body[pb,qb,rb,ub,tb,sb] = -bbbbbb
							test_3_body[pb,rb,qb,ub,tb,sb] =  bbbbbb
							test_3_body[qb,pb,rb,ub,tb,sb] =  bbbbbb
							test_3_body[qb,rb,pb,ub,tb,sb] = -bbbbbb
							test_3_body[rb,pb,qb,ub,tb,sb] = -bbbbbb
							test_3_body[rb,qb,pb,ub,tb,sb] =  bbbbbb
							test_3_body[ub,tb,sb,pb,qb,rb] = -bbbbbb
							test_3_body[ub,tb,sb,pb,rb,qb] =  bbbbbb
							test_3_body[ub,tb,sb,qb,pb,rb] =  bbbbbb
							test_3_body[ub,tb,sb,qb,rb,pb] = -bbbbbb
							test_3_body[ub,tb,sb,rb,pb,qb] = -bbbbbb
							test_3_body[ub,tb,sb,rb,qb,pb] =  bbbbbb
	return test_3_body

comment = """
test_3_body = rand_3_body(n_act)
print("3-Body Tensor Tests")
tprint(test_3_body)
print("Hermitian:")
tprint(test_3_body - np.einsum("pqrstu->stupqr",test_3_body,optimize="optimal"))
print("Antisymmetric?")
print("pqrstu->prqstu")
tprint(test_3_body + np.einsum("pqrstu->prqstu",test_3_body,optimize="optimal"))
print("pqrstu->qprstu")
tprint(test_3_body + np.einsum("pqrstu->qprstu",test_3_body,optimize="optimal"))
print("pqrstu->qrpstu")
tprint(test_3_body - np.einsum("pqrstu->qrpstu",test_3_body,optimize="optimal"))
print("pqrstu->rpqstu")
tprint(test_3_body - np.einsum("pqrstu->rpqstu",test_3_body,optimize="optimal"))
print("pqrstu->rqpstu")
tprint(test_3_body + np.einsum("pqrstu->rqpstu",test_3_body,optimize="optimal"))
print("pqrstu->pqrsut")
tprint(test_3_body + np.einsum("pqrstu->pqrsut",test_3_body,optimize="optimal"))
print("pqrstu->prqsut")
tprint(test_3_body - np.einsum("pqrstu->prqsut",test_3_body,optimize="optimal"))
print("pqrstu->qprsut")
tprint(test_3_body - np.einsum("pqrstu->qprsut",test_3_body,optimize="optimal"))
print("pqrstu->qrpsut")
tprint(test_3_body + np.einsum("pqrstu->qrpsut",test_3_body,optimize="optimal"))
print("pqrstu->rpqsut")
tprint(test_3_body + np.einsum("pqrstu->rpqsut",test_3_body,optimize="optimal"))
print("pqrstu->rqpsut")
tprint(test_3_body - np.einsum("pqrstu->rqpsut",test_3_body,optimize="optimal"))
print("pqrstu->pqrtsu")
tprint(test_3_body + np.einsum("pqrstu->pqrtsu",test_3_body,optimize="optimal"))
print("pqrstu->prqtsu")
tprint(test_3_body - np.einsum("pqrstu->prqtsu",test_3_body,optimize="optimal"))
print("pqrstu->qprtsu")
tprint(test_3_body - np.einsum("pqrstu->qprtsu",test_3_body,optimize="optimal"))
print("pqrstu->qrptsu")
tprint(test_3_body + np.einsum("pqrstu->qrptsu",test_3_body,optimize="optimal"))
print("pqrstu->rpqtsu")
tprint(test_3_body + np.einsum("pqrstu->rpqtsu",test_3_body,optimize="optimal"))
print("pqrstu->rqptsu")
tprint(test_3_body - np.einsum("pqrstu->rqptsu",test_3_body,optimize="optimal"))
print("pqrstu->pqrtus")
tprint(test_3_body - np.einsum("pqrstu->pqrtus",test_3_body,optimize="optimal"))
print("pqrstu->prqtus")
tprint(test_3_body + np.einsum("pqrstu->prqtus",test_3_body,optimize="optimal"))
print("pqrstu->qprtus")
tprint(test_3_body + np.einsum("pqrstu->qprtus",test_3_body,optimize="optimal"))
print("pqrstu->qrptus")
tprint(test_3_body - np.einsum("pqrstu->qrptus",test_3_body,optimize="optimal"))
print("pqrstu->rpqtus")
tprint(test_3_body - np.einsum("pqrstu->rpqtus",test_3_body,optimize="optimal"))
print("pqrstu->rqptus")
tprint(test_3_body + np.einsum("pqrstu->rqptus",test_3_body,optimize="optimal"))
print("pqrstu->pqrust")
tprint(test_3_body - np.einsum("pqrstu->pqrust",test_3_body,optimize="optimal"))
print("pqrstu->prqust")
tprint(test_3_body + np.einsum("pqrstu->prqust",test_3_body,optimize="optimal"))
print("pqrstu->qprust")
tprint(test_3_body + np.einsum("pqrstu->qprust",test_3_body,optimize="optimal"))
print("pqrstu->qrpust")
tprint(test_3_body - np.einsum("pqrstu->qrpust",test_3_body,optimize="optimal"))
print("pqrstu->rpqust")
tprint(test_3_body - np.einsum("pqrstu->rpqust",test_3_body,optimize="optimal"))
print("pqrstu->rqpust")
tprint(test_3_body + np.einsum("pqrstu->rqpust",test_3_body,optimize="optimal"))
print("pqrstu->pqruts")
tprint(test_3_body + np.einsum("pqrstu->pqruts",test_3_body,optimize="optimal"))
print("pqrstu->prquts")
tprint(test_3_body - np.einsum("pqrstu->prquts",test_3_body,optimize="optimal"))
print("pqrstu->qpruts")
tprint(test_3_body - np.einsum("pqrstu->qpruts",test_3_body,optimize="optimal"))
print("pqrstu->qrputs")
tprint(test_3_body + np.einsum("pqrstu->qrputs",test_3_body,optimize="optimal"))
print("pqrstu->rpquts")
tprint(test_3_body + np.einsum("pqrstu->rpquts",test_3_body,optimize="optimal"))
print("pqrstu->rqputs")
tprint(test_3_body - np.einsum("pqrstu->rqputs",test_3_body,optimize="optimal"))
"""

def rand_3_body_r(n_orb):
	test_3_body = np.zeros((2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb,2*n_orb))
	for p in range(0,n_orb):
		pa = 2*p 
		pb = 2*p+1 
		for q in range(0,n_orb):
			qa = 2*q 
			qb = 2*q+1 
			for r in range(0,n_orb):
				ra = 2*r 
				rb = 2*r+1 
				for s in range(0,n_orb):
					sa = 2*s 
					sb = 2*s+1 
					for t in range(0,n_orb):
						ta = 2*t 
						tb = 2*t+1 
						for u in range(0,n_orb):
							ua = 2*u 
							ub = 2*u+1 
							pqr_stu = 2*np.random.rand()-1 
							pqr_sut = 2*np.random.rand()-1 
							pqr_tsu = 2*np.random.rand()-1 
							pqr_tus = 2*np.random.rand()-1 
							pqr_ust = 2*np.random.rand()-1 
							pqr_uts = 2*np.random.rand()-1 
							if (p < q):
								if(q < r):
									if(s < t):
										if(t < u):
											aaaaaa = pqr_stu - pqr_sut - pqr_tsu + pqr_tus + pqr_ust - pqr_uts
											test_3_body[pa,qa,ra,sa,ta,ua] =  aaaaaa 
											test_3_body[pa,qa,ra,sa,ua,ta] = -aaaaaa 
											test_3_body[pa,qa,ra,ta,sa,ua] = -aaaaaa
											test_3_body[pa,qa,ra,ta,ua,sa] =  aaaaaa 
											test_3_body[pa,qa,ra,ua,sa,ta] =  aaaaaa 
											test_3_body[pa,qa,ra,ua,ta,sa] = -aaaaaa
											test_3_body[sa,ta,ua,pa,qa,ra] =  aaaaaa 
											test_3_body[sa,ua,ta,pa,qa,ra] = -aaaaaa 
											test_3_body[ta,sa,ua,pa,qa,ra] = -aaaaaa
											test_3_body[ta,ua,sa,pa,qa,ra] =  aaaaaa 
											test_3_body[ua,sa,ta,pa,qa,ra] =  aaaaaa 
											test_3_body[ua,ta,sa,pa,qa,ra] = -aaaaaa

											test_3_body[pa,ra,qa,sa,ta,ua] = -aaaaaa 
											test_3_body[pa,ra,qa,sa,ua,ta] =  aaaaaa 
											test_3_body[pa,ra,qa,ta,sa,ua] =  aaaaaa
											test_3_body[pa,ra,qa,ta,ua,sa] = -aaaaaa 
											test_3_body[pa,ra,qa,ua,sa,ta] = -aaaaaa 
											test_3_body[pa,ra,qa,ua,ta,sa] =  aaaaaa
											test_3_body[sa,ta,ua,pa,ra,qa] = -aaaaaa 
											test_3_body[sa,ua,ta,pa,ra,qa] =  aaaaaa 
											test_3_body[ta,sa,ua,pa,ra,qa] =  aaaaaa
											test_3_body[ta,ua,sa,pa,ra,qa] = -aaaaaa 
											test_3_body[ua,sa,ta,pa,ra,qa] = -aaaaaa 
											test_3_body[ua,ta,sa,pa,ra,qa] =  aaaaaa

											test_3_body[qa,pa,ra,sa,ta,ua] = -aaaaaa 
											test_3_body[qa,pa,ra,sa,ua,ta] =  aaaaaa 
											test_3_body[qa,pa,ra,ta,sa,ua] =  aaaaaa
											test_3_body[qa,pa,ra,ta,ua,sa] = -aaaaaa 
											test_3_body[qa,pa,ra,ua,sa,ta] = -aaaaaa 
											test_3_body[qa,pa,ra,ua,ta,sa] =  aaaaaa
											test_3_body[sa,ta,ua,qa,pa,ra] = -aaaaaa 
											test_3_body[sa,ua,ta,qa,pa,ra] =  aaaaaa 
											test_3_body[ta,sa,ua,qa,pa,ra] =  aaaaaa
											test_3_body[ta,ua,sa,qa,pa,ra] = -aaaaaa 
											test_3_body[ua,sa,ta,qa,pa,ra] = -aaaaaa 
											test_3_body[ua,ta,sa,qa,pa,ra] =  aaaaaa

											test_3_body[qa,ra,pa,sa,ta,ua] =  aaaaaa 
											test_3_body[qa,ra,pa,sa,ua,ta] = -aaaaaa 
											test_3_body[qa,ra,pa,ta,sa,ua] = -aaaaaa
											test_3_body[qa,ra,pa,ta,ua,sa] =  aaaaaa 
											test_3_body[qa,ra,pa,ua,sa,ta] =  aaaaaa 
											test_3_body[qa,ra,pa,ua,ta,sa] = -aaaaaa
											test_3_body[sa,ta,ua,qa,ra,pa] =  aaaaaa 
											test_3_body[sa,ua,ta,qa,ra,pa] = -aaaaaa 
											test_3_body[ta,sa,ua,qa,ra,pa] = -aaaaaa
											test_3_body[ta,ua,sa,qa,ra,pa] =  aaaaaa 
											test_3_body[ua,sa,ta,qa,ra,pa] =  aaaaaa 
											test_3_body[ua,ta,sa,qa,ra,pa] = -aaaaaa

											test_3_body[ra,pa,qa,sa,ta,ua] =  aaaaaa 
											test_3_body[ra,pa,qa,sa,ua,ta] = -aaaaaa 
											test_3_body[ra,pa,qa,ta,sa,ua] = -aaaaaa
											test_3_body[ra,pa,qa,ta,ua,sa] =  aaaaaa 
											test_3_body[ra,pa,qa,ua,sa,ta] =  aaaaaa 
											test_3_body[ra,pa,qa,ua,ta,sa] = -aaaaaa
											test_3_body[sa,ta,ua,ra,pa,qa] =  aaaaaa 
											test_3_body[sa,ua,ta,ra,pa,qa] = -aaaaaa 
											test_3_body[ta,sa,ua,ra,pa,qa] = -aaaaaa
											test_3_body[ta,ua,sa,ra,pa,qa] =  aaaaaa 
											test_3_body[ua,sa,ta,ra,pa,qa] =  aaaaaa 
											test_3_body[ua,ta,sa,ra,pa,qa] = -aaaaaa

											test_3_body[ra,qa,pa,sa,ta,ua] = -aaaaaa 
											test_3_body[ra,qa,pa,sa,ua,ta] =  aaaaaa 
											test_3_body[ra,qa,pa,ta,sa,ua] =  aaaaaa
											test_3_body[ra,qa,pa,ta,ua,sa] = -aaaaaa 
											test_3_body[ra,qa,pa,ua,sa,ta] = -aaaaaa 
											test_3_body[ra,qa,pa,ua,ta,sa] =  aaaaaa
											test_3_body[sa,ta,ua,ra,qa,pa] = -aaaaaa 
											test_3_body[sa,ua,ta,ra,qa,pa] =  aaaaaa 
											test_3_body[ta,sa,ua,ra,qa,pa] =  aaaaaa
											test_3_body[ta,ua,sa,ra,qa,pa] = -aaaaaa 
											test_3_body[ua,sa,ta,ra,qa,pa] = -aaaaaa 
											test_3_body[ua,ta,sa,ra,qa,pa] =  aaaaaa

											test_3_body[pb,qb,rb,sb,tb,ub] =  aaaaaa 
											test_3_body[pb,qb,rb,sb,ub,tb] = -aaaaaa 
											test_3_body[pb,qb,rb,tb,sb,ub] = -aaaaaa
											test_3_body[pb,qb,rb,tb,ub,sb] =  aaaaaa 
											test_3_body[pb,qb,rb,ub,sb,tb] =  aaaaaa 
											test_3_body[pb,qb,rb,ub,tb,sb] = -aaaaaa
											test_3_body[sb,tb,ub,pb,qb,rb] =  aaaaaa 
											test_3_body[sb,ub,tb,pb,qb,rb] = -aaaaaa 
											test_3_body[tb,sb,ub,pb,qb,rb] = -aaaaaa
											test_3_body[tb,ub,sb,pb,qb,rb] =  aaaaaa 
											test_3_body[ub,sb,tb,pb,qb,rb] =  aaaaaa 
											test_3_body[ub,tb,sb,pb,qb,rb] = -aaaaaa

											test_3_body[pb,rb,qb,sb,tb,ub] = -aaaaaa 
											test_3_body[pb,rb,qb,sb,ub,tb] =  aaaaaa 
											test_3_body[pb,rb,qb,tb,sb,ub] =  aaaaaa
											test_3_body[pb,rb,qb,tb,ub,sb] = -aaaaaa 
											test_3_body[pb,rb,qb,ub,sb,tb] = -aaaaaa 
											test_3_body[pb,rb,qb,ub,tb,sb] =  aaaaaa
											test_3_body[sb,tb,ub,pb,rb,qb] = -aaaaaa 
											test_3_body[sb,ub,tb,pb,rb,qb] =  aaaaaa 
											test_3_body[tb,sb,ub,pb,rb,qb] =  aaaaaa
											test_3_body[tb,ub,sb,pb,rb,qb] = -aaaaaa 
											test_3_body[ub,sb,tb,pb,rb,qb] = -aaaaaa 
											test_3_body[ub,tb,sb,pb,rb,qb] =  aaaaaa

											test_3_body[qb,pb,rb,sb,tb,ub] = -aaaaaa 
											test_3_body[qb,pb,rb,sb,ub,tb] =  aaaaaa 
											test_3_body[qb,pb,rb,tb,sb,ub] =  aaaaaa
											test_3_body[qb,pb,rb,tb,ub,sb] = -aaaaaa 
											test_3_body[qb,pb,rb,ub,sb,tb] = -aaaaaa 
											test_3_body[qb,pb,rb,ub,tb,sb] =  aaaaaa
											test_3_body[sb,tb,ub,qb,pb,rb] = -aaaaaa 
											test_3_body[sb,ub,tb,qb,pb,rb] =  aaaaaa 
											test_3_body[tb,sb,ub,qb,pb,rb] =  aaaaaa
											test_3_body[tb,ub,sb,qb,pb,rb] = -aaaaaa 
											test_3_body[ub,sb,tb,qb,pb,rb] = -aaaaaa 
											test_3_body[ub,tb,sb,qb,pb,rb] =  aaaaaa

											test_3_body[qb,rb,pb,sb,tb,ub] =  aaaaaa 
											test_3_body[qb,rb,pb,sb,ub,tb] = -aaaaaa 
											test_3_body[qb,rb,pb,tb,sb,ub] = -aaaaaa
											test_3_body[qb,rb,pb,tb,ub,sb] =  aaaaaa 
											test_3_body[qb,rb,pb,ub,sb,tb] =  aaaaaa 
											test_3_body[qb,rb,pb,ub,tb,sb] = -aaaaaa
											test_3_body[sb,tb,ub,qb,rb,pb] =  aaaaaa 
											test_3_body[sb,ub,tb,qb,rb,pb] = -aaaaaa 
											test_3_body[tb,sb,ub,qb,rb,pb] = -aaaaaa
											test_3_body[tb,ub,sb,qb,rb,pb] =  aaaaaa 
											test_3_body[ub,sb,tb,qb,rb,pb] =  aaaaaa 
											test_3_body[ub,tb,sb,qb,rb,pb] = -aaaaaa

											test_3_body[rb,pb,qb,sb,tb,ub] =  aaaaaa 
											test_3_body[rb,pb,qb,sb,ub,tb] = -aaaaaa 
											test_3_body[rb,pb,qb,tb,sb,ub] = -aaaaaa
											test_3_body[rb,pb,qb,tb,ub,sb] =  aaaaaa 
											test_3_body[rb,pb,qb,ub,sb,tb] =  aaaaaa 
											test_3_body[rb,pb,qb,ub,tb,sb] = -aaaaaa
											test_3_body[sb,tb,ub,rb,pb,qb] =  aaaaaa 
											test_3_body[sb,ub,tb,rb,pb,qb] = -aaaaaa 
											test_3_body[tb,sb,ub,rb,pb,qb] = -aaaaaa
											test_3_body[tb,ub,sb,rb,pb,qb] =  aaaaaa 
											test_3_body[ub,sb,tb,rb,pb,qb] =  aaaaaa 
											test_3_body[ub,tb,sb,rb,pb,qb] = -aaaaaa

											test_3_body[rb,qb,pb,sb,tb,ub] = -aaaaaa 
											test_3_body[rb,qb,pb,sb,ub,tb] =  aaaaaa 
											test_3_body[rb,qb,pb,tb,sb,ub] =  aaaaaa
											test_3_body[rb,qb,pb,tb,ub,sb] = -aaaaaa 
											test_3_body[rb,qb,pb,ub,sb,tb] = -aaaaaa 
											test_3_body[rb,qb,pb,ub,tb,sb] =  aaaaaa
											test_3_body[sb,tb,ub,rb,qb,pb] = -aaaaaa 
											test_3_body[sb,ub,tb,rb,qb,pb] =  aaaaaa 
											test_3_body[tb,sb,ub,rb,qb,pb] =  aaaaaa
											test_3_body[tb,ub,sb,rb,qb,pb] = -aaaaaa 
											test_3_body[ub,sb,tb,rb,qb,pb] = -aaaaaa 
											test_3_body[ub,tb,sb,rb,qb,pb] =  aaaaaa
							if(p < q):
								if(s < t):
									aabaab = pqr_stu - pqr_tsu
									test_3_body[pa,qa,rb,sa,ta,ub] =  aabaab
									test_3_body[pa,qa,rb,sa,ub,ta] = -aabaab
									test_3_body[pa,qa,rb,ta,sa,ub] = -aabaab
									test_3_body[pa,qa,rb,ta,ub,sa] =  aabaab 
									test_3_body[pa,qa,rb,ub,sa,ta] =  aabaab
									test_3_body[pa,qa,rb,ub,ta,sa] = -aabaab
									test_3_body[sa,ta,ub,pa,qa,rb] =  aabaab
									test_3_body[sa,ub,ta,pa,qa,rb] = -aabaab
									test_3_body[ta,sa,ub,pa,qa,rb] = -aabaab
									test_3_body[ta,ub,sa,pa,qa,rb] =  aabaab 
									test_3_body[ub,sa,ta,pa,qa,rb] =  aabaab
									test_3_body[ub,ta,sa,pa,qa,rb] = -aabaab

									test_3_body[pb,qb,ra,sb,tb,ua] =  aabaab
									test_3_body[pb,qb,ra,sb,ua,tb] = -aabaab
									test_3_body[pb,qb,ra,tb,sb,ua] = -aabaab
									test_3_body[pb,qb,ra,tb,ua,sb] =  aabaab 
									test_3_body[pb,qb,ra,ua,sb,tb] =  aabaab
									test_3_body[pb,qb,ra,ua,tb,sb] = -aabaab
									test_3_body[sb,tb,ua,pb,qb,ra] =  aabaab
									test_3_body[sb,ua,tb,pb,qb,ra] = -aabaab
									test_3_body[tb,sb,ua,pb,qb,ra] = -aabaab
									test_3_body[tb,ua,sb,pb,qb,ra] =  aabaab 
									test_3_body[ua,sb,tb,pb,qb,ra] =  aabaab
									test_3_body[ua,tb,sb,pb,qb,ra] = -aabaab

									test_3_body[pa,rb,qa,sa,ta,ub] = -aabaab
									test_3_body[pa,rb,qa,sa,ub,ta] =  aabaab
									test_3_body[pa,rb,qa,ta,sa,ub] =  aabaab
									test_3_body[pa,rb,qa,ta,ub,sa] = -aabaab 
									test_3_body[pa,rb,qa,ub,sa,ta] = -aabaab
									test_3_body[pa,rb,qa,ub,ta,sa] =  aabaab
									test_3_body[sa,ta,ub,pa,rb,qa] = -aabaab
									test_3_body[sa,ub,ta,pa,rb,qa] =  aabaab
									test_3_body[ta,sa,ub,pa,rb,qa] =  aabaab
									test_3_body[ta,ub,sa,pa,rb,qa] = -aabaab 
									test_3_body[ub,sa,ta,pa,rb,qa] = -aabaab
									test_3_body[ub,ta,sa,pa,rb,qa] =  aabaab

									test_3_body[pb,ra,qb,sb,tb,ua] = -aabaab
									test_3_body[pb,ra,qb,sb,ua,tb] =  aabaab
									test_3_body[pb,ra,qb,tb,sb,ua] =  aabaab
									test_3_body[pb,ra,qb,tb,ua,sb] = -aabaab 
									test_3_body[pb,ra,qb,ua,sb,tb] = -aabaab
									test_3_body[pb,ra,qb,ua,tb,sb] =  aabaab
									test_3_body[sb,tb,ua,pb,ra,qb] = -aabaab
									test_3_body[sb,ua,tb,pb,ra,qb] =  aabaab
									test_3_body[tb,sb,ua,pb,ra,qb] =  aabaab
									test_3_body[tb,ua,sb,pb,ra,qb] = -aabaab 
									test_3_body[ua,sb,tb,pb,ra,qb] = -aabaab
									test_3_body[ua,tb,sb,pb,ra,qb] =  aabaab

									test_3_body[qa,pa,rb,sa,ta,ub] = -aabaab
									test_3_body[qa,pa,rb,sa,ub,ta] =  aabaab
									test_3_body[qa,pa,rb,ta,sa,ub] =  aabaab
									test_3_body[qa,pa,rb,ta,ub,sa] = -aabaab 
									test_3_body[qa,pa,rb,ub,sa,ta] = -aabaab
									test_3_body[qa,pa,rb,ub,ta,sa] =  aabaab
									test_3_body[sa,ta,ub,qa,pa,rb] = -aabaab
									test_3_body[sa,ub,ta,qa,pa,rb] =  aabaab
									test_3_body[ta,sa,ub,qa,pa,rb] =  aabaab
									test_3_body[ta,ub,sa,qa,pa,rb] = -aabaab 
									test_3_body[ub,sa,ta,qa,pa,rb] = -aabaab
									test_3_body[ub,ta,sa,qa,pa,rb] =  aabaab

									test_3_body[qb,pb,ra,sb,tb,ua] = -aabaab
									test_3_body[qb,pb,ra,sb,ua,tb] =  aabaab
									test_3_body[qb,pb,ra,tb,sb,ua] =  aabaab
									test_3_body[qb,pb,ra,tb,ua,sb] = -aabaab 
									test_3_body[qb,pb,ra,ua,sb,tb] = -aabaab
									test_3_body[qb,pb,ra,ua,tb,sb] =  aabaab
									test_3_body[sb,tb,ua,qb,pb,ra] = -aabaab
									test_3_body[sb,ua,tb,qb,pb,ra] =  aabaab
									test_3_body[tb,sb,ua,qb,pb,ra] =  aabaab
									test_3_body[tb,ua,sb,qb,pb,ra] = -aabaab 
									test_3_body[ua,sb,tb,qb,pb,ra] = -aabaab
									test_3_body[ua,tb,sb,qb,pb,ra] =  aabaab

									test_3_body[qa,rb,pa,sa,ta,ub] =  aabaab
									test_3_body[qa,rb,pa,sa,ub,ta] = -aabaab
									test_3_body[qa,rb,pa,ta,sa,ub] = -aabaab
									test_3_body[qa,rb,pa,ta,ub,sa] =  aabaab 
									test_3_body[qa,rb,pa,ub,sa,ta] =  aabaab
									test_3_body[qa,rb,pa,ub,ta,sa] = -aabaab
									test_3_body[sa,ta,ub,qa,rb,pa] =  aabaab
									test_3_body[sa,ub,ta,qa,rb,pa] = -aabaab
									test_3_body[ta,sa,ub,qa,rb,pa] = -aabaab
									test_3_body[ta,ub,sa,qa,rb,pa] =  aabaab 
									test_3_body[ub,sa,ta,qa,rb,pa] =  aabaab
									test_3_body[ub,ta,sa,qa,rb,pa] = -aabaab

									test_3_body[qb,ra,pb,sb,tb,ua] =  aabaab
									test_3_body[qb,ra,pb,sb,ua,tb] = -aabaab
									test_3_body[qb,ra,pb,tb,sb,ua] = -aabaab
									test_3_body[qb,ra,pb,tb,ua,sb] =  aabaab 
									test_3_body[qb,ra,pb,ua,sb,tb] =  aabaab
									test_3_body[qb,ra,pb,ua,tb,sb] = -aabaab
									test_3_body[sb,tb,ua,qb,ra,pb] =  aabaab
									test_3_body[sb,ua,tb,qb,ra,pb] = -aabaab
									test_3_body[tb,sb,ua,qb,ra,pb] = -aabaab
									test_3_body[tb,ua,sb,qb,ra,pb] =  aabaab 
									test_3_body[ua,sb,tb,qb,ra,pb] =  aabaab
									test_3_body[ua,tb,sb,qb,ra,pb] = -aabaab

									test_3_body[rb,pa,qa,sa,ta,ub] =  aabaab
									test_3_body[rb,pa,qa,sa,ub,ta] = -aabaab
									test_3_body[rb,pa,qa,ta,sa,ub] = -aabaab
									test_3_body[rb,pa,qa,ta,ub,sa] =  aabaab 
									test_3_body[rb,pa,qa,ub,sa,ta] =  aabaab
									test_3_body[rb,pa,qa,ub,ta,sa] = -aabaab
									test_3_body[sa,ta,ub,rb,pa,qa] =  aabaab
									test_3_body[sa,ub,ta,rb,pa,qa] = -aabaab
									test_3_body[ta,sa,ub,rb,pa,qa] = -aabaab
									test_3_body[ta,ub,sa,rb,pa,qa] =  aabaab 
									test_3_body[ub,sa,ta,rb,pa,qa] =  aabaab
									test_3_body[ub,ta,sa,rb,pa,qa] = -aabaab

									test_3_body[ra,pb,qb,sb,tb,ua] =  aabaab
									test_3_body[ra,pb,qb,sb,ua,tb] = -aabaab
									test_3_body[ra,pb,qb,tb,sb,ua] = -aabaab
									test_3_body[ra,pb,qb,tb,ua,sb] =  aabaab 
									test_3_body[ra,pb,qb,ua,sb,tb] =  aabaab
									test_3_body[ra,pb,qb,ua,tb,sb] = -aabaab
									test_3_body[sb,tb,ua,ra,pb,qb] =  aabaab
									test_3_body[sb,ua,tb,ra,pb,qb] = -aabaab
									test_3_body[tb,sb,ua,ra,pb,qb] = -aabaab
									test_3_body[tb,ua,sb,ra,pb,qb] =  aabaab 
									test_3_body[ua,sb,tb,ra,pb,qb] =  aabaab
									test_3_body[ua,tb,sb,ra,pb,qb] = -aabaab

									test_3_body[rb,qa,pa,sa,ta,ub] = -aabaab
									test_3_body[rb,qa,pa,sa,ub,ta] =  aabaab
									test_3_body[rb,qa,pa,ta,sa,ub] =  aabaab
									test_3_body[rb,qa,pa,ta,ub,sa] = -aabaab 
									test_3_body[rb,qa,pa,ub,sa,ta] = -aabaab
									test_3_body[rb,qa,pa,ub,ta,sa] =  aabaab
									test_3_body[sa,ta,ub,rb,qa,pa] = -aabaab
									test_3_body[sa,ub,ta,rb,qa,pa] =  aabaab
									test_3_body[ta,sa,ub,rb,qa,pa] =  aabaab
									test_3_body[ta,ub,sa,rb,qa,pa] = -aabaab 
									test_3_body[ub,sa,ta,rb,qa,pa] = -aabaab
									test_3_body[ub,ta,sa,rb,qa,pa] =  aabaab

									test_3_body[ra,qb,pb,sb,tb,ua] = -aabaab
									test_3_body[ra,qb,pb,sb,ua,tb] =  aabaab
									test_3_body[ra,qb,pb,tb,sb,ua] =  aabaab
									test_3_body[ra,qb,pb,tb,ua,sb] = -aabaab 
									test_3_body[ra,qb,pb,ua,sb,tb] = -aabaab
									test_3_body[ra,qb,pb,ua,tb,sb] =  aabaab
									test_3_body[sb,tb,ua,ra,qb,pb] = -aabaab
									test_3_body[sb,ua,tb,ra,qb,pb] =  aabaab
									test_3_body[tb,sb,ua,ra,qb,pb] =  aabaab
									test_3_body[tb,ua,sb,ra,qb,pb] = -aabaab 
									test_3_body[ua,sb,tb,ra,qb,pb] = -aabaab
									test_3_body[ua,tb,sb,ra,qb,pb] =  aabaab
	return test_3_body

comment= """
test_3_body = rand_3_body_r(n_act)
print("3-Body Tensor Tests")
print("Hermitian:")
tprint(test_3_body - np.einsum("pqrstu->stupqr",test_3_body,optimize="optimal"))
print("a<->b")
test_3_body_sf = np.zeros((2*n_act,2*n_act,2*n_act,2*n_act,2*n_act,2*n_act))
for p in range(0,n_act):
	pa = 2*p 
	pb = 2*p+1 
	for q in range(0,n_act):
		qa = 2*q 
		qb = 2*q+1 
		for r in range(0,n_act):
			ra = 2*r 
			rb = 2*r+1 
			for s in range(0,n_act):
				sa = 2*s 
				sb = 2*s+1 
				for t in range(0,n_act):
					ta = 2*t 
					tb = 2*t+1 
					for u in range(0,n_act):
						ua = 2*u 
						ub = 2*u+1 
						test_3_body_sf[pb,qb,rb,sb,tb,ub] = test_3_body[pa,qa,ra,sa,ta,ua]
						test_3_body_sf[pa,qa,ra,sa,ta,ua] = test_3_body[pb,qb,rb,sb,tb,ub]

						test_3_body_sf[pa,qa,rb,sa,ta,ub] = test_3_body[pb,qb,ra,sb,tb,ua]
						test_3_body_sf[pa,qa,rb,sa,tb,ua] = test_3_body[pb,qb,ra,sb,ta,ub]
						test_3_body_sf[pa,qa,rb,sb,ta,ua] = test_3_body[pb,qb,ra,sa,tb,ub]
						test_3_body_sf[pa,qb,ra,sa,ta,ub] = test_3_body[pb,qa,rb,sb,tb,ua]
						test_3_body_sf[pa,qb,ra,sa,tb,ua] = test_3_body[pb,qa,rb,sb,ta,ub]
						test_3_body_sf[pa,qb,ra,sb,ta,ua] = test_3_body[pb,qa,rb,sa,tb,ub]
						test_3_body_sf[pb,qa,ra,sa,ta,ub] = test_3_body[pa,qb,rb,sb,tb,ua]
						test_3_body_sf[pb,qa,ra,sa,tb,ua] = test_3_body[pa,qb,rb,sb,ta,ub]
						test_3_body_sf[pb,qa,ra,sb,ta,ua] = test_3_body[pa,qb,rb,sa,tb,ub]

						test_3_body_sf[pa,qb,rb,sa,tb,ub] = test_3_body[pb,qa,ra,sb,ta,ua]
						test_3_body_sf[pa,qb,rb,sb,ta,ub] = test_3_body[pb,qa,ra,sa,tb,ua]
						test_3_body_sf[pa,qb,rb,sb,tb,ua] = test_3_body[pb,qa,ra,sa,ta,ub]
						test_3_body_sf[pb,qa,rb,sa,tb,ub] = test_3_body[pa,qb,ra,sb,ta,ua]
						test_3_body_sf[pb,qa,rb,sb,ta,ub] = test_3_body[pa,qb,ra,sa,tb,ua]
						test_3_body_sf[pb,qa,rb,sb,tb,ua] = test_3_body[pa,qb,ra,sa,ta,ub]
						test_3_body_sf[pb,qb,ra,sa,tb,ub] = test_3_body[pa,qa,rb,sb,ta,ua]
						test_3_body_sf[pb,qb,ra,sb,ta,ub] = test_3_body[pa,qa,rb,sa,tb,ua]
						test_3_body_sf[pb,qb,ra,sb,tb,ua] = test_3_body[pa,qa,rb,sa,ta,ub]
						
tprint(test_3_body - test_3_body_sf)
print("Antisymmetric?")
print("pqrstu->prqstu")
tprint(test_3_body + np.einsum("pqrstu->prqstu",test_3_body,optimize="optimal"))
print("pqrstu->qprstu")
tprint(test_3_body + np.einsum("pqrstu->qprstu",test_3_body,optimize="optimal"))
print("pqrstu->qrpstu")
tprint(test_3_body - np.einsum("pqrstu->qrpstu",test_3_body,optimize="optimal"))
print("pqrstu->rpqstu")
tprint(test_3_body - np.einsum("pqrstu->rpqstu",test_3_body,optimize="optimal"))
print("pqrstu->rqpstu")
tprint(test_3_body + np.einsum("pqrstu->rqpstu",test_3_body,optimize="optimal"))
print("pqrstu->pqrsut")
tprint(test_3_body + np.einsum("pqrstu->pqrsut",test_3_body,optimize="optimal"))
print("pqrstu->prqsut")
tprint(test_3_body - np.einsum("pqrstu->prqsut",test_3_body,optimize="optimal"))
print("pqrstu->qprsut")
tprint(test_3_body - np.einsum("pqrstu->qprsut",test_3_body,optimize="optimal"))
print("pqrstu->qrpsut")
tprint(test_3_body + np.einsum("pqrstu->qrpsut",test_3_body,optimize="optimal"))
print("pqrstu->rpqsut")
tprint(test_3_body + np.einsum("pqrstu->rpqsut",test_3_body,optimize="optimal"))
print("pqrstu->rqpsut")
tprint(test_3_body - np.einsum("pqrstu->rqpsut",test_3_body,optimize="optimal"))
print("pqrstu->pqrtsu")
tprint(test_3_body + np.einsum("pqrstu->pqrtsu",test_3_body,optimize="optimal"))
print("pqrstu->prqtsu")
tprint(test_3_body - np.einsum("pqrstu->prqtsu",test_3_body,optimize="optimal"))
print("pqrstu->qprtsu")
tprint(test_3_body - np.einsum("pqrstu->qprtsu",test_3_body,optimize="optimal"))
print("pqrstu->qrptsu")
tprint(test_3_body + np.einsum("pqrstu->qrptsu",test_3_body,optimize="optimal"))
print("pqrstu->rpqtsu")
tprint(test_3_body + np.einsum("pqrstu->rpqtsu",test_3_body,optimize="optimal"))
print("pqrstu->rqptsu")
tprint(test_3_body - np.einsum("pqrstu->rqptsu",test_3_body,optimize="optimal"))
print("pqrstu->pqrtus")
tprint(test_3_body - np.einsum("pqrstu->pqrtus",test_3_body,optimize="optimal"))
print("pqrstu->prqtus")
tprint(test_3_body + np.einsum("pqrstu->prqtus",test_3_body,optimize="optimal"))
print("pqrstu->qprtus")
tprint(test_3_body + np.einsum("pqrstu->qprtus",test_3_body,optimize="optimal"))
print("pqrstu->qrptus")
tprint(test_3_body - np.einsum("pqrstu->qrptus",test_3_body,optimize="optimal"))
print("pqrstu->rpqtus")
tprint(test_3_body - np.einsum("pqrstu->rpqtus",test_3_body,optimize="optimal"))
print("pqrstu->rqptus")
tprint(test_3_body + np.einsum("pqrstu->rqptus",test_3_body,optimize="optimal"))
print("pqrstu->pqrust")
tprint(test_3_body - np.einsum("pqrstu->pqrust",test_3_body,optimize="optimal"))
print("pqrstu->prqust")
tprint(test_3_body + np.einsum("pqrstu->prqust",test_3_body,optimize="optimal"))
print("pqrstu->qprust")
tprint(test_3_body + np.einsum("pqrstu->qprust",test_3_body,optimize="optimal"))
print("pqrstu->qrpust")
tprint(test_3_body - np.einsum("pqrstu->qrpust",test_3_body,optimize="optimal"))
print("pqrstu->rpqust")
tprint(test_3_body - np.einsum("pqrstu->rpqust",test_3_body,optimize="optimal"))
print("pqrstu->rqpust")
tprint(test_3_body + np.einsum("pqrstu->rqpust",test_3_body,optimize="optimal"))
print("pqrstu->pqruts")
tprint(test_3_body + np.einsum("pqrstu->pqruts",test_3_body,optimize="optimal"))
print("pqrstu->prquts")
tprint(test_3_body - np.einsum("pqrstu->prquts",test_3_body,optimize="optimal"))
print("pqrstu->qpruts")
tprint(test_3_body - np.einsum("pqrstu->qpruts",test_3_body,optimize="optimal"))
print("pqrstu->qrputs")
tprint(test_3_body + np.einsum("pqrstu->qrputs",test_3_body,optimize="optimal"))
print("pqrstu->rpquts")
tprint(test_3_body + np.einsum("pqrstu->rpquts",test_3_body,optimize="optimal"))
print("pqrstu->rqputs")
tprint(test_3_body - np.einsum("pqrstu->rqputs",test_3_body,optimize="optimal"))
"""
def rand_t1_ext(n_a,n_act,n_orb):
	n_virt_a = n_orb-n_a
	n_virt_act_a = n_act-n_a
	n_virt_ext_a = n_orb-n_act
	test_t1a = np.zeros((n_a,n_virt_a))
	test_t1b = np.zeros((n_a,n_virt_a))
	for i in range(0,n_a):
		for A in range(0,n_virt_ext_a):
			aa = 2*np.random.rand()-1 
			bb = 2*np.random.rand()-1 
			test_t1a[i,A+n_virt_act_a] = aa 
			test_t1b[i,A+n_virt_act_a] = bb
	return test_t1a,test_t1b  

def rand_t1_ext_r(n_a,n_act,n_orb):
	n_virt_a = n_orb-n_a
	n_virt_act_a = n_act-n_a
	n_virt_ext_a = n_orb-n_act
	test_t1a = np.zeros((n_a,n_virt_a))
	test_t1b = np.zeros((n_a,n_virt_a))
	for i in range(0,n_a):
		for A in range(0,n_virt_ext_a):
			aa = 2*np.random.rand()-1 
			test_t1a[i,A+n_virt_act_a] = aa 
			test_t1b[i,A+n_virt_act_a] = aa 
	return test_t1a,test_t1b

def rand_t2_ext(n_a,n_act,n_orb):
	n_virt_a = n_orb-n_a
	n_virt_act_a = n_act-n_a
	n_virt_ext_a = n_orb-n_act
	test_t2aa = np.zeros((n_a,n_a,n_virt_a,n_virt_a))
	test_t2ab = np.zeros((n_a,n_a,n_virt_a,n_virt_a))
	test_t2bb = np.zeros((n_a,n_a,n_virt_a,n_virt_a))
	# t_{ia,ja}^{aa,Ba}/t_{ib,jb}^{ab,Bb} 
	for i in range(0,n_a):
		for j in range(i+1,n_a):
			for a in range(0,n_virt_act_a):
				for B in range(0,n_virt_ext_a):
					aa = 2*np.random.rand()-1 
					bb = 2*np.random.rand()-1 
					test_t2aa[i,j,a,B+n_virt_act_a] =  aa 
					test_t2aa[i,j,B+n_virt_act_a,a] = -aa
					test_t2aa[j,i,a,B+n_virt_act_a] = -aa 
					test_t2aa[j,i,B+n_virt_act_a,a] =  aa

					test_t2bb[i,j,a,B+n_virt_act_a] =  bb 
					test_t2bb[i,j,B+n_virt_act_a,a] = -bb
					test_t2bb[j,i,a,B+n_virt_act_a] = -bb 
					test_t2bb[j,i,B+n_virt_act_a,a] =  bb
	# t_{ia,ja}^{Aa,Ba}/t_{ib,jb}^{Ab,Bb}
	for i in range(0,n_a):
		for j in range(i+1,n_a):
			for A in range(0,n_virt_ext_a):
				for B in range(A+1,n_virt_ext_a):
					aa = 2*np.random.rand()-1
					bb = 2*np.random.rand()-1 
					test_t2aa[i,j,A+n_virt_act_a,B+n_virt_act_a] =  aa
					test_t2aa[i,j,B+n_virt_act_a,A+n_virt_act_a] = -aa
					test_t2aa[j,i,A+n_virt_act_a,B+n_virt_act_a] = -aa
					test_t2aa[j,i,B+n_virt_act_a,A+n_virt_act_a] =  aa

					test_t2bb[i,j,A+n_virt_act_a,B+n_virt_act_a] =  bb
					test_t2bb[i,j,B+n_virt_act_a,A+n_virt_act_a] = -bb
					test_t2bb[j,i,A+n_virt_act_a,B+n_virt_act_a] = -bb
					test_t2bb[j,i,B+n_virt_act_a,A+n_virt_act_a] =  bb
	# t_{ia,jb}^{Aa,bb}/t_{ia,jb}^{aa,Bb}
	for i in range(0,n_a):
		for j in range(0,n_a):
			for A in range(0,n_virt_ext_a):
				for b in range(0,n_virt_act_a):
					aa = 2*np.random.rand()-1 
					bb = 2*np.random.rand()-1 
					test_t2ab[i,j,A+n_virt_act_a,b] = aa 
					test_t2ab[i,j,b,A+n_virt_act_a] = bb 
	# t_{ia,jb}^{Aa,Bb}
	for i in range(0,n_a):
		for j in range(0,n_a):
			for A in range(0,n_virt_ext_a):
				for B in range(0,n_virt_ext_a):
					aa = 2*np.random.rand()-1 
					test_t2ab[i,j,A+n_virt_act_a,B+n_virt_act_a] = aa 
	return test_t2aa,test_t2ab,test_t2bb

def rand_t2_ext_r(n_a,n_act,n_orb):
	n_virt_a = n_orb-n_a
	n_virt_act_a = n_act-n_a
	n_virt_ext_a = n_orb-n_act
	test_t2aa = np.zeros((n_a,n_a,n_virt_a,n_virt_a))
	test_t2ab = np.zeros((n_a,n_a,n_virt_a,n_virt_a))
	test_t2bb = np.zeros((n_a,n_a,n_virt_a,n_virt_a))
	# t_{ia,ja}^{aa,Ba}/t_{ib,jb}^{ab,Bb} 
	for i in range(0,n_a):
		for j in range(i+1,n_a):
			for a in range(0,n_virt_act_a):
				for B in range(0,n_virt_ext_a):
					aa = 2*np.random.rand()-1 
					test_t2aa[i,j,a,B+n_virt_act_a] =  aa 
					test_t2aa[i,j,B+n_virt_act_a,a] = -aa
					test_t2aa[j,i,a,B+n_virt_act_a] = -aa 
					test_t2aa[j,i,B+n_virt_act_a,a] =  aa

					test_t2bb[i,j,a,B+n_virt_act_a] =  aa 
					test_t2bb[i,j,B+n_virt_act_a,a] = -aa
					test_t2bb[j,i,a,B+n_virt_act_a] = -aa 
					test_t2bb[j,i,B+n_virt_act_a,a] =  aa
	# t_{ia,ja}^{Aa,Ba}/t_{ib,jb}^{Ab,Bb}
	for i in range(0,n_a):
		for j in range(i+1,n_a):
			for A in range(0,n_virt_ext_a):
				for B in range(A+1,n_virt_ext_a):
					aa = 2*np.random.rand()-1 
					test_t2aa[i,j,A+n_virt_act_a,B+n_virt_act_a] =  aa
					test_t2aa[i,j,B+n_virt_act_a,A+n_virt_act_a] = -aa
					test_t2aa[j,i,A+n_virt_act_a,B+n_virt_act_a] = -aa
					test_t2aa[j,i,B+n_virt_act_a,A+n_virt_act_a] =  aa

					test_t2bb[i,j,A+n_virt_act_a,B+n_virt_act_a] =  aa
					test_t2bb[i,j,B+n_virt_act_a,A+n_virt_act_a] = -aa
					test_t2bb[j,i,A+n_virt_act_a,B+n_virt_act_a] = -aa
					test_t2bb[j,i,B+n_virt_act_a,A+n_virt_act_a] =  aa
	# t_{ia,jb}^{Aa,bb}/t_{ia,jb}^{aa,Bb}
	for i in range(0,n_a):
		for j in range(0,n_a):
			for A in range(0,n_virt_ext_a):
				for b in range(0,n_virt_int_a):
					aa = 2*np.random.rand()-1 
					test_t2ab[i,j,A+n_virt_act_a,b] = aa 
					test_t2ab[j,i,b,A+n_virt_act_a] = aa 
	# t_{ia,jb}^{Aa,Bb}
	for i in range(0,n_a):
		for j in range(0,n_a):
			for A in range(0,n_virt_ext_a):
				for B in range(0,n_virt_ext_a):
					aa = 2*np.random.rand()-1 
					test_t2ab[i,j,A+n_virt_act_a,B+n_virt_act_a] = aa 
					test_t2ab[i,j,A+n_virt]
	return test_t2aa,test_t2ab,test_t2bb


