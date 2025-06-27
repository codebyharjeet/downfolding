import wicked as w  
from IPython.display import display, Math, Latex

def latex(expr):
	"""Function to render any object that has a member latex() function"""
	display(Math(expr.latex()))

# orbital spaces occ, virt_int, virt_ext
w.reset_space()
w.add_space("o", "fermion", "occupied", ['i','j','k','l','m','n','p','q'])
w.add_space("v", "fermion", "unoccupied", ['a','b','c','d','e','g','x','y']) 
w.add_space("V", "fermion", "unoccupied", ['A','B','C','D','E','G','X','Y'])
wt = w.WickTheorem()

# external amplitudes
s1_ext = w.op('t1',['V+ o']) - w.op('t1',['o+ V'])
# print("s1_ext")
# print(s1_ext)

s2_ext =  w.op('t2',['v+ V+ o o']) - w.op('t2',['o+ o+ V v'])
s2_ext += w.op('t2',['V+ V+ o o']) - w.op('t2',['o+ o+ V V'])
# print("s2_ext")
# print(s2_ext)

# Fock operator
F = w.utils.gen_op('f',1,'ovV','ovV')
# print("Fock")
# print(F)

# Interaction operator
W = w.utils.gen_op('v',2,'ovV','ovV')
# print("Interaction")
# print(V)

# [F,s1_ext]
print("[F,s1_ext]")
Fs1 = w.commutator(F,s1_ext)
expr = wt.contract(Fs1,minrank=0,maxrank=6)
mbeq = expr.to_manybody_equations('fs1')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))
# for eq in mbeq['|']:
# 	print(eq.compile('einsum'))

# for eq in mbeq['o|o']:
# 	print(eq.compile('einsum'))

# for eq in mbeq['o|v']:
# 	print(eq.compile('einsum'))

# for eq in mbeq['v|o']:
# 	print(eq.compile('einsum'))

# [F,s2_ext]
print("[F,s2_ext]")
Fs2 = w.commutator(F,s2_ext)
expr = wt.contract(Fs2,minrank=0,maxrank=6)
mbeq = expr.to_manybody_equations('fs2')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [W,s1_ext]
print("[W,s1_ext]")
Ws1 = w.commutator(W,s1_ext)
expr = wt.contract(Ws1,minrank=0,maxrank=6)
mbeq = expr.to_manybody_equations('ws1')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [W,s2_ext]
print("[W,s2_ext]")
Ws2 = w.commutator(W,s2_ext)
expr = wt.contract(Ws2,minrank=0,maxrank=6)
mbeq = expr.to_manybody_equations('ws2')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[F,s1_ext],s1_ext]
print("[[F,s1_ext],s1_ext]")
Fs1s1 = w.commutator(F,s1_ext,s1_ext)
expr = wt.contract(Fs1s1,minrank=0,maxrank=6)
mbeq = expr.to_manybody_equations('fs1s1')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[F,s1_ext],s2_ext]
print("[[F,s1_ext],s2_ext]")
Fs1s2 = w.commutator(F,s1_ext,s2_ext)
expr = wt.contract(Fs1s2,minrank=0,maxrank=6)
mbeq = expr.to_manybody_equations('fs1s2')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[F,s2_ext],s1_ext]
print("[[F,s2_ext],s1_ext]")
Fs2s1 = w.commutator(F,s2_ext,s1_ext)
expr = wt.contract(Fs2s1,minrank=0,maxrank=6)
mbeq = expr.to_manybody_equations('fs2s1')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[F,s2_ext],s2_ext]
print("[[F,s2_ext],s2_ext]")
Fs2s2 = w.commutator(F,s2_ext,s2_ext)
expr = wt.contract(Fs2s2,minrank=0,maxrank=6)
mbeq = expr.to_manybody_equations('fs2s2')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[W,s1_ext],s1_ext]
print("[[W,s1_ext],s1_ext]")
Ws1s1 = w.commutator(W,s1_ext,s1_ext)
expr = wt.contract(Ws1s1,minrank=0,maxrank=8)
mbeq = expr.to_manybody_equations('ws1s1')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[W,s1_ext],s2_ext]
print("[[W,s1_ext],s2_ext]")
Ws1s2 = w.commutator(W,s1_ext,s2_ext)
expr = wt.contract(Ws1s2,minrank=0,maxrank=8)
mbeq = expr.to_manybody_equations('ws1s2')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[W,s2_ext],s1_ext]
print("[[W,s2_ext],s1_ext]")
Ws2s1 = w.commutator(W,s2_ext,s1_ext)
expr = wt.contract(Ws2s1,minrank=0,maxrank=8)
mbeq = expr.to_manybody_equations('ws2s1')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[W,s2_ext],s2_ext]
print("[[W,s2_ext],s2_ext]")
Ws2s2 = w.commutator(W,s2_ext,s2_ext)
expr = wt.contract(Ws2s2,minrank=0,maxrank=8)
mbeq = expr.to_manybody_equations('ws2s2')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[[F,s1_ext],s1_ext],s1_ext]
print("[[[F,s1_ext],s1_ext],s1_ext]")
Fs1s1s1 = w.commutator(F,s1_ext,s1_ext,s1_ext)
expr = wt.contract(Fs1s1s1,minrank=0,maxrank=8)
mbeq = expr.to_manybody_equations('fs1s1s1')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[[F,s1_ext],s1_ext],s2_ext]
print("[[[F,s1_ext],s1_ext],s2_ext]")
Fs1s1s2 = w.commutator(F,s1_ext,s1_ext,s2_ext)
expr = wt.contract(Fs1s1s2,minrank=0,maxrank=8)
mbeq = expr.to_manybody_equations('fs1s1s2')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[[F,s1_ext],s2_ext],s1_ext]
print("[[[F,s1_ext],s2_ext],s1_ext]")
Fs1s2s1 = w.commutator(F,s1_ext,s2_ext,s1_ext)
expr = wt.contract(Fs1s2s1,minrank=0,maxrank=8)
mbeq = expr.to_manybody_equations('fs1s2s1')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[[F,s1_ext],s2_ext],s2_ext]
print("[[[F,s1_ext],s2_ext],s2_ext]")
Fs1s2s2 = w.commutator(F,s1_ext,s2_ext,s2_ext)
expr = wt.contract(Fs1s2s2,minrank=0,maxrank=8)
mbeq = expr.to_manybody_equations('fs1s2s2')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[[F,s2_ext],s1_ext],s1_ext]
print("[[[F,s2_ext],s1_ext],s1_ext]")
Fs2s1s1 = w.commutator(F,s2_ext,s1_ext,s1_ext)
expr = wt.contract(Fs2s1s1,minrank=0,maxrank=8)
mbeq = expr.to_manybody_equations('fs2s1s1')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[[F,s2_ext],s1_ext],s2_ext]
print("[[[F,s2_ext],s1_ext],s2_ext]")
Fs2s1s2 = w.commutator(F,s2_ext,s1_ext,s2_ext)
expr = wt.contract(Fs2s1s2,minrank=0,maxrank=8)
mbeq = expr.to_manybody_equations('fs2s1s2')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[[F,s2_ext],s2_ext],s1_ext]
print("[[[F,s2_ext],s2_ext],s1_ext]")
Fs2s2s1 = w.commutator(F,s2_ext,s2_ext,s1_ext)
expr = wt.contract(Fs2s2s1,minrank=0,maxrank=8)
mbeq = expr.to_manybody_equations('fs2s2s1')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))

# [[[F,s2_ext],s2_ext],s2_ext]
print("[[[F,s2_ext],s2_ext],s2_ext]")
Fs2s2s2 = w.commutator(F,s2_ext,s2_ext,s2_ext)
expr = wt.contract(Fs2s2s2,minrank=0,maxrank=8)
mbeq = expr.to_manybody_equations('fs2s2s2')
for key in mbeq.keys():
	for eq in mbeq[key]:
		print(eq.compile('einsum'))