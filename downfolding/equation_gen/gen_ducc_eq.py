import wicked as w  
from IPython.display import display, Math, Latex

def latex(expr):
	"""Function to render any object that has a member latex() function"""
	display(Math(expr.latex()))

def print_manybody_equations(label, op, minrank=0, maxrank=8):
    expr = wt.contract(op, minrank=minrank, maxrank=maxrank)
    mbeq = expr.to_manybody_equations(label)

    for key in mbeq.keys():
        # The key looks like '|' (scalar), 'o|o', 'O|v', etc.
        # Remove the pipe to just look at the raw space characters
        idx_str = key.replace('|', '')
        
        # 1. Filter out terms that touch the external spaces (O or V)
        # Since external spaces are uppercase, we check for any uppercase letters
        if any(c.isupper() for c in idx_str):
            continue
            
        for eq in mbeq[key]:
            # Get the default wicked string (e.g. "fs1oo += 1.0 * np.einsum(...)")
            einsum_str = eq.compile('einsum')
            
            # 2. Reformat the left-hand side into dictionary notation
            if ' += ' in einsum_str:
                lhs, rhs = einsum_str.split(' += ', 1)
                
                # Extract the spaces by removing the prefix label
                # If label is "fs1" and lhs is "fs1oo", suffix is "oo"
                suffix = lhs[len(label):]
                
                # If the suffix is empty, it's the scalar part ("c")
                if suffix == "":
                    new_lhs = f'{label}["c"]'
                else:
                    new_lhs = f'{label}["{suffix}"]'
                    
                print(f'{new_lhs} += {rhs}')

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
print_manybody_equations('fs1', Fs1, minrank=0, maxrank=4)

# expr = wt.contract(Fs1,minrank=0,maxrank=4)
# mbeq = expr.to_manybody_equations('fs1')
# for key in mbeq.keys():
# 	for eq in mbeq[key]:
# 		print(eq.compile('einsum'))

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
print_manybody_equations('fs2', Fs2, minrank=0, maxrank=4)


# [W,s1_ext]
print("[W,s1_ext]")
Ws1 = w.commutator(W,s1_ext)
print_manybody_equations('ws1', Ws1, minrank=0, maxrank=4)


# [W,s2_ext]
print("[W,s2_ext]")
Ws2 = w.commutator(W,s2_ext)
print_manybody_equations('ws2', Ws2, minrank=0, maxrank=4)


# [[F,s1_ext],s1_ext]
print("[[F,s1_ext],s1_ext]")
Fs1s1 = w.commutator(F,s1_ext,s1_ext)
print_manybody_equations('fs1s1', Fs1s1, minrank=0, maxrank=4)


# [[F,s1_ext],s2_ext]
print("[[F,s1_ext],s2_ext]")
Fs1s2 = w.commutator(F,s1_ext,s2_ext)
print_manybody_equations('fs1s2', Fs1s2, minrank=0, maxrank=4)


# [[F,s2_ext],s1_ext]
print("[[F,s2_ext],s1_ext]")
Fs2s1 = w.commutator(F,s2_ext,s1_ext)
print_manybody_equations('fs2s1', Fs2s1, minrank=0, maxrank=4)


# [[F,s2_ext],s2_ext]
print("[[F,s2_ext],s2_ext]")
Fs2s2 = w.commutator(F,s2_ext,s2_ext)
print_manybody_equations('fs2s2', Fs2s2, minrank=0, maxrank=4)


# [[W,s1_ext],s1_ext]
print("[[W,s1_ext],s1_ext]")
Ws1s1 = w.commutator(W,s1_ext,s1_ext)
print_manybody_equations('ws1s1', Ws1s1, minrank=0, maxrank=4)


# [[W,s1_ext],s2_ext]
print("[[W,s1_ext],s2_ext]")
Ws1s2 = w.commutator(W,s1_ext,s2_ext)
print_manybody_equations('ws1s2', Ws1s2, minrank=0, maxrank=4)


# [[W,s2_ext],s1_ext]
print("[[W,s2_ext],s1_ext]")
Ws2s1 = w.commutator(W,s2_ext,s1_ext)
print_manybody_equations('ws2s1', Ws2s1, minrank=0, maxrank=4)


# [[W,s2_ext],s2_ext]
print("[[W,s2_ext],s2_ext]")
Ws2s2 = w.commutator(W,s2_ext,s2_ext)
print_manybody_equations('ws2s2', Ws2s2, minrank=0, maxrank=4)


# [[[F,s1_ext],s1_ext],s1_ext]
print("[[[F,s1_ext],s1_ext],s1_ext]")
Fs1s1s1 = w.commutator(F,s1_ext,s1_ext,s1_ext)
print_manybody_equations('fs1s1s1', Fs1s1s1, minrank=0, maxrank=4)


# [[[F,s1_ext],s1_ext],s2_ext]
print("[[[F,s1_ext],s1_ext],s2_ext]")
Fs1s1s2 = w.commutator(F,s1_ext,s1_ext,s2_ext)
print_manybody_equations('fs1s1s2', Fs1s1s2, minrank=0, maxrank=4)


# [[[F,s1_ext],s2_ext],s1_ext]
print("[[[F,s1_ext],s2_ext],s1_ext]")
Fs1s2s1 = w.commutator(F,s1_ext,s2_ext,s1_ext)
print_manybody_equations('fs1s2s1', Fs1s2s1, minrank=0, maxrank=4)

# [[[F,s1_ext],s2_ext],s2_ext]
print("[[[F,s1_ext],s2_ext],s2_ext]")
Fs1s2s2 = w.commutator(F,s1_ext,s2_ext,s2_ext)
print_manybody_equations('fs1s2s2', Fs1s2s2, minrank=0, maxrank=4)

# [[[F,s2_ext],s1_ext],s1_ext]
print("[[[F,s2_ext],s1_ext],s1_ext]")
Fs2s1s1 = w.commutator(F,s2_ext,s1_ext,s1_ext)
print_manybody_equations('fs2s1s1', Fs2s1s1, minrank=0, maxrank=4)

# [[[F,s2_ext],s1_ext],s2_ext]
print("[[[F,s2_ext],s1_ext],s2_ext]")
Fs2s1s2 = w.commutator(F,s2_ext,s1_ext,s2_ext)
print_manybody_equations('fs2s1s2', Fs2s1s2, minrank=0, maxrank=4)

# [[[F,s2_ext],s2_ext],s1_ext]
print("[[[F,s2_ext],s2_ext],s1_ext]")
Fs2s2s1 = w.commutator(F,s2_ext,s2_ext,s1_ext)
print_manybody_equations('fs2s2s1', Fs2s2s1, minrank=0, maxrank=4)

# [[[F,s2_ext],s2_ext],s2_ext]
print("[[[F,s2_ext],s2_ext],s2_ext]")
Fs2s2s2 = w.commutator(F,s2_ext,s2_ext,s2_ext)
print_manybody_equations('fs2s2s2', Fs2s2s2, minrank=0, maxrank=4)