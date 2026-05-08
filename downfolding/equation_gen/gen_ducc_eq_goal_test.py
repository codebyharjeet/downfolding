import wicked as w
from itertools import product
from IPython.display import display, Math, Latex

def latex(expr):
    """Render any object that has a latex() member."""
    display(Math(expr.latex()))

# ---------------------------------------------------------------------
# Orbital spaces:
#   O = external/core occupied
#   o = active occupied
#   v = active virtual
#   V = external virtual
# ---------------------------------------------------------------------
w.reset_space()

# w.add_space("O", "fermion", "occupied",   ['I','J','K','L','M','N','P','Q'])
w.add_space("o", "fermion", "occupied",   ['i','j','k','l','m','n','p','q'])
w.add_space("v", "fermion", "unoccupied", ['a','b','c','d','e','g','x','y'])
w.add_space("V", "fermion", "unoccupied", ['A','B','C','D','E','G','X','Y'])

wt = w.WickTheorem()

# All orbital spaces included in the Hamiltonian
all_spaces = "ovV"

# ---------------------------------------------------------------------
# Fock and two-body interaction operators
# ---------------------------------------------------------------------
F = w.utils.gen_op('f', 1, all_spaces, all_spaces)
W = w.utils.gen_op('v', 2, all_spaces, all_spaces)

# ---------------------------------------------------------------------
# External anti-Hermitian amplitudes
#
# Build all occupied -> virtual excitations:
#     occupied:   O, o
#     virtual:    v, V
#
# Then remove the purely active pieces:
#     t1: v+ o
#     t2: v+ v+ o o
#
# Therefore s_ext contains every amplitude touching at least one
# external index O or V.
# ---------------------------------------------------------------------

# Singles:
# Conceptually includes V+ o, v+ O, V+ O and adjoints.
T1_all = w.utils.gen_op('t1', 1, 'vV', 'o', diagonal=False)
T1_act = w.op('t1', ['v+ o'])
T1_ext = T1_all - T1_act
s1_ext = T1_ext - T1_ext.adjoint()

# Doubles:
# Conceptually includes all {v,V}{v,V} <- {O,o}{O,o}
# blocks except the purely active v v <- o o block.
T2_all = w.utils.gen_op('t2', 2, 'vV', 'o', diagonal=False)
T2_act = w.op('t2', ['v+ v+ o o'])
T2_ext = T2_all - T2_act
s2_ext = T2_ext - T2_ext.adjoint()

# Optional checks
# print("s1_ext")
# print(s1_ext)
# print("s2_ext")
# print(s2_ext)

# ---------------------------------------------------------------------
# Helpers for printing many-body equations
# ---------------------------------------------------------------------
S = {
    "s1": s1_ext,
    "s2": s2_ext,
}

def pretty_nested_commutator(base, seq):
    """
    Example:
        base='F', seq=('s1','s2') -> [[F,s1_ext],s2_ext]
    """
    text = base
    for name in seq:
        text = f"[{text},{name}_ext]"
    return text

def print_manybody_equations(label, op, minrank=0, maxrank=8):
    expr = wt.contract(op, minrank=minrank, maxrank=maxrank)
    mbeq = expr.to_manybody_equations(label)

    for key in mbeq.keys():
        for eq in mbeq[key]:
            print(eq.compile('einsum'))

def print_manybody_equations_1(label, op, minrank=0, maxrank=8):
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

def run_commutator(h_code, h_print, H, seq, maxrank):
    """
    h_code  : label prefix used for tensor/equation names, e.g. 'f' or 'w'
    h_print : printed operator name, e.g. 'F' or 'W'
    H       : Wicked operator
    seq     : tuple such as ('s1', 's2')
    maxrank : max rank for Wick contraction
    """
    label = h_code + ''.join(seq)
    print(pretty_nested_commutator(h_print, seq))

    comm = w.commutator(H, *[S[name] for name in seq])
    print_manybody_equations_1(label, comm, minrank=0, maxrank=maxrank)




# First order:
#   [F,s1], [F,s2], [W,s1], [W,s2]
for h_code, h_print, H in [
    ('f', 'F', F),
    ('w', 'W', W),
]:
    for seq in [('s1',), ('s2',)]:
        run_commutator(h_code, h_print, H, seq, maxrank=4)

# Second order:
#   [[F,si],sj]
#   [[W,si],sj]
for h_code, h_print, H, maxrank in [
    ('f', 'F', F, 4),
    ('w', 'W', W, 4),
]:
    for seq in product(('s1', 's2'), repeat=2):
        run_commutator(h_code, h_print, H, seq, maxrank=maxrank)

# Third order with F:
#   [[[F,si],sj],sk]
for seq in product(('s1', 's2'), repeat=3):
    run_commutator('f', 'F', F, seq, maxrank=4)