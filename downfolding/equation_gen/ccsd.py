import wicked as w
import numpy as np
from IPython.display import display, Math, Latex

def generate_equation(mbeq, nocc, nvir):
    res_sym = f"R{'o' * nocc}{'v' * nvir}"
    code = [
        f"def evaluate_residual_{nocc}_{nvir}(H,T):",
        "    # contributions to the residual",
    ]

    if nocc + nvir == 0:
        code.append("    R = 0.0")
    else:
        dims = ",".join(["nocc"] * nocc + ["nvir"] * nvir)
        code.append(f"    {res_sym} = np.zeros(({dims}))")

    for eq in mbeq["o" * nocc + "|" + "v" * nvir]:
        contraction = eq.compile("einsum")
        code.append(f"    {contraction}")

    code.append(f"    return {res_sym}")
    funct = "\n".join(code)
    return funct

w.reset_space()
w.add_space("o", "fermion", "occupied", ["i", "j", "k", "l", "m", "n"])
w.add_space("v", "fermion", "unoccupied", ["a", "b", "c", "d", "e", "f"])
wt = w.WickTheorem()


Top = w.op("T", ["v+ o", "v+ v+ o o"])

Hop = w.utils.gen_op("F", 1, "ov", "ov") + w.utils.gen_op("V", 2, "ov", "ov")
# Vop = w.utils.gen_op("V", 2, "ov", "ov")
# the similarity-transformed Hamiltonian truncated to the four-nested commutator term
Hbar = w.bch_series(Hop, Top, 4)

wt = w.WickTheorem()
# expr = wt.contract(w.rational(1), Hbar, 0, 4)
expr = wt.contract(Hbar, 0, 4)
mbeq = expr.to_manybody_equation("R")

energy_eq = generate_equation(mbeq, 0, 0)
exec(energy_eq)
print(energy_eq)

t1_eq = generate_equation(mbeq, 1, 1)
exec(t1_eq)
print(t1_eq)

t2_eq = generate_equation(mbeq, 2, 2)
exec(t2_eq)
print(t2_eq)
