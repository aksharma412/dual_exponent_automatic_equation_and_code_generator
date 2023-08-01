from sympy.physics.secondquant import (AntiSymmetricTensor, Commutator, PermutationOperator, evaluate_deltas, NO, Fd, F, wicks, substitute_dummies, simplify_index_permutations)
from sympy import (latex, Dummy, Rational, symbols)

dummy_dict = {
    'above' : 'cdefgh',
    'below' : 'klmno',
    'general' : 'pqrstu'}

i = symbols('i', below_fermi=True, cls=Dummy)
a = symbols('a', above_fermi=True, cls=Dummy)
j = symbols('j', below_fermi=True, cls=Dummy)
b = symbols('b', above_fermi=True, cls=Dummy)
p, q, r, s = symbols('p,q,r,s', cls=Dummy)

# Building the normal ordered Hamiltonian
fock = AntiSymmetricTensor('f', (p,), (q,))
pr = NO(Fd(p)*F(q))
V = AntiSymmetricTensor('v',(p,q),(r,s))
pqsr = NO(Fd(p)*Fd(q)*F(s)*F(r))
H = fock*pr + Rational(1,4)*V*pqsr

# Building the T amplitudes, T1 and T2 are made in the conventional way whereas we use a T2 decomposition for T3
def get_T():
    i, j = symbols('i,j', below_fermi=True, cls=Dummy)
    a, b = symbols('a,b', above_fermi=True, cls=Dummy)
    t1 = AntiSymmetricTensor('t', (a,), (i,))*NO(Fd(a)*F(i))
    t2 = Rational(1,4)*AntiSymmetricTensor('t', (a,b), (i,j))*NO(Fd(a)*Fd(b)*F(j)*F(i))
    #t3 = Rational(1,36)*AntiSymmetricTensor('t', (a,b,c), (i,j,k))*NO(Fd(a)*Fd(b)*Fd(c)*F(k)*F(j)*F(i))
    return t1 + t2

# Building one of the T2 like intermediates for T3
def get_S():
    i, j, k,l = symbols('i,j,k,l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a,b,c,d', above_fermi=True, cls=Dummy)
    # s_vooo = AntiSymmetricTensor('S', (a,l), (i,j))*NO(Fd(a)*Fd(l)*F(j)*F(i))
    # s_vvvo = AntiSymmetricTensor('S', (a,b), (i,d))*NO(Fd(a)*Fd(b)*F(d)*F(i))
    s_vooo = AntiSymmetricTensor('s', (a,l), (i,j))*NO(Fd(a)*Fd(l)*F(j)*F(i))
    s_vvvo = AntiSymmetricTensor('s', (a,b), (i,d))*NO(Fd(a)*Fd(b)*F(d)*F(i))
    
    return s_vooo, s_vvvo


def get_T2_ld():
    i, j, k, l = symbols('i,j,k,l', below_fermi=True, cls=Dummy)
    a, b, c, d = symbols('a,b,c,d', above_fermi=True, cls=Dummy)
    t2l = AntiSymmetricTensor('t', (b,c), (l,k))*NO(Fd(b)*Fd(c)*F(k)*F(l))
    t2d = AntiSymmetricTensor('t', (d,c), (j,k))*NO(Fd(d)*Fd(c)*F(k)*F(j))
    return t2l, t2d

def get_ST():
    s_vooo, s_vvvo = get_S()
    T2_l, T2_d = get_T2_ld()
    C = Commutator
    tmp = wicks(C(s_vooo, T2_l)) + wicks(C(s_vvvo, T2_d))
    tmp = evaluate_deltas(tmp)
    tmp = substitute_dummies(tmp, new_indices=True, pretty_indices = dummy_dict)
    return tmp


C = Commutator
T3_new = get_ST()
T = get_T() + T3_new
comm1 = wicks(C(H, T))
comm1 = evaluate_deltas(comm1)
comm1 = substitute_dummies(comm1)

T3_new = get_ST()
T = get_T() + T3_new
comm2 = wicks(C(comm1, T))
comm2 = evaluate_deltas(comm2)
comm2 = substitute_dummies(comm2)

T3_new = get_ST()
T = get_T() + T3_new
comm3 = wicks(C(comm2, T))
comm3 = evaluate_deltas(comm3)
comm3 = substitute_dummies(comm3)

T3_new = get_ST()
T = get_T() + T3_new
comm4 = wicks(C(comm3, T))
comm4 = evaluate_deltas(comm4)
comm4 = substitute_dummies(comm4)

eq = H + comm1 + comm2/2 + comm3/6 + comm4/24
eq = eq.expand()
eq = evaluate_deltas(eq)
eq = substitute_dummies(eq, new_indices=True,
        pretty_indices=dummy_dict)



# i, j, k, l  = symbols('i,j,k,l', below_fermi=True)
# a, b, c, d = symbols('a,b,c,d', above_fermi=True)
# energy = wicks(eq, simplify_dummies=True,
#         keep_only_fully_contracted=True)
# print(latex(energy))
# print("\nEnd of Energy")

# eqT1 = wicks(NO(Fd(i)*F(a))*eq, simplify_dummies=True, 
#               keep_only_fully_contracted=True, simplify_kronecker_deltas=True)

# eqT2 = wicks(NO(Fd(i)*Fd(j)*F(b)*F(a))*eq, simplify_dummies=True, 
#               keep_only_fully_contracted=True, simplify_kronecker_deltas=True)

# eqT3 = wicks(NO(Fd(i)*Fd(j)*Fd(k)*F(c)*F(b)*F(a))*eq, simplify_dummies=True, 
#               keep_only_fully_contracted=True, simplify_kronecker_deltas=True)

# temp = get_ST()
# eqT3_new = wicks(temp*eq, simplify_dummies=True, 
#               keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
