# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:30:31 2023

@author: Aparna K
"""

# -*- coding: utf-8 -*-
from sympy import (symbols, Rational, latex, Dummy)
import re
from sympy.physics.secondquant import (AntiSymmetricTensor, wicks,
        F, Fd, NO, evaluate_deltas, substitute_dummies, Commutator,
        simplify_index_permutations, PermutationOperator)


my_tensors = {
    't_oovv' : 'T2',
    't_ov' : 'T1',
    's_oovo' : 'So',
    's_ovvv' : 'Sv',
    'v_oooo' : 'Voooo',
    'v_ooov' : 'Vooov',
    'v_oovv' : 'Voovv',
    'v_ovov' : 'Vovov',
    'v_ovvv' : 'Vovvv',
    'v_vvvv' : 'Vvvvv',
    'f_oo' : 'fock_oo',
    'f_vv' : 'fock_vv',
    'f_ov' : 'fock_ov',
    'I1' : 'I1',
    'I2' : 'I2',
    'I3' : 'I3',
    'I4' : 'I4'
}

target = {
    'energy' : ' ',
    'T1' : 'ia',
    'T2' : 'ijab',
    'T3' : 'ijkabc'
}


def get_what(type_of_info):
    pretty_dummies_dict = {
        'above': 'cdefgh',
        'below': 'klmno',
        'general': 'pqrstu'
    }

    i = symbols('i', below_fermi=True, cls=Dummy)
    a = symbols('a', above_fermi=True, cls=Dummy)
    j = symbols('j', below_fermi=True, cls=Dummy)
    b = symbols('b', above_fermi=True, cls=Dummy)
    p, q, r, s = symbols('p,q,r,s', cls=Dummy)

    fock = AntiSymmetricTensor('f', (p,), (q,))
    pr = NO(Fd(p)*F(q))
    V = AntiSymmetricTensor('v',(p,q),(r,s))
    pqsr = NO(Fd(p)*Fd(q)*F(s)*F(r))
    H = fock*pr + Rational(1,4)*V*pqsr
    H

    def get_T():
        i, j = symbols('i,j', below_fermi=True, cls=Dummy)
        a, b = symbols('a,b', above_fermi=True, cls=Dummy)
        t_ai = AntiSymmetricTensor('t', (a,), (i,))*NO(Fd(a)*F(i))
        t_abij = Rational(1,4)*AntiSymmetricTensor('t', (a,b), (i,j))*NO(Fd(a)*Fd(b)*F(j)*F(i))
        return t_ai + t_abij

    get_T()

    C = Commutator
    T = get_T()
    print("commutator 1...")
    comm1 = wicks(C(H, T))
    comm1 = evaluate_deltas(comm1)
    comm1 = substitute_dummies(comm1)

    T = get_T()
    print("commutator 2...")
    comm2 = wicks(C(comm1, T))
    comm2 = evaluate_deltas(comm2)
    comm2 = substitute_dummies(comm2)

    T = get_T()
    print("commutator 3...")
    comm3 = wicks(C(comm2, T))
    comm3 = evaluate_deltas(comm3)
    comm3 = substitute_dummies(comm3)

    T = get_T()
    print("commutator 4...")
    comm4 = wicks(C(comm3, T))
    comm4 = evaluate_deltas(comm4)
    comm4 = substitute_dummies(comm4)

    eq = H + comm1 + comm2/2 + comm3/6 + comm4/24
    eq = eq.expand()
    eq = evaluate_deltas(eq)
    eq = substitute_dummies(eq, new_indices=True,
            pretty_indices=pretty_dummies_dict)

    i, j, k, l = symbols('i,j,k,l', below_fermi=True)
    a, b, c, d = symbols('a,b,c,d', above_fermi=True)
    energy = wicks(eq, simplify_dummies=True,
            keep_only_fully_contracted=True)
    
    eqT1 = wicks(NO(Fd(i)*F(a))*eq, simplify_dummies=True, 
                 keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
    
    eqT2 = wicks(NO(Fd(i)*Fd(j)*F(b)*F(a))*eq, simplify_dummies=True, 
                 keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
    if type_of_info == 'energy':
        return energy   
    elif type_of_info == 'eqT1':
        return eqT1
    elif type_of_info == 'eqT2':
        return eqT2
    else:
        raise NameError("Enter a valid equation type")            
            
            
def calc_rank(splitted_out):
    rank = []
    for term in splitted_out:
        wanted = re.search(r'\(.*?\)', term)
        wanted = wanted[0][1:-1]
        rank.append(len(wanted.split(',')[1]))
    return rank
    
    
def space_identifier(splitted_out, tensor):
    spaces = []
    for term in splitted_out:
        wanted = re.search(r'\(.*?\)', term)
        wanted = wanted[0][1:-1]
        indx = wanted.split(',')
        space = ''
        tmp = 0
        for p in indx:
            if len(p) == 2:
                if p[0] <= 'h':
                    space += 'v'
                else:
                    space += 'o'
                if p[1] <= 'h':
                    space += 'v'
                else:
                    space += 'o'
            else:   
                if p <= 'h':
                    space += 'v'
                else:
                    space += 'o'
        spaces.append(space)
        
    for t in range(len(tensor)):
        if tensor[t] == 't':
            mid = len(spaces[t])/2
            mid = int(mid)
            spaces[t] = spaces[t][mid:] + spaces[t][:mid]
        tmp += 1
    return spaces
    
def indices_contracting(splitted_out):
    indices = []
    for term in splitted_out:
        wanted = re.search(r'\(.*?\)', term)
        wanted = wanted[0][1:-1]
        indx = wanted.replace(',','')
        indices.append(indx)
    return indices

def find_tensor(splitted_out):
    tensor = []
    for term in splitted_out:
        tensor.append(term[1])
    return tensor

def create_intermediate(indices, tensor, num, splitted_out):
    mid_1 = int(len(indices[0])/2)
    mid_2 = int(len(indices[1])/2)
    int_indx = indices[0][:mid_1] + indices[1][:mid_2] + indices[0][mid_1:] + indices[1][mid_2:]
    indices.remove(indices[0])
    indices[0] = int_indx
    tensor.remove(tensor[0])
    #print(num)
    tensor[0] = f'I{num}'
    #print(tensor)
    
    

def print_einsum_out(indices, tensor, phase, number, name):
    print(f"{name} {phase}= {number}*np.einsum('{indices[0]}, {indices[1]} -> {target[name]}', {my_tensors[tensor[0]]}, {my_tensors[tensor[1]]}, optimize = 'optimal')")

def print_einsum_with_int_out(indices, tensor, phase, number, name, tar):
    print(f"{name} {phase}= {number}*np.einsum('{indices[0]}, {indices[1]} -> {tar}', {my_tensors[tensor[0]]}, {my_tensors[tensor[1]]}, optimize = 'optimal')")
    


#print("hi")
equation = input("Please enter the type of equation to be converted to code. eg energy, eqT1 or eqT2 :: ")
inp = get_what(equation)
if equation == 'energy':
    name = 'energy'
elif equation == 'eqT1':
    name = 'T1'
elif equation == 'eqT2':
    name = 'T2'
else:
    raise NameError("Doesnt belong to any equation type mentioned")
tot_len = len(inp.args)
lis = []
for i in inp.args:
    out = re.sub('\\\\frac{(.+?)}{(\d+?)}', '(1/\\2)*\\1', latex(i))
    out = re.sub('{(\w+?)}_{(\w+?)}', '(\\1,\\2)', out)
    lis.append(out)
    if out[0] == '+' or out[0] == '-':
        phase = out[0]
        out = out[2:]
    else:
        phase = '+'
    

    splitted_out = out.split()
    #if len(splitted_out<=2):
     
        #continue
    #else:
    rank = []
    space = []
    indices = []
    tensor = []
    
    #print("here")
    
    if '*' in splitted_out[0]:
        number = splitted_out[0].split('*')[0]
        splitted_out[0] = splitted_out[0].split('*')[1]
    else:
        number = 1
    
    rank = calc_rank(splitted_out)
    #print(rank)
    tensor = find_tensor(splitted_out)
    space = space_identifier(splitted_out, tensor)
    #print(space)
    indices = indices_contracting(splitted_out)
    #print(indices)
    
        
        # rank.append(calc_rank(splitted_out[flag]))
        # space.append(space_identifier(splitted_out[flag]))
        # indices.append(indices_contracting(splitted_out[flag]))
        # tensor.append(find_tensor(splitted_out[flag]))
        
        #flag += 1
        
    for k in range(len(tensor)):
        tensor[k] = tensor[k] + '_' + space[k]
    #print(tensor)
    
    if len(rank) == 2:
        print_einsum_out(indices, tensor, phase, number, name)
     
    if len(rank) > 2:
        var_1 = len(rank)
        num = 1
        while var_1 > 1:
            if len(indices) == 2:
                print_einsum_out(indices, tensor, phase, number, name)
                # for foo in tensor:
                #     if 'I' in foo:
                #         print_einsum_with_int_in_left(indices, tensor, phase, number, name = 'ener', tar = f"I{num}")
                #     else:
                #         print_einsum_out(indices, tensor, phase, number, name='ener')
                break
            print_einsum_with_int_out(indices, tensor, phase, number, name, tar = f'I{num}')
            create_intermediate(indices, tensor, num, splitted_out)
            num += 1
            var_1 -= 1
            
            
        
        

      
        
        
    
    

