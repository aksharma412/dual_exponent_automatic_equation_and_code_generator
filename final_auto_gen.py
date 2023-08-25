# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:17:30 2023

@author: Aparna K
"""

import pickle
import os
from sympy import (symbols, Rational, latex, Dummy)
import re
import sys
from sympy.physics.secondquant import (AntiSymmetricTensor, wicks,
        F, Fd, NO, evaluate_deltas, substitute_dummies, Commutator,
        simplify_index_permutations, PermutationOperator)

# global list that stores the standard forms of the two electron integrals
standard = ['oooo', 'ooov', 'oovv', 'ovov', 'ovvv', 'vvvv']

# global list  that stores the standard forms of the S type tensor defined (to get T3 like terms from two T2 like terms)
std_s = ['oovo', 'ovvv']

# dictionary that maps the tensor definition in the equation to what we want in the generated code
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

# dictionary mapping the observables to the target indices to be used in the generated code
target = {
    'energy' : ' ',
    'T1' : 'ia',
    'T2' : 'ijab',
    'T3' : 'ijkabc'
}
# Path to the cache file
cache_file = "cached_data.pkl"

def get_cached_data():
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    else:
        return None

def cache_data(data):
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

def get_what():
    pretty_dummies_dict = {
        'above': 'cdefgh',
        'below': 'klmno',
        'general': 'pqrstu'
    }
    
    # define the symbols required to construct the hamiltonian
    i = symbols('i', below_fermi=True, cls=Dummy)
    a = symbols('a', above_fermi=True, cls=Dummy)
    j = symbols('j', below_fermi=True, cls=Dummy)
    b = symbols('b', above_fermi=True, cls=Dummy)
    p, q, r, s = symbols('p,q,r,s', cls=Dummy)

    # build the hamiltonian - f+V
    fock = AntiSymmetricTensor('f', (p,), (q,))
    pr = NO(Fd(p)*F(q))
    V = AntiSymmetricTensor('v',(p,q),(r,s))
    pqsr = NO(Fd(p)*Fd(q)*F(s)*F(r))
    H = fock*pr + Rational(1,4)*V*pqsr
    H

    # def get_T():
    #     i, j = symbols('i,j', below_fermi=True, cls=Dummy)
    #     a, b = symbols('a,b', above_fermi=True, cls=Dummy)
    #     t_ai = AntiSymmetricTensor('t', (a,), (i,))*NO(Fd(a)*F(i))
    #     t_abij = Rational(1,4)*AntiSymmetricTensor('t', (a,b), (i,j))*NO(Fd(a)*Fd(b)*F(j)*F(i))
    #     return t_ai + t_abij

    # get_T()
    
    # Building the T amplitudes, T1 and T2 are made in the conventional way whereas we use a T2 decomposition for T3
    def get_T1():
        i, j = symbols('i,j', below_fermi=True, cls=Dummy)
        a, b = symbols('a,b', above_fermi=True, cls=Dummy)
        t1 = AntiSymmetricTensor('t', (a,), (i,))*NO(Fd(a)*F(i))
        return t1
    
    def get_T2():
        i, j = symbols('i,j', below_fermi=True, cls=Dummy)
        a, b = symbols('a,b', above_fermi=True, cls=Dummy)
        t2 = Rational(1,4)*AntiSymmetricTensor('t', (a,b), (i,j))*NO(Fd(a)*Fd(b)*F(j)*F(i))
        return t2

    # Building one of the T2 like intermediates for T3
    def get_S():
        i, j, k,l = symbols('i,j,k,l', below_fermi=True, cls=Dummy)
        a, b, c, d = symbols('a,b,c,d', above_fermi=True, cls=Dummy)
        s_vooo = AntiSymmetricTensor('s', (a,l), (i,j))*NO(Fd(a)*Fd(l)*F(j)*F(i))
        s_vvvo = AntiSymmetricTensor('s', (a,b), (i,d))*NO(Fd(a)*Fd(b)*F(d)*F(i))
        return s_vooo, s_vvvo
    
    # building the T3 like term using the intermediate defined above
    def get_ST():
        s_vooo, s_vvvo = get_S()
        t2 = get_T2()
        C = Commutator
        tmp = wicks(C(s_vooo, t2)) + wicks(C(s_vvvo, t2))
        tmp = evaluate_deltas(tmp)
        tmp = substitute_dummies(tmp, new_indices=True, pretty_indices = pretty_dummies_dict)
        return tmp

    # Carrying out the BCH expansion, truncated at fourth commutator
    C = Commutator
    T = get_T1() + get_T2() + get_ST()
    print("commutator 1...")
    comm1 = wicks(C(H, T))
    comm1 = evaluate_deltas(comm1)
    comm1 = substitute_dummies(comm1)

    T = get_T1() + get_T2() + get_ST()
    print("commutator 2...")
    comm2 = wicks(C(comm1, T))
    comm2 = evaluate_deltas(comm2)
    comm2 = substitute_dummies(comm2)

    T = get_T1() + get_T2() + get_ST()
    print("commutator 3...")
    comm3 = wicks(C(comm2, T))
    comm3 = evaluate_deltas(comm3)
    comm3 = substitute_dummies(comm3)

    T = get_T1() + get_T2() + get_ST()
    print("commutator 4...")
    comm4 = wicks(C(comm3, T))
    comm4 = evaluate_deltas(comm4)
    comm4 = substitute_dummies(comm4)

    # simplifying the bch expansion by substituting dummies
    eq = H + comm1 + comm2/2 + comm3/6 + comm4/24
    eq = eq.expand()
    eq = evaluate_deltas(eq)
    eq = substitute_dummies(eq, new_indices=True,
            pretty_indices=pretty_dummies_dict)
    
    # getting the energy equation
    i, j, k, l = symbols('i,j,k,l', below_fermi=True)
    a, b, c, d = symbols('a,b,c,d', above_fermi=True)
    energy = wicks(eq, simplify_dummies=True,
            keep_only_fully_contracted=True)
    
    # defining the T1 amplitude equation
    eqT1 = wicks(NO(Fd(i)*F(a))*eq, simplify_dummies=True, 
                 keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
    
    # defining the T2 amplitude equation
    eqT2 = wicks(NO(Fd(i)*Fd(j)*F(b)*F(a))*eq, simplify_dummies=True, 
                 keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
    
    # defining the t3 amplitude equation
    eqT3 = wicks(NO(Fd(i)*Fd(j)*Fd(k)*F(c)*F(b)*F(a))*eq, simplify_dummies=True, keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
    
    return energy, eqT1, eqT2, eqT3

# Load cached data
cached_data = get_cached_data()

if cached_data is None:
    # If cached data doesn't exist, compute it and cache it
    cached_data = get_what()
    cache_data(cached_data)

    # Uncomment the following line to exit the program after caching
    # exit()

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

def equivalent_form(eri, tensor_string):
    eq_forms = []
    eq_forms.append(eri[0] + eri[1] + eri[2] + eri[3])
    eq_forms.append(eri[1] + eri[0] + eri[3] + eri[2])
    eq_forms.append(eri[2] + eri[3] + eri[0] + eri[1])
    eq_forms.append(eri[3] + eri[2] + eri[1] + eri[0])
    eq_forms.append(eri[2] + eri[1] + eri[0] + eri[3])
    eq_forms.append(eri[3] + eri[0] + eri[1] + eri[2])
    eq_forms.append(eri[0] + eri[3] + eri[2] + eri[1])
    eq_forms.append(eri[1] + eri[2] + eri[3] + eri[0])
    
    space_eq = []
    for t in eq_forms:
        flag = ''
        if t[0] <= 'h':
            flag += 'v'
        else:
            flag += 'o'
        if t[1] <= 'h':
            flag += 'v'
        else:
            flag += 'o'
        if t[2] <= 'h':
            flag += 'v'
        else:
            flag += 'o'
        if t[3] <= 'h':
            flag += 'v'
        else:
            flag += 'o'
        space_eq.append(flag)
    
        
    for itm in space_eq:
        if itm in standard and tensor_string == 'v':
            result = itm
            break
        if itm in std_s and tensor_string == 's':
            result = itm
            break
    equivalent_tensor_with_space = []
    for l in range(len(space_eq)):
        equivalent_tensor_with_space = eq_forms[l] + space_eq[l]
            
    return result

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
    

#inp_list = []
equation = input("Please enter the type of equation to be converted to code. eg energy, eqT1 or eqT2. Or enter 0 to exit :: ")
# Now you can use the cached_data variable in your program
energy, eqT1, eqT2, eqT3 = cached_data
#inp_list = get_what()
while equation != '0':
    
    if equation == 'energy':
        name = 'energy'
        inp = cached_data[0]
    elif equation == "eqT1":
        name = 'T1'
        inp = cached_data[1]
    elif equation == 'eqT2':
        name = 'T2'
        inp = cached_data[2]
    elif equation == 'eqT3':
        name = 'T3'
        inp = cached_data[3]
        
# while inp != '0':
#     if equation == 'energy':
#         name = 'energy'
#     elif equation == 'eqT1':
#         name = 'T1'
#     elif equation == 'eqT2':
#         name = 'T2'
#     else:
#         raise NameError("Doesnt belong to any equation type mentioned")
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
        if  splitted_out[0] == '2':
            number = splitted_out[0]
            splitted_out = splitted_out[1:]
        # if type(int(splitted_out[0])) == int:
        #     number = splitted_out[0]
        #     splitted_out = splitted_out[1:]
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
            
        # print(rank)
        # print(space)
        # print(tensor)
        # print(indices)
        for i in range(len(space)):
            if len(space[i]) == 4 and tensor[i] == 'v':
                if space[i] in standard:
                    continue
                else:
                    space[i] = equivalent_form(indices[i], tensor[i])
            if len(space[i]) == 4 and tensor[i] == 's':
                if space[i] in std_s:
                    continue
                else:
                    space[i] = equivalent_form(indices[i], tensor[i])
                    # t = indices[i]
                    # flag = ''
                    # if t[0] <= 'h':
                    #     flag += 'v'
                    # else:
                    #     flag += 'o'
                    # if t[1] <= 'h':
                    #     flag += 'v'
                    # else:
                    #     flag += 'o'
                    # if t[2] <= 'h':
                    #     flag += 'v'
                    # else:
                    #     flag += 'o'
                    # if t[3] <= 'h':
                    #     flag += 'v'
                    # else:
                    #     flag += 'o'
                    # space[i] = flag
                
                
                
        for k in range(len(tensor)):
            intmdt = []
            tensor[k] = tensor[k] + '_' + space[k]
            # if tensor[k][0] == 's' or tensor[k][0] == 'v':
            #     intmdt = equivalent_form(tensor[k])
                
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
    repeat = input("Want to continue ? (Please enter y/n) :: ")  
    if repeat == 'y':
        equation = input("Please enter the type of equation to be converted to code. eg energy, eqT1 or eqT2. Or enter 0 to exit :: ")
        
    elif repeat == 'n':
        sys.exit()
    else:
        raise NameError("Please enter a valid option!")         
                

