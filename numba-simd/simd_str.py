import numpy as np
from numba import jit
import timeit
import random
import string

NUM_ITERS = 100
STRING_LENGTH = 100000

def run_test(fn, arg1, arg2, label, show_instr=False):
    if 'simd' in fn.__name__:
        print("JIT Compiling {}".format(label))
        # must run at least one time to JIT compile the function
        t = timeit.timeit('{}({},{})'.format(fn.__name__, arg1, arg2),"from __main__ import {}, {}, {}".format(fn.__name__, arg1, arg2) ,number=1)

    print("Running %s" % label)
    t = timeit.timeit('{}({},{})'.format(fn.__name__, arg1, arg2),"from __main__ import {}, {}, {}".format(fn.__name__, arg1, arg2) ,number=NUM_ITERS)
    print("Time per iteration: {} s".format(t / NUM_ITERS))
    if 'simd' in fn.__name__ and show_instr == True:
        print("{} instructions".format(label))
        find_instr(fn, keyword='xmm', sig=0)
        find_instr(fn, keyword='ymm', sig=0)
    print("")
    
def find_instr(func, keyword, sig=0, limit=5):
    count = 0
    for l in func.inspect_asm(func.signatures[sig]).split('\n'):
        if keyword in l:
            count += 1
            print(l)
            if count >= limit:
                break
    if count == 0:
        print('No %s instructions found' % keyword)


s1 = ''.join(random.choice(string.ascii_letters) for i in range(STRING_LENGTH))
s2 = ''.join(random.choice(string.ascii_letters) for i in range(STRING_LENGTH))

npchar_1 = np.char.array(s1)
npchar_2 = np.char.array(s2)

ascii1 = s1.encode(encoding="ascii")
ascii2 = s2.encode(encoding="ascii")

npascii1 = np.frombuffer(ascii1, dtype='S1')
npascii2 = np.frombuffer(ascii2, dtype='S1')

utf32_1 = s1.encode(encoding="utf-32")
utf32_2 = s2.encode(encoding="utf-32")

nputf32_1 = np.frombuffer(utf32_1, dtype='S1')
nputf32_2 = np.frombuffer(utf32_2, dtype='S1')


'''
to upper test
'''
def to_upper_str(s, s2):
    return [x.upper() for x in s]
# run_test('to_upper_str', 's1', 's2', "to upper str")

def to_upper_np(npchar_1, s2):
    return np.char.upper(npchar_1)

# run_test('to_upper_np', 'npchar_1', 's2', "to upper np char array")

'''
plain old str
'''
def plain_str_eq(s1, s2):
    return [-1 if s1[i] != s2[i] else 0 for i in range(len(s1))]
# run_test('plain_str_eq', 's1', 's2', "PLAIN STR")

'''
np char array
'''
def np_char_eq(s1, s2):
    return np.char.equal(s1, s2)

# run_test('np_char_eq', 'npchar_1', 'npchar_2', "NUMPY CHAR ARRAY WITH NP EQUALS")

'''
equivalence of chars
for loop
vanilla python
str / bytestring
'''
def str_eq(s1, s2):
    res = np.zeros(len(s1))
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            res[i] = -1
    return res

# run_test('str_eq', 's1', 's2', "STR")
# run_test('str_eq', 'ascii1', 'ascii2', "STR")
# run_test('str_eq', 'npascii1', 'npascii2', "NUMPY BYTES")
# run_test('str_eq', 'utf32_1', 'utf32_2', "UTF 32")
# run_test('str_eq', 'nputf32_1', 'nputf32_2', "NUMPY UTF 32")
# run_test('str_eq', 'npchar_1', 'npchar_2', "NUMPY CHAR ARRAY")


'''
equivalence of chars
for loop
SIMD
str / bytestring
'''
@jit(nopython=True)
def str_eq_simd(s1, s2):
    res = np.zeros(len(s1))
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            res[i] = -1
    return res

run_test(str_eq_simd, 's1', 's2', "SIMD STR", True)

#print(str_eq_simd.)

# run_test('str_eq_simd', 'ascii1', 'ascii2', "SIMD ASCII")
# run_test('str_eq_simd', 'npascii1', 'npascii2', "SIMD NUMPY BYTES")
# run_test('str_eq_simd', 'utf32_1', 'utf32_2', "SIMD UTF 32")
# run_test('str_eq_simd', 'nputf32_1', 'nputf32_2', "SIMD NUMPY UTF 32")
#
#
#run_test('str_eq_simd', 'npchar_1', 'npchar_2', "SIMD NUMPY CHAR ARRAY")


# '''
# Equivalence of chars 
# vanilla python 
# for loop
# np.char.array
# '''
# def arr_eq(l1, l2):
#     res = np.zeros(len(l1))
#     for i in range(l1.shape[0]):
#         if l1[i] != l2[i]:
#             res[i] = -1
#     return res


# '''
# Equivalence of chars
# SIMD
# for loop
# np.char.array / 
# '''
# @jit(nopython=True)
# def arr_eq_simd(l1, l2):
#     res = np.zeros(l1.shape[0])
#     for i in range(l1.shape[0]):
#         if l1[i] != l2[i]:
#             res[i] = -1
#     return res



'''
equivalence of chars
np.equal
np.char.array
error: 
'''
# def arr_eq_np(l1, l2):
#     return np.equal(l1, l2)

# print("arr eq np")
# t2 = timeit.timeit('arr_eq_np(l1,l2)',"from __main__ import arr_eq_np, l1, l2")
# print("arr eq np {}".format(t2))


'''
equivalence of chars
SIMD
np.equal
np.char.array
error: 
'''
# @jit(nopython=True)
# def arr_eq_np_simd(l1, l2):
#     return np.equal(l1, l2)

# print("arr eq simd np")
# t2 = timeit.timeit('arr_eq_np_simd(l1,l2)',"from __main__ import arr_eq_np_simd, l1, l2",number=10)
# print("arr eq simd np {}".format(t2))


# print("arr eq simd np")
# find_instr(arr_eq_simd_np, keyword='ymm', sig=0)
# find_instr(arr_eq_simd_np, keyword='xmm', sig=0)
