import numpy as np
from numba import jit
import timeit
import random
import string

STRING_LENGTH = 100000
NUM_ITERS = 100

SHOW_INSTR = True

# ********************************************************************
#
# Testing utilities
#
# ********************************************************************
def run_test(fn, arg1, arg2, label, show_instr=False):
    if 'simd' in fn.__name__:
        print("JIT Compiling {}".format(label))
        # must run at least one time to JIT compile the function
        t = timeit.timeit('{}({},{})'.format(fn.__name__, arg1, arg2),"from __main__ import {}, {}, {}".format(fn.__name__, arg1, arg2) ,number=1)

    print("Running %s" % label)
    t = timeit.timeit('{}({},{})'.format(fn.__name__, arg1, arg2),"from __main__ import {}, {}, {}".format(fn.__name__, arg1, arg2) ,number=NUM_ITERS)
    print("Time per iteration: {:.4e} s".format(t / NUM_ITERS))
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
# ********************************************************************


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
plain old str
'''
def plain_str_eq(s1, s2):
    return [-1 if s1[i] != s2[i] else 0 for i in range(len(s1))]

run_test(plain_str_eq, 's1', 's2', "string eq using a for loop")

'''
np char array
'''
def np_char_eq(s1, s2):
    return np.char.equal(s1, s2)

run_test(np_char_eq, 'npchar_1', 'npchar_2', "np char array with np.char.equal()")

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

run_test(str_eq, 's1', 's2', "string eq for unicode")
run_test(str_eq, 'ascii1', 'ascii2', "string eq for ascii")
run_test(str_eq, 'npascii1', 'npascii2', "string eq for np from ascii")
run_test(str_eq, 'utf32_1', 'utf32_2', "string eq for utf32")
run_test(str_eq, 'nputf32_1', 'nputf32_2', "string eq for np from utf32")
run_test(str_eq, 'npchar_1', 'npchar_2', "string eq for np char array")


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

run_test(str_eq_simd, 's1', 's2', "simd string eq for unicode", show_instr=SHOW_INSTR)
run_test(str_eq_simd, 'ascii1', 'ascii2', "simd string eq for ascii", show_instr=SHOW_INSTR)
run_test(str_eq_simd, 'npascii1', 'npascii2', "simd string eq for np from ascii", show_instr=SHOW_INSTR)
run_test(str_eq_simd, 'utf32_1', 'utf32_2', "simd string eq for utf32", show_instr=SHOW_INSTR)
run_test(str_eq_simd, 'nputf32_1', 'nputf32_2', "simd string eq for np from utf32", show_instr=SHOW_INSTR)
# run_test(str_eq_simd, 'npchar_1', 'npchar_2', "simd string eq for np char array", show_instr=SHOW_INSTR)
# ^ compilation hangs indefinitely
