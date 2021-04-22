import numpy as np
import timeit
import random
import string

STRING_LENGTH = 100000
NUM_ITERS = 1000

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


def to_upper_str(s, s2):
    return [x.upper() for x in s]

run_test(to_upper_str, 's1', 's2', "to upper str")

def to_upper_np(npchar_1, s2):
    return np.char.upper(npchar_1)

run_test(to_upper_np, 'npchar_1', 's2', "to upper np char array")
