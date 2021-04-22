import numpy as np
from numba import jit
import timeit

# ********************************************************************
#
# Testing utilities
#
# ********************************************************************
def run_test(fn, arg1, label, show_instr=False):
    if 'simd' in fn.__name__:
        print("JIT Compiling {}".format(label))
        # must run at least one time to JIT compile the function
        t = timeit.timeit('{}({})'.format(fn.__name__, arg1),"from __main__ import {}, {}".format(fn.__name__, arg1), number=1)

    print("Running %s" % label)
    t = timeit.timeit('{}({})'.format(fn.__name__, arg1),"from __main__ import {}, {}".format(fn.__name__, arg1) ,number=NUM_ITERS)
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

NUM_ITERS = 1000

vals = np.random.randint(0, 500, ((1 << 16) + 10), dtype=np.uint32)

'''
Sum values in vanilla python with a for loop
'''
def vals_sum(x):
    res = 0
    for i in range(x.shape[0]):
        res += x[i]
    return res

run_test(vals_sum, 'vals', 'vanilla sum with for loop')

'''
Sum values in vanilla python with a for loop and a filter
'''
def vals_sum_filter(x):
    res = 0
    for i in range(x.shape[0]):
        val = x[i]
        if val >= 128:
            res += val
    return res

run_test(vals_sum_filter, 'vals', 'vanilla sum with for loop & filter')

'''
Sum values using numpy
'''
def vals_sum_np(x):
    return np.sum(x)

run_test(vals_sum_np, 'vals', 'np.sum()')

'''
Sum values using numpy & where
'''
def vals_sum_filter_np(x):
    return np.sum(np.where(x > 100, x, 0))

run_test(vals_sum_filter_np, 'vals', 'np.sum() with np.where()')

'''
array sum using simd with a for loop
'''
@jit(nopython=True)
def simd_sum(x):
    res = 0
    for i in range(x.shape[0]):
        res += x[i]
    return res

run_test(simd_sum, 'vals', 'numba simd with for loop')

'''
array sum using simd with numpy.sum()
'''
@jit(nopython=True)
def simd_sum_np(x):
    return np.sum(x)

run_test(simd_sum_np, 'vals', 'numba simd with np.sum()')

'''
array sum using simd with a for loop and a filter
'''
@jit(nopython=True)
def simd_sum_filter(x):
    res = 0
    for i in range(x.shape[0]):
        val = x[i]
        if val >= 100:
            res += val
    return res

run_test(simd_sum_filter, 'vals', 'numba simd with for loop & filter')

'''
array sum using simd with numpy.sum() and numpy.where() filter
'''
@jit(nopython=True)
def simd_sum_filter_np(x):
    return np.sum(np.where(x > 127, x, 0))

run_test(simd_sum_filter, 'vals', 'numba simd with np.sum() and np.where()')
