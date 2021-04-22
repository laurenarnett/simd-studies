import numpy as np
from numba import jit
import timeit

import ctypes

import llvmlite.binding as llvm
llvm.set_option('', '--debug-only=loop-vectorize')

pmb = PassManagerBuilder.populate(loop_vectorize=False)



def find_instr(func, keyword, sig=0, limit=5):
    count = 0
    for l in func.inspect_asm(func.signatures[sig]).split('\n'):
        if keyword in l:
            count += 1
            print(l)
            if count >= limit:
                break
    if count == 0:
        print('No instructions found')

vals = np.random.randint(0, 500, ((1 << 16) + 10), dtype=np.uint32)

NUM_REPEATS = 10

'''
Sum values in vanilla python with a for loop
'''
print("***********************************")
print("SUM")
print("***********************************")
def vals_sum(x):
    res = 0
    for i in range(x.shape[0]):
        res += x[i]
    return res

print("vals sum")
t1 = timeit.timeit('vals_sum(vals)',"from __main__ import vals_sum, vals", number=1000)
print("vals sum {}".format(t1 / NUM_REPEATS))

'''
Sum values in vanilla python with a for loop and a filter
'''
print("***********************************")
print("SUM FILTER")
print("***********************************")
def vals_sum_filter(x):
    res = 0
    for i in range(x.shape[0]):
        val = x[i]
        if val >= 128:
            res += val
    return res

print("vals sum filter")
# t1 = timeit.timeit('vals_sum_filter(vals)',"from __main__ import vals_sum_filter, vals", number=NUM_REPEATS)
# print("vals sum filter {}".format(t1 / NUM_REPEATS))


'''
Sum values using numpy
'''
print("***********************************")
print("SUM NP")
print("***********************************")
def vals_sum_np(x):
    return np.sum(x)

print("sum np")
t1 = timeit.timeit('vals_sum_np(vals)',"from __main__ import vals_sum_np, vals", number=NUM_REPEATS)
print("sum np {}".format(t1 / NUM_REPEATS))

'''
Sum values using numpy & where
'''
print("***********************************")
print("SUM NP FILTER")
print("***********************************")
def vals_sum_filter_np(x):
    return np.sum(np.where(x > 127, x, 0))

print("sum np filter")
t1 = timeit.timeit('vals_sum_filter_np(vals)',"from __main__ import vals_sum_filter_np, vals", number=NUM_REPEATS)
print("sum np filter {}".format(t1 / NUM_REPEATS))



'''
array sum using simd with a for loop
'''
print("***********************************")
print("SIMD SUM")
print("***********************************")


@jit(nopython=True)
def simd_sum(x):
    res = 0
    for i in range(x.shape[0]):
        res += x[i]
    return res

print("simd sum precompile")
t1 = timeit.timeit('simd_sum(vals)',"from __main__ import simd_sum, vals",number=NUM_REPEATS)
print("pre compile simd sum {}".format(t1 / NUM_REPEATS))

print("Actual simd sum")
t1 = timeit.timeit('simd_sum(vals)',"from __main__ import simd_sum, vals", number=NUM_REPEATS)
print("Actual simd sum {}".format(t1 / NUM_REPEATS))

find_instr(simd_sum, keyword='ymm', sig=0)
find_instr(simd_sum, keyword='xmm', sig=0)


'''
array sum using simd with numpy.sum()
'''
print("***********************************")
print("SIMD SUM NP")
print("***********************************")
@jit(nopython=True)
def simd_sum_np(x):
    return np.sum(x)

print("simd sum np precompile")
t1 = timeit.timeit('simd_sum_np(vals)',"from __main__ import simd_sum_np, vals", number=NUM_REPEATS)
print("pre compile simd sum np {}".format(t1 / NUM_REPEATS))

print("Actual simd np sum")
t1 = timeit.timeit('simd_sum_np(vals)',"from __main__ import simd_sum_np, vals", number=NUM_REPEATS)
print("Actual simd sum np {}".format(t1 / NUM_REPEATS))

find_instr(simd_sum_np, keyword='ymm', sig=0)
find_instr(simd_sum_np, keyword='xmm', sig=0)



'''
array sum using simd with a for loop and a filter
'''
print("***********************************")
print("SIMD SUM FILTER")
print("***********************************")
@jit(nopython=True)
def simd_sum_filter(x):
    res = 0
    for i in range(x.shape[0]):
        val = x[i]
        if val >= 128:
            res += val
    return res

print("simd sum filter precompile")
t2 = timeit.timeit('simd_sum_filter(vals)',"from __main__ import simd_sum_filter, vals", number = NUM_REPEATS)
print("pre compile simd sum filter {}".format(t2 / NUM_REPEATS))

print("Actual simd sum filter")
t2 = timeit.timeit('simd_sum_filter(vals)',"from __main__ import simd_sum_filter, vals",number=NUM_REPEATS)
print("Actual simd sum filter {}".format(t2 / NUM_REPEATS))

find_instr(simd_sum_filter, keyword='ymm', sig=0)
find_instr(simd_sum_filter, keyword='xmm', sig=0)

'''
array sum using simd with numpy.sum() and numpy.where() filter
'''
print("***********************************")
print("SIMD SUM FILTER NP")
print("***********************************")
@jit(nopython=True)
def simd_sum_filter_np(x):
    return np.sum(np.where(x > 127, x, 0))
print("simd sum filter np pre compile")
t2 = timeit.timeit('simd_sum_filter_np(vals)',"from __main__ import simd_sum_filter_np, vals",number=NUM_REPEATS)
print("simd sum filter np precompile {}".format(t2 / NUM_REPEATS))

print("Actual simd sum filter np")
t2 = timeit.timeit('simd_sum_filter(vals)',"from __main__ import simd_sum_filter, vals", number=NUM_REPEATS)
print("Actual simd sum filter np {}".format(t2 / NUM_REPEATS))

find_instr(simd_sum_filter_np, keyword='ymm', sig=0)
find_instr(simd_sum_filter_np, keyword='xmm', sig=0)

