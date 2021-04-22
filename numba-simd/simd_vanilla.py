import numpy as np
import timeit
import time

vals = np.random.randint(0, 500, ((1 << 16) + 10), dtype=np.uint32)


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

#print("skipping\n")
print("vals sum")
t1 = timeit.timeit('vals_sum(vals)',"from __main__ import vals_sum, vals",number=1000)
print("vals sum {}".format(t1 / 1000.0))

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

#print("skipping\n")
print("vals sum filter")
t1 = timeit.timeit('vals_sum_filter(vals)',"from __main__ import vals_sum_filter, vals",number=1000)
print("vals sum filter {}".format(t1 / 1000.0))


'''
Sum values using numpy
'''
print("***********************************")
print("SUM NP")
print("***********************************")
def vals_sum_np(x):
    return np.sum(x)

print("sum np")
t1 = timeit.timeit('vals_sum_np(vals)',"from __main__ import vals_sum_np, vals", number=1000)
print("sum np {}".format(t1 / 1000.0))

