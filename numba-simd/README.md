# Numba + LLVM auto-vectorization

This study looks at the performance of Numba with SIMD auto-vectorization on various datatypes.

## Tests

### Array Sum
 - `array_sum.py`
This test sums all elements in an array with and without an `element >= 100` filter using different combinations of Numba + SIMD, Numpy, and vanilla Python.

### String Equivalence
 - `string_eq.py`
This test compares performance of a string equivalence program with Numba + SIMD and vanilla Python. Runs over different string encoding datatypes: unicode, utf-32, ascii, and np.char.array.

## Dependencies
Must have Numba installed with a version of SciPy that was compiled against a BLAS library. The Anaconda distribution is recommended for this and will install the appropriate dependencies.

`conda install numba`

See [here](https://numba.pydata.org/numba-doc/latest/user/installing.html) for more info.


## Run test

``` sh
python3 array_sum.py
```

## Sample output

``` sh
Running vanilla sum with for loop
Time per iteration: 9.3717e-03 s

Running vanilla sum with for loop & filter
Time per iteration: 6.8812e-02 s

Running np.sum()
Time per iteration: 3.6783e-05 s

Running np.sum() with np.where()
Time per iteration: 1.6089e-04 s

JIT Compiling numba simd with for loop
Running numba simd with for loop
Time per iteration: 4.6753e-06 s

JIT Compiling numba simd with np.sum()
Running numba simd with np.sum()
Time per iteration: 4.6891e-06 s

JIT Compiling numba simd with for loop & filter
Running numba simd with for loop & filter
Time per iteration: 8.7231e-06 s

JIT Compiling numba simd with np.sum() and np.where()
Running numba simd with np.sum() and np.where()
Time per iteration: 8.6740e-06 s
```
