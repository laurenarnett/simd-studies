# C Vector Intrinsics

This program seeks to examine how vector intrinsics & loop unrolling scale with increasingly complex programs.

## Run all tests

``` sh
cd array-sum
make
./simd
```

## Sample output

``` sh
Generating a randomized array.
Starting randomized sum.
Time taken: 0.000120 s
Sum: 548900372480
Starting randomized unrolled sum.
Time taken: 0.000092 s
Sum: 548900372480
Starting randomized SIMD sum.
Time taken: 0.000041 s
Sum: 548900372480
Starting randomized sum with >= 100 filter.
Time taken: 0.000277 s
Sum: 466427445248
Starting randomized unrolled sum with >= 100 filter.
Time taken: 0.000233 s
Sum: 466427445248
Starting randomized SIMD sum with >= 100 filter.
Time taken: 0.000084 s
Sum: 466427445248
Starting randomized sum with mod 2 filter.
Time taken: 0.000343 s
Sum: 274405457920
Starting randomized unrolled sum with mod 2 filter.
Time taken: 0.000283 s
Sum: 274405457920
Starting randomized SIMD sum with mod 2 filter.
Time taken: 0.000134 s
Sum: 274405457920
```

## Tests
These tests sum all elements in an array using a for-loop, loop unrolling, and vector intrinsics.

### Array Sum
 - `sum.c`
This test sums all elements in an array.

### Array Sum with >= 100 Filter
 - `filter.c`
This test sums all elements in an array where `element >= 100`.

### Array Sum with mod 2 Filter
 - `mod.c`
This test sums all elements in an array where `element % 2 == 1`.
