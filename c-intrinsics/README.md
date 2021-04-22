# C Vector Intrinsics

This program seeks to see vector intrinsics scale with increasingly complex programs.

## Run all tests

``` sh
make
./simd
```

## Sample output

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
