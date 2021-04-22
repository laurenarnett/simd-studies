#ifndef COMMON_H
#define COMMON_H

#include <x86intrin.h>

#define NUM_ELEMENTS ((1 << 16) + 10)
#define NUM_ITERS (1 << 16)

long long int sum_filter(unsigned int vals[NUM_ELEMENTS]);
long long int sum_mod(unsigned int vals[NUM_ELEMENTS]);

long long int sum_simd_filter(unsigned int vals[NUM_ELEMENTS]);
long long int sum_simd_mod(unsigned int vals[NUM_ELEMENTS]);
long long int sum_simd(unsigned int vals[NUM_ELEMENTS]);

#endif
