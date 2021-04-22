#ifndef __SUM_H_
#define __SUM_H_

#include <x86intrin.h>
#include "common.h"

long long int sum(unsigned int vals[NUM_ELEMENTS]);
long long int sum_unrolled(unsigned int vals[NUM_ELEMENTS]);
long long int sum_simd(unsigned int vals[NUM_ELEMENTS]);

#endif // __SUM_H_
