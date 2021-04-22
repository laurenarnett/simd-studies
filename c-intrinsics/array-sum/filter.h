#ifndef __FILTER_H_
#define __FILTER_H_

#include "common.h"

long long int sum_filter(unsigned int vals[NUM_ELEMENTS]);
long long int sum_filter_unrolled(unsigned int vals[NUM_ELEMENTS]);
long long int sum_simd_filter(unsigned int vals[NUM_ELEMENTS]);

#endif // __FILTER_H_
