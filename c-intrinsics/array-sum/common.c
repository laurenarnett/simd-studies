#include <stdio.h>
#include <x86intrin.h>
#include <stdint.h>
#include <string.h>
#include "common.h"

void print128_num(__m128i var)
{
    uint32_t val[4];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %u %u %u %u \n",
           val[0], val[1], val[2], val[3]);
}
