#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "sum.h"
#include "filter.h"
#include "mod.h"

int main(int argc, char* argv[]) {
    printf("Generate a randomized array.\n");
    unsigned int vals[NUM_ELEMENTS];
    long long int reference;
    long long int simd;
    long long int unrolled;
    for(unsigned int i = 0; i < NUM_ELEMENTS; i++) vals[i] = rand() % 256;

	/* SUM TESTS */
    printf("Starting randomized sum.\n");
    printf("Sum: %lld\n", reference = sum(vals));

    printf("Starting randomized unrolled sum.\n");
    printf("Sum: %lld\n", unrolled = sum_unrolled(vals));
    if (unrolled != reference) {
        printf("OH NO! unrolled sum %lld doesn't match reference sum %lld!\n", unrolled, reference);
    }

    printf("Starting randomized SIMD sum.\n");
    printf("Sum: %lld\n", simd = sum_simd(vals));
    if (simd != reference) {
        printf("OH NO! SIMD sum %lld doesn't match reference sum %lld!\n", simd, reference);
    }

	/* FILTER TESTS */
    printf("Starting randomized sum with >= 100 filter.\n");
    printf("Sum: %lld\n", reference = sum_filter(vals));

    printf("Starting randomized unrolled sum with >= 100 filter.\n");
    printf("Sum: %lld\n", unrolled = sum_filter_unrolled(vals));
    if (unrolled != reference) {
        printf("OH NO! unrolled sum %lld doesn't match reference sum %lld!\n", unrolled, reference);
    }

    printf("Starting randomized SIMD sum with >= 100 filter.\n");
    printf("Sum: %lld\n", simd = sum_simd_filter(vals));
    if (simd != reference) {
        printf("OH NO! SIMD sum %lld doesn't match reference sum %lld!\n", simd, reference);
    }

	/* MOD 2 FILTER TESTS */
    printf("Starting randomized sum with mod 2 filter.\n");
    printf("Sum: %lld\n", reference = sum_mod(vals));

    printf("Starting randomized SIMD sum with mod 2 filter.\n");
    printf("Sum: %lld\n", simd = sum_simd_mod(vals));
    if (simd != reference) {
        printf("OH NO! SIMD sum %lld doesn't match reference sum %lld!\n", simd, reference);
    }
}
