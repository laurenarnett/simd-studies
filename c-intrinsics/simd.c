#include <stdio.h>
#include <stdlib.h>
#include "common.h"

int main(int argc, char* argv[]) {
	printf("Let's generate a randomized array.\n");
	unsigned int vals[NUM_ELEMENTS];
	long long int reference;
	long long int reference_filter;
	long long int reference_mod;
	long long int simd;
	long long int simdu;
	for(unsigned int i = 0; i < NUM_ELEMENTS; i++) vals[i] = rand() % 256;

	/* printf("Starting randomized sum.\n"); */
	/* printf("Sum: %lld\n", reference = sum(vals)); */

	/* printf("Starting randomized sum with filter.\n"); */
	/* printf("Sum: %lld\n", reference_filter = sum_filter(vals)); */

	printf("Starting randomized sum with mod 2 filter.\n");
	printf("Sum: %lld\n", reference_mod = sum_mod(vals));

	/* printf("Starting randomized unrolled sum.\n"); */
	/* printf("Sum: %lld\n", sum_unrolled(vals)); */

	/* printf("Starting randomized SIMD sum.\n"); */
	/* printf("Sum: %lld\n", simd = sum_simd(vals)); */
	/* if (simd != reference) { */
	/* 	printf("OH NO! SIMD sum %lld doesn't match reference sum %lld!\n", simd, reference); */
	/* } */

	/* printf("Starting randomized SIMD sum with filter.\n"); */
	/* printf("Sum: %lld\n", simd = sum_simd_filter(vals)); */
	/* if (simd != reference_filter) { */
	/* 	printf("OH NO! SIMD sum %lld doesn't match reference sum %lld!\n", simd, reference); */
	/* } */

	printf("Starting randomized SIMD sum with mod 2 filter.\n");
	printf("Sum: %lld\n", simd = sum_simd_mod(vals));
	if (simd != reference_mod) {
		printf("OH NO! SIMD sum %lld doesn't match reference sum %lld!\n", simd, reference);
	}
}
