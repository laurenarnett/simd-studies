#include <time.h>
#include <stdio.h>
#include <x86intrin.h>
#include <stdint.h>
#include <string.h>
#include "sum.h"

long long int sum(unsigned int vals[NUM_ELEMENTS]) {
    clock_t start = clock();
    long long int sum = 0;
    for(unsigned int w = 0; w < NUM_ITERS; w++) {
        for(unsigned int i = 0; i < NUM_ELEMENTS; i++) {
            sum += vals[i];
        }
    }
    clock_t end = clock();
    printf("Time taken: %Lf s\n", ((long double)(end - start) / CLOCKS_PER_SEC) / NUM_ITERS);
    return sum;
}

long long int sum_unrolled(unsigned int vals[NUM_ELEMENTS]) {
    clock_t start = clock();
    long long int sum = 0;
    for(unsigned int w = 0; w < NUM_ITERS; w++) {
        for(unsigned int i = 0; i < NUM_ELEMENTS / 4 * 4; i += 4) {
            sum += vals[i];
            sum += vals[i + 1];
            sum += vals[i + 2];
            sum += vals[i + 3];
        }
        // tailcase
        for(unsigned int i = NUM_ELEMENTS / 4 * 4; i < NUM_ELEMENTS; i++) {
                sum += vals[i];
        }
    }
    clock_t end = clock();
    printf("Time taken: %Lf s\n", (long double)(end - start) / CLOCKS_PER_SEC / NUM_ITERS);
    return sum;
}

long long int sum_simd(unsigned int vals[NUM_ELEMENTS]) {
    clock_t start = clock();
    long long int result = 0;
    for(unsigned int w = 0; w < NUM_ITERS; w++) {
        int res[4];
        res[0] = 0;
        res[1] = 0;
        res[2] = 0;
        res[3] = 0;
        __m128i vsum = _mm_setzero_si128();

        //Each vector is 128 bits, so we can store 4 32 bit integers
        for (unsigned int i = 0; i < NUM_ELEMENTS - (NUM_ELEMENTS % 4); i += 4) {

            //Vector of next four array values
            __m128i vector = _mm_loadu_si128(&vals[i]);

            // sum the vector with the existing sum
            vsum = _mm_add_epi32(vector, vsum);
        }

        _mm_storeu_si128((__m128i *)res, vsum);

        result += res[0];
        result += res[1];
        result += res[2];
        result += res[3];

        // tail case
        for (unsigned int i = NUM_ELEMENTS - (NUM_ELEMENTS % 4); i < NUM_ELEMENTS; i++) {
                result += vals[i];
        }
    }
    clock_t end = clock();
    printf("Time taken: %Lf s\n", (((long double)(end - start) / CLOCKS_PER_SEC) / NUM_ITERS));
    return result;
}
