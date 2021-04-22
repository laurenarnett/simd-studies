#include <time.h>
#include <stdio.h>
#include <x86intrin.h>
#include <stdint.h>
#include <string.h>
#include "mod.h"

long long int sum_mod(unsigned int vals[NUM_ELEMENTS]) {
    clock_t start = clock();
    long long int sum = 0;
    for(unsigned int w = 0; w < NUM_ITERS; w++) {
        for(unsigned int i = 0; i < NUM_ELEMENTS; i++) {
            if(vals[i] % 2 == 1)
                sum += vals[i];
        }
    }
    clock_t end = clock();
    printf("Time taken: %Lf s\n", ((long double)(end - start) / CLOCKS_PER_SEC) / NUM_ITERS);
    return sum;
}

long long int sum_mod_unrolled(unsigned int vals[NUM_ELEMENTS]) {
    clock_t start = clock();
    long long int sum = 0;
    for(unsigned int w = 0; w < NUM_ITERS; w++) {
        for(unsigned int i = 0; i < NUM_ELEMENTS / 4 * 4; i += 4) {
            if(vals[i] % 2) sum += vals[i];
            if(vals[i + 1] % 2) sum += vals[i + 1];
            if(vals[i + 2] % 2) sum += vals[i + 2];
            if(vals[i + 3] % 2) sum += vals[i + 3];
        }
        // tailcase
        for(unsigned int i = NUM_ELEMENTS / 4 * 4; i < NUM_ELEMENTS; i++) {
            if (vals[i] % 2)
                sum += vals[i];
        }
    }
    clock_t end = clock();
    printf("Time taken: %Lf s\n", (long double)(end - start) / CLOCKS_PER_SEC / NUM_ITERS);
    return sum;
}

long long int sum_simd_mod(unsigned int vals[NUM_ELEMENTS]) {
    clock_t start = clock();
    long long int final_result = 0;
    __m128i zero_vec = _mm_setzero_si128();//zero vector
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
            __m128i values_vec = _mm_loadu_si128(&vals[i]);

            //Shift integers left by 31 bits, while shifting in zeros
            __m128i left_shift_vec = _mm_slli_epi32(values_vec, 31);

            //Shift integers right by 31 bits, while shifting in zeros
            //Gives us just the first bit from each integer in values vec
            __m128i mod_2_vec = _mm_srli_epi32(left_shift_vec, 31);

            //Mask mod_2_vec with [0, 0, 0, 0]
            //0xFFFFFFFF for integers in mod_2_vec > 0, else 0
            __m128i mask = _mm_cmpgt_epi32(mod_2_vec, zero_vec);

            //The valid vector, even numbers replaced with zero
            __m128i valid_vec = _mm_and_si128(values_vec, mask);

            //sum the valid vector with the existing sum
            vsum = _mm_add_epi32(valid_vec, vsum);
        }

        _mm_storeu_si128((__m128i *)res, vsum);

        final_result += res[0];
        final_result += res[1];
        final_result += res[2];
        final_result += res[3];

        //tail case
        for (unsigned int i = NUM_ELEMENTS - (NUM_ELEMENTS % 4); i < NUM_ELEMENTS; i++) {
            if(vals[i] % 2 == 1)
                final_result += vals[i];
        }
    }
    clock_t end = clock();
    printf("Time taken: %Lf s\n", (((long double)(end - start) / CLOCKS_PER_SEC) / NUM_ITERS));
    return final_result;
}
