#include <time.h>
#include <stdio.h>
#include <x86intrin.h>
#include <stdint.h>
#include <string.h>
#include "common.h"

void print128_num(__m128i var)
{
    uint32_t val[4];
    memcpy(val, &var, sizeof(val));
    printf("Numerical: %lu %lu %lu %lu \n", 
           val[0], val[1], val[2], val[3]);
}


long long int sum_filter(unsigned int vals[NUM_ELEMENTS]) {
    clock_t start = clock();

        printf("starting sum");
        long long int sum = 0;
    for(unsigned int w = 0; w < NUM_ITERS; w++) {
        for(unsigned int i = 0; i < NUM_ELEMENTS; i++) {
            if(vals[i] >= 128) {
                sum += vals[i];
            }
        }
        }
    clock_t end = clock();
    printf("Time taken: %Lf s\n", ((long double)(end - start) / CLOCKS_PER_SEC) / NUM_ITERS);
    return sum;
}

long long int sum_mod(unsigned int vals[NUM_ELEMENTS]) {
    clock_t start = clock();

        printf("starting sum");
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

long long int sum(unsigned int vals[NUM_ELEMENTS]) {
    clock_t start = clock();

        printf("starting sum");
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
            if(vals[i] >= 128) sum += vals[i];
            if(vals[i + 1] >= 128) sum += vals[i + 1];
            if(vals[i + 2] >= 128) sum += vals[i + 2];
            if(vals[i + 3] >= 128) sum += vals[i + 3];
        }
                // tailcase
        for(unsigned int i = NUM_ELEMENTS / 4 * 4; i < NUM_ELEMENTS; i++) {
            if (vals[i] >= 128) {
                sum += vals[i];
            }
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

                //Each vector is 128 bits, so can store 4 32 bit integers
                for (unsigned int i = 0; i < NUM_ELEMENTS - (NUM_ELEMENTS % 4); i += 4) {

                    //Vector of next four array values
                    __m128i values_vec = _mm_loadu_si128(&vals[i]);

                    __m128i left_shift_vec = _mm_slli_epi32(values_vec, 31);

                    __m128i mod_2_vec = _mm_srli_epi32(left_shift_vec, 31);
                    //Mask mod_2_vec with [0, 0, 0, 0]
                    __m128i mask = _mm_cmpgt_epi32(mod_2_vec, zero_vec);

                    //The valid vector, even numbers replaced with zero
                    __m128i valid_vec = _mm_and_si128(values_vec, mask);

                    vsum = _mm_add_epi32(valid_vec, vsum);
                }

                _mm_storeu_si128((__m128i *)res, vsum);

                final_result += res[0];
                final_result += res[1];
                final_result += res[2];
                final_result += res[3];

                //tail case
                for (unsigned int i = NUM_ELEMENTS - (NUM_ELEMENTS % 4); i < NUM_ELEMENTS; i++) {
                    if(vals[i] % 2 == 1) {
                        final_result += vals[i];
                    }
                }

        }
    clock_t end = clock();
    printf("Time taken: %Lf s\n", (((long double)(end - start) / CLOCKS_PER_SEC) / NUM_ITERS));
    return final_result;
}



long long int sum_simd_filter(unsigned int vals[NUM_ELEMENTS]) {
    clock_t start = clock();
    __m128i _100 = _mm_set1_epi32(100);//120 vector
        long long int result = 0;
    for(unsigned int w = 0; w < NUM_ITERS; w++) {
                int valid[4];
                valid[0] = 0;
                valid[1] = 0;
                valid[2] = 0;
                valid[3] = 0;
                __m128i vsum = _mm_setzero_si128();

                //Each vector is 128 bits, so can store 4 32 bit integers
                for (unsigned int i = 0; i < NUM_ELEMENTS - (NUM_ELEMENTS % 4); i += 4) {

                    //Vector of next four array values
                    __m128i vector = _mm_loadu_si128(&vals[i]);

                    //Mask it with [100, 100, 100, 100]
                    __m128i mask = _mm_cmpgt_epi32(vector, _100);


                    //The valid vector, numbers less than 100 replaced with zero
                    __m128i valid_vector = _mm_and_si128(vector, mask);

                    vsum = _mm_add_epi32(valid_vector, vsum);
                }

                _mm_storeu_si128((__m128i *)valid, vsum);

                result += valid[0];
                result += valid[1];
                result += valid[2];
                result += valid[3];

                //tail case
                for (unsigned int i = NUM_ELEMENTS - (NUM_ELEMENTS % 4); i < NUM_ELEMENTS; i++) {
                    if(vals[i] >= 128) {
                        result += vals[i];
                    }
                }
        }
    clock_t end = clock();
    printf("Time taken: %Lf s\n", (((long double)(end - start) / CLOCKS_PER_SEC) / NUM_ITERS));
    return result;
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

                //Each vector is 128 bits, so can store 4 32 bit integers
                for (unsigned int i = 0; i < NUM_ELEMENTS - (NUM_ELEMENTS % 4); i += 4) {

                    //Vector of next four array values
                    __m128i vector = _mm_loadu_si128(&vals[i]);

                    // Add
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
