CC=gcc
CFLAGS=--std=c99 -Wall # -O3

simd: simd.o common.o sum.o filter.o mod.o
	$(CC) $(CFLAGS) -o simd simd.o common.o sum.o filter.o mod.o

simd.o: simd.c
	$(CC) $(CFLAGS) -S simd.c
	$(CC) $(CFLAGS) -c simd.c

common.o: common.c common.h
	$(CC) $(CFLAGS) -S common.c
	$(CC) $(CFLAGS) -c common.c

sum.o: sum.c sum.h
	$(CC) $(CFLAGS) -S sum.c
	$(CC) $(CFLAGS) -c sum.c

filter.o: filter.c filter.h
	$(CC) $(CFLAGS) -S filter.c
	$(CC) $(CFLAGS) -c filter.c

mod.o: mod.c mod.h
	$(CC) $(CFLAGS) -S mod.c
	$(CC) $(CFLAGS) -c mod.c

clean:
	rm -f simd
	rm -f *.o
	rm -f *.s
