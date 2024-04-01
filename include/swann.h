#ifndef SWAN_H
#define SWAN_H

#define USE_MATH_DEFINES

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <lapacke/lapack.h>

void CalcEigen(lapack_int n, lapack_complex_double *es, double *ev);
void GenBand(int num_threads, int Dim, int Nb, int Nk, int Nl, double *k, int *site, int *obt, double *t, double *band);

#endif
