#ifndef _Lattice_
#define _Lattice_

#define NU	24
#define NC	12
#define tNC 24

#define Spmax	2
#define Dim	3
#define Factor	(8*PI*PI*PI)

#define Nintx		128
#define control_char	'u'
#define control_para	Uinter

#define PI		3.14159265358979323846264338
#define Wmax		128
#define Wrange		10

#include <omp.h>
#include <hdf5.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#include "matrix.h"

typedef struct{
	double t[3][3];
} m33;

typedef struct{
	double t[5][5];
} m55;

typedef struct{
	int i;
	int j;
	int k;
	int mu;
	int nu;
	double sum[3];
	gsl_complex t[3][3];
} NN;

typedef struct{
	int i;
	int j;
	int k;
	int p;
	int q;
	double t_real;
	double t_imag;
} Latt;

#endif
