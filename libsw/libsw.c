#include "swann.h"

void CalcEigen(lapack_int n, lapack_complex_double *es, double *ev) {
	char jobz='V', uplo='L';
	lapack_int lda=n, lwork=2*n-1, info;
	double rwork[3*n-2];
	lapack_complex_double work[lwork];

	LAPACK_zheev(&jobz, &uplo, &n, es, &lda, ev, work, &lwork, rwork, &info);
	if(info) {
		printf("LAPACK_zheev FAIL\n");
		exit(1);
	}
}

void GenBand(int num_threads, int Dim, int Nb, int Nk, int Nl, double *k, int *site, int *obt, double *t, double *band) {
	int i, j, n;
	double dot;
	lapack_complex_double tb[Nb*Nb];

//#pragma omp parallel for ordered private(i, j, dot, tb) num_threads(num_threads)
	for(n=0; n<Nk; n++) {
		memset(tb, 0, sizeof(tb));
		for(i=0; i<Nl; i++) {
			dot = 0;
			for(j=0; j<Dim; j++) dot += k[Dim*n + j] * site[Dim*i + j];
			tb[Nb*(obt[2*i]-1) + (obt[2*i+1]-1)] += (t[2*i] + t[2*i+1] * I) * (cos(dot) + sin(dot) * I);
		}
		CalcEigen(Nb, tb, &band[Nb*n]);
	}
}
