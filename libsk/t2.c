#include "lattice.h"

gsl_complex zero;
gsl_complex **Ecluster;
NN Enn[48];
double MU=-4.414653, Ts, Tp, Td, To, Tn;

// pyro_nn.c
int multi_m33t( double A[3][3], double B[3][3], double C[3][3] ){
	int i, j, k;

	for(i=0; i<3; i++) for(j=0; j<3; j++){
		C[i][j] = 0;
		for(k=0; k<3; k++)
			C[i][j] += A[i][k] * B[k][j];
	}
	return 0;
}
int multi_m55t( double A[5][5], double B[5][5], double C[5][5] ){
	int i, j, k;

	for(i=0; i<5; i++) for(j=0; j<5; j++){
		C[i][j] = 0;
		for(k=0; k<5; k++)
			C[i][j] += A[i][k] * B[k][j];
	}
	return 0;
}
void print_m55d(char *comment, double matrix[5][5]){
	int i, j;
	printf("<:: %s ::>--------------\n", comment);
	for(i=0; i<5; i++){
		for(j=0; j<5; j++)
			printf("%9.6lf\t", matrix[i][j] );
		printf("\n");
	}
	printf("------------------------------\n\n");
}

int init_ESK(double ESK[5][5], double l, double m, double n){
	int mu, nu;
	double l2,m2,n2=0, root3=sqrt(3), dds, ddp, ddd;
	l2 = l*l;	m2 = m*m;	n2 = n*n;
	dds = Ts; ddp = Tp; ddd = Td;
	double temp[5][5] = {
		{
			3*dds*l2*m2 + ddp*((1 - 4*m2)*l2 + m2) + ddd*(l2*m2 + n2),
			l*(-4*ddp*m2 + 3*dds*m2 + ddp + ddd*(m2 - 1))* n,
			(-4*ddp*l2 + 3*dds*l2 + ddp + ddd*(l2 - 1))*m*n,
			1/2.*(ddd - 4*ddp + 3*dds)*l*m*(l2 - m2),
			-(1/2.)*root3*l* m*(4*ddp*n2 + dds*(l2 + m2 - 2*n2) - ddd*(n2 + 1))
		},
		{
			 l*(-4*ddp*m2 + 3*dds*m2 + ddp + ddd*(m2 - 1))*n, 
			 3*dds*m2*n2 + ddp*(-4*n2*m2 + m2 + n2) + ddd*(l2 + m2*n2), 
			 l*m*(-4*ddp*n2 + 3*dds*n2 + ddp + ddd*(n2 - 1)), 
			 1/2.*m*(3*dds*(l2 - m2) + ddd*(l2 - m2 + 2) + ddp*(-4*l2 + 4*m2 - 2))*n,
			 -(1/2.)*root3*m*n*(ddd*(l2 + m2) + dds*(l2 + m2 - 2*n2) - 2*ddp*(l2 + m2 - n2))
		},
		{
			 (-4*ddp*l2 + 3*dds*l2 + ddp + ddd*(l2 - 1))*m*n,
			 l*m*(-4*ddp*n2 + 3*dds*n2 + ddp + ddd*(n2 - 1)),
			 3*dds*l2*n2 + ddd*(m2 + l2*n2) + ddp*((1 - 4*n2)*l2 + n2),
			 1/2.*l*(ddd*(l2 - m2 - 2) + 3*dds*(l2 - m2) + ddp*(-4*l2 + 4*m2 + 2))*n,
			 -(1/2.)*root3*l*n*(ddd*(l2 + m2) + dds*(l2 + m2 - 2*n2) - 2*ddp*(l2 + m2 - n2))
		},
		{
			 1/2.*(ddd - 4*ddp + 3*dds)*l*m*(l2 - m2),
			 1/2.*m*(3*dds*(l2 - m2) + ddd*(l2 - m2 + 2) + ddp*(-4*l2 + 4*m2 - 2))*n,
			 1/2.*l*(ddd*(l2 - m2 - 2) + 3*dds*(l2 - m2) + ddp*(-4*l2 + 4*m2 + 2))*n,
			 1/2.*(3/2.*dds*(l2 - m2)*(l2 - m2) + 2*ddp*(l2 + m2 - (l2 - m2)*(l2 - m2)) + 2*ddd*(1/4.*(l2 - m2)*(l2 - m2) + n2)),
			 -(1/4.)*root3*(l2 - m2)*(4*ddp*n2 + dds*(l2 + m2 - 2*n2) - ddd*(n2 + 1))
		},
		{
			-(1/2.)*root3*l*m*(4*ddp*n2 + dds*(l2 + m2 - 2*n2) - ddd*(n2 + 1)),
			-(1/2.)*root3*m*n*(ddd*(l2 + m2) + dds*(l2 + m2 - 2*n2) - 2*ddp*(l2 + m2 - n2)),
			-(1/2.)*root3*l*n*(ddd*(l2 + m2) + dds*(l2 + m2 - 2*n2) - 2*ddp*(l2 + m2 - n2)),
			-(1/4.)*root3*(l2 - m2)*(4*ddp*n2 + dds*(l2 + m2 - 2*n2) - ddd*(n2 + 1)),
			1/4.*(3*ddd*(l2 + m2)*(l2 + m2) + 12*ddp*n2*(l2 + m2) + dds*(l2 + m2 - 2*n2)*(l2 + m2 - 2*n2))
		}
	};

	for(mu=0; mu<5; mu++) for(nu=0; nu<5; nu++)
		ESK[mu][nu] = temp[mu][nu];
	return 0;
}

int init_super_r(int super[3][3], int r[4][3]){
	int super_init[3][3] = {
		{2, 2, 0},
		{0, 2, 2},
		{2, 0, 2},
	};
	int r_init[4][3] = {
		{0,0,0},
		{1,1,0},
		{0,1,1},
		{1,0,1},
	};
	int i,j;
	for(i=0; i<3; i++) for(j=0; j<3; j++)	super[i][j] = super_init[i][j];
	for(i=0; i<4; i++) for(j=0; j<3; j++)	r[i][j] = r_init[i][j];
	return 0;
}

/*
int init_super_r8(int super[3][3], int r[8][3]){
	int super_init[3][3] = {
		{4, 4, 0},
		{0, 2, 2},
		{2, 0, 2},
	};
	int r_init[8][3] = {
		{0,0,0},
		{1,1,0},
		{0,1,1},
		{1,0,1},
		{2,2,0},
		{2,3,1},
		{3,2,1},
		{3,3,0},
	};
	int i,j;
	for(i=0; i<3; i++) for(j=0; j<3; j++)	super[i][j] = super_init[i][j];
	for(i=0; i<8; i++) for(j=0; j<3; j++)	r[i][j] = r_init[i][j];
	return 0;
}
*/

int init_NN(NN Enn[48]){
	// Direct hopping
	int ii, jj, kk, i, j, x, y, super[3][3], r[NC][3], sum[3];
	double l, m, n, ESK[5][5];
	init_super_r(super, r);

	int number=0;
	for(ii=-2; ii<3; ii++) for(jj=-2; jj<3; jj++) for(kk=-2; kk<3; kk++){
		for(i=0; i<4; i++) for(j=0; j<4; j++){
			for(x=0; x<3; x++) sum[x] = ii*super[0][x] + jj*super[1][x] + kk*super[2][x] + r[j][x] - r[i][x];
			if( sum[0]*sum[0] + sum[1]*sum[1] + sum[2]*sum[2] == 6 ){
				//printf("%d sum=(%d,%d,%d) %d %d %d for (%d, %d)\n", number, sum[0], sum[1], sum[2], ii, jj, kk, i, j);
				l = sum[0]/sqrt(6);
				m = sum[1]/sqrt(6);
				n = sum[2]/sqrt(6);

				init_ESK(ESK, l, m, n);
				for(x=0; x<3; x++) for(y=0; y<3; y++)
					Enn[number].t[x][y] = gsl_complex_mul_real(gsl_complex_rect(ESK[x][y], 0), Tn); // t2g
				Enn[number].i = ii;
				Enn[number].j = jj;
				Enn[number].k = kk;
				Enn[number].mu = i;
				Enn[number].nu = j;
				number++;
			}
		}
	}
	//printf("number=%d\n", number);

	return 0;
}

int init_Ecluster(gsl_complex **matrix){
	int i, j, x, y, mu, nu;

	//Construct tau
	m33 tau[4][4], tvec[4], R[4], RT[4], tij[4][4];
	for(i=0; i<4; i++) for(x=0; x<3; x++) for(y=0; y<3; y++){
		tvec[i].t[x][y] = 0;
	}
	tvec[1].t[1][2] =  1;	tvec[1].t[2][1] =  1; 
	tvec[2].t[0][2] =  1;	tvec[2].t[2][0] =  1; 
	tvec[3].t[0][1] = -1;	tvec[3].t[1][0] = -1;

	int coef_tau[4][4] = {
		{0,2,1,3},
		{3,0,1,2},
		{3,2,0,1},
		{2,3,1,0}
	};
	for(i=0; i<4; i++) for(j=0; j<4; j++) {
		for(x=0; x<3; x++) for(y=0; y<3; y++) 
			tau[i][j].t[x][y] = tvec[coef_tau[i][j]].t[x][y];
	}
	//for(i=0; i<4; i++) for(j=0; j<4; j++){
	//	sprintf(debug, "tau[%d][%d]", i, j);
	//	print_m33t(debug, tau[i][j].t);
	//}
	
	//Construct R
	int Rt3[4][3][3] = {
		{ { 2, -1, -2}, {-1,  2, -2}, { 2,  2,  1} },
		{ { 2,  2,  1}, {-2,  1,  2}, { 1, -2,  2} },
		{ { 1, -2,  2}, { 2,  2,  1}, {-2,  1,  2} },
		{ { 1, -2,  2}, {-2, -2, -1}, { 2, -1, -2} }
	};
	for(i=0; i<4; i++) for(x=0; x<3; x++) for(y=0; y<3; y++) {
		R[i].t[x][y]  = (double) Rt3[i][x][y]/3.;
		RT[i].t[x][y] = (double) Rt3[i][y][x]/3.;
	}
	//for(i=0; i<4; i++){
	//	sprintf(debug, "R[%d]", i);
	//	print_m33t(debug, R[i].t);
	//}
	
	//Construct tij
	m33 left, right;
	for(i=0; i<4; i++) for(j=0; j<4; j++){
		multi_m33t(R[j].t,	tau[j][i].t,	right.t);
		multi_m33t(tau[i][j].t,	RT[i].t,	left.t);
		multi_m33t(left.t,	right.t,	tij[i][j].t);

		for(mu=0; mu<3; mu++) for(nu=0; nu<3; nu++){
			tij[i][j].t[mu][nu] *= To;
		}
		//print_m33t("left", left.t);
		//print_m33t("right", right.t);
		//sprintf(debug, "tij[%d][%d]", i, j);
		//print_m33t(debug, tij[i][j].t);
	}

	for(i=0; i<4; i++) for(j=0; j<4; j++){
		for(x=0; x<3; x++) for(y=0; y<3; y++)
			matrix[3*i+x][3*j+y] = gsl_complex_rect(tij[i][j].t[x][y], 0); // t2g
	}
	for(mu=0; mu<NC; mu++)
		matrix[mu][mu]
			= gsl_complex_sub(
					matrix[mu][mu],
					gsl_complex_rect( MU, 0 )
			);

	// Direct hopping
	int super[3][3], r[4][3], sum[3];
	double l, m, n, ESK[5][5];
	init_super_r(super, r);

	for(i=0; i<4; i++) for(j=0; j<4; j++){
		for(x=0; x<3; x++) sum[x] = r[j][x] - r[i][x];
		if( sum[0]*sum[0] + sum[1]*sum[1] + sum[2]*sum[2] == 2 ){
			l = sum[0]/sqrt(2);
			m = sum[1]/sqrt(2);
			n = sum[2]/sqrt(2);

			init_ESK(ESK, l, m, n);
			for(x=0; x<3; x++) for(y=0; y<3; y++)
				matrix[3*i+x][3*j+y] = gsl_complex_add( gsl_complex_rect(ESK[x][y], 0), matrix[3*i+x][3*j+y] ); // t2g
		}
	}

	return 0;
}

// t2.c
int construct_lattice(Latt *data){
	int num=12*12*2 - 12*3 + 48*9;
	int num_check=0, d2, displace[3], i, j, ii, jj, kk, x, y;

	int r[NC][3], super[3][3];
	init_super_r(super, r);

	for(ii=-2; ii<3; ii++) for(jj=-2; jj<3; jj++) for(kk=-2; kk<3; kk++){
		for(i=0; i<4; i++) for(j=0; j<4; j++){
			d2 = 0;
			for(x=0; x<3; x++) {
				displace[x] = ii*super[0][x] + jj*super[1][x] + kk*super[2][x] + r[j][x] - r[i][x];
				d2 += displace[x]*displace[x];
			}
			if(d2 <= 2){
				for(x=0; x<3; x++) for(y=0; y<3; y++) {
					data[num_check].i = ii;
					data[num_check].j = jj;
					data[num_check].k = kk;
					data[num_check].p = 3*i + x + 1;
					data[num_check].q = 3*j + y + 1;

					data[num_check].t_real = GSL_REAL(Ecluster[3*i+x][3*j+y]);
					data[num_check].t_imag = GSL_IMAG(Ecluster[3*i+x][3*j+y]);
					num_check++;
				}
			}
		}
	}

	for(i=0; i<48; i++){ // next nearest-neighbors
		for(x=0; x<3; x++) for(y=0; y<3; y++){
			data[num_check].i = Enn[i].i;
			data[num_check].j = Enn[i].j;
			data[num_check].k = Enn[i].k;
			data[num_check].p = 3*Enn[i].mu + x + 1;
			data[num_check].q = 3*Enn[i].nu + y + 1;

			data[num_check].t_real = GSL_REAL(Enn[i].t[x][y]);
			data[num_check].t_imag = GSL_IMAG(Enn[i].t[x][y]);
			num_check++;
		}
	}
	
	if( num != num_check )	printf("num != num_check\n");

	return num;
}

void save_lattice_h5(Latt *data, int lines, char *path_lat) {
	hid_t file_id, datatype_id, dataset_id, dataspace_id;

	file_id = H5Fcreate(path_lat, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

	datatype_id = H5Tcreate(H5T_COMPOUND, sizeof(Latt));
	H5Tinsert(datatype_id, "i", HOFFSET(Latt, i), H5T_NATIVE_INT);
	H5Tinsert(datatype_id, "j", HOFFSET(Latt, j), H5T_NATIVE_INT);
	H5Tinsert(datatype_id, "k", HOFFSET(Latt, k), H5T_NATIVE_INT);
	H5Tinsert(datatype_id, "p", HOFFSET(Latt, p), H5T_NATIVE_INT);
	H5Tinsert(datatype_id, "q", HOFFSET(Latt, q), H5T_NATIVE_INT);
	H5Tinsert(datatype_id, "t_real", HOFFSET(Latt, t_real), H5T_NATIVE_DOUBLE);
	H5Tinsert(datatype_id, "t_imag", HOFFSET(Latt, t_imag), H5T_NATIVE_DOUBLE);

	dataspace_id = H5Screate_simple(1, (hsize_t[1]){lines}, NULL);
	dataset_id = H5Dcreate2(file_id, "/lat", datatype_id, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Dwrite(dataset_id, datatype_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

	H5Tclose(datatype_id);
	H5Sclose(dataspace_id);
	H5Dclose(dataset_id);
	H5Fclose(file_id);
}

void gen_lattice(int num_thread, double *params, char *path_lat) {
	omp_set_num_threads(num_thread);

	int lines;
	Latt *data=malloc(sizeof(Latt) * 2048); // more than enough
	Ecluster=mkgscmatrixd(NC, NC);

	Ts = params[0];
	Tp = params[1];
	Td = params[2];
	To = 0;
	Tn = params[3];

	init_Ecluster(Ecluster);
	init_NN(Enn);
	lines = construct_lattice(data);

	save_lattice_h5(data, lines, path_lat);

	free(data);
	freegscmatrixd(Ecluster, NC);
}

