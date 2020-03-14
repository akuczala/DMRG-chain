#ifndef LALGEBRA_H
#define LALGEBRA_H

#include <armadillo>

using namespace arma;
using namespace std;
//complex vector magnitude
double mag(cx_vec v);
double mag(vec v);
//sets elements < prec equal to 0
vec roundoff(vec v, double prec);
cx_vec roundoff(cx_vec v, double prec);
//sets elements < prec equal to 0
cx_mat roundoff(cx_mat A,  double prec);
void print(sp_mat A, string s);
cx_mat kron(cx_mat* Alist, int len);
sp_mat kron(arma::sp_mat A, arma::sp_mat B);
sp_mat kron(sp_mat* Alist, int len);
//so functions can output eigenvalues and eigenvectors
struct eigsystem {
	vec eigvals; cx_mat eigvecs;
};
eigsystem Lanczos(cx_mat A, int m, cx_vec guess);
eigsystem magicEigensolver(sp_mat A, int nevals);
double spmag(sp_mat A);
eigsystem Lanczos(sp_mat A, int m, int nevals, vec guess);

sp_mat toSparse(mat A);
sp_mat toSparse(cx_mat A);
mat fromSparse(sp_mat A);
#endif
