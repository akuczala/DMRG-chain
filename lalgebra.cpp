#include <armadillo>
#include "lalgebra.h"
using namespace arma;
using namespace std;
//complex vector magnitude
double mag(cx_vec v)
{
	return sqrt(real(cdot(v,v)));
}
double mag(vec v)
{
	return sqrt(dot(v,v));
}
//sets elements < prec equal to 0
vec roundoff(vec v, double prec)
{
	uvec comp = (abs(real(v)) > (prec)*ones(v.n_elem));
	vec roundv = v % comp;
	// % is element-wise multiplication
	return roundv;
}
cx_vec roundoff(cx_vec v, double prec)
{
	uvec compRe = (abs(real(v)) > (prec)*ones(v.n_elem));
	uvec compIm = (abs(imag(v)) > (prec)*ones(v.n_elem));
	//for rounding nonzero numbers to precision
	//vec realv = floor(real(v)*(1/prec))*prec;
	//vec imagv = floor(imag(v)*(1/prec))*prec;
	vec realv = real(v) % compRe;
	vec imagv = imag(v) % compIm;
	cx_vec roundv = cx_vec(realv,imagv);
	// % is element-wise multiplication
	return roundv;
}
//sets elements < prec equal to 0
cx_mat roundoff(cx_mat A,  double prec)
{
	/*make matrix with elements 1,0 if element is larger/smaller than
	precision. note that this MUST be done for both real/imag parts
	for numerical stability. otherwise small imaginary parts can
	blow up*/
	umat compRe = (abs(real(A)) > (prec)*ones(A.n_rows,A.n_cols));
	umat compIm = (abs(imag(A)) > (prec)*ones(A.n_rows,A.n_cols));
	//for rounding nonzero numbers to precision
	//mat realA = floor(real(A)*(1/prec))*prec;
	//mat imagA = floor(imag(A)*(1/prec))*prec;
	mat realA = real(A) % compRe;
	mat imagA = imag(A) % compIm;
	cx_mat roundA = cx_mat(realA,imagA);
	return roundA;
}
//print sparse matrix like a matrix
void print(sp_mat A,string s)
{
	cout << s << endl;
	for(int i=0;i<A.n_rows;i++)
	{
		for(int j=0;j<A.n_cols;j++)
			cout << A(i,j) << " ";
		cout << endl;
	}
}
			
//kronecker list of matrices
cx_mat kron(cx_mat* Alist, int len)
	{
		if(len == 1)
			return Alist[0];
		Alist[len-2] = kron(Alist[len-2],Alist[len-1]);
		return kron(Alist,len-1);
	}
//kronecker product pair of (real) sparse matrices
sp_mat kron(sp_mat A, sp_mat B)
{
	if(A.n_rows == 0 || B.n_rows == 0 || A.n_cols == 0 || B.n_cols == 0)
		return sp_mat();
	//cout << "kron" << endl;
	int numVals = A.n_nonzero*B.n_nonzero;
	vec values(numVals);
	umat indices(2,numVals);
	//might be a faster way with col_ptrs
	int Acol = 0;
	for(int i=0;i<A.n_nonzero;i++)
	{
		int Arow = A.row_indices[i];
		while(A.col_ptrs[Acol+1] == i)
		{
			Acol++;
		}
		int Bcol = 0;
		for(int j=0;j<B.n_nonzero;j++)
		{
			int Brow = B.row_indices[j];
			while(B.col_ptrs[Bcol+1] == j)
			{
				
				Bcol++;
			}
			int index = i*B.n_nonzero + j;
			values[index] = A.values[i]*B.values[j];
			indices(0,index) = Arow*B.n_rows + Brow;
			indices(1,index) = Acol*B.n_cols + Bcol;
		}
	}
	//sp_mat out(A.n_rows*B.n_rows,A.n_cols*B.n_cols);
	sp_mat out(indices,values,A.n_rows*B.n_rows,A.n_cols*B.n_cols);
	return out;
	
}
//kronecker list of (real) sparse matrices
sp_mat kron(sp_mat* Alist, int len)
	{
		if(len == 1)
			return Alist[0];
		Alist[len-2] = kron(Alist[len-2],Alist[len-1]);
		return kron(Alist,len-1);
	}
eigsystem Lanczos(const cx_mat A, int m, cx_vec guess)
{
	int n = A.n_rows;
	cx_mat v = zeros<cx_mat>(n,m+1);
	cx_mat w = zeros<cx_mat>(n,m+1);
	//vec randvec = randu<vec>(n);
	//v.col(1) = conv_to<cx_vec>::from(randvec);
	//don't need to explicitly say v.col(0) = 0 vector
	v.col(0) = zeros<cx_vec>(n);
	v.col(1)= guess;
	//if(true || guess.n_elem == n)
	//	v.col(1) = guess;
	//else
	//	v.col(1) = randu<cx_vec>(n);
		
	//cout << "random: " << endl;
	//v.col(1).print();
	//v.col(1) = randu<cx_vec>(n);
	v.col(1) = v.col(1)/mag(v.col(1));
	//for checking orthogonality
	w.col(0) = v.col(1);
	cx_vec a(m+1);
	vec b(m+1);
	for(int j=1;j<m-1;j++)
	{
		w.col(j) = A*v.col(j);
		a(j) = cdot(w.col(j),v.col(j));
		w.col(j) = w.col(j) - a(j)*v.col(j) - b(j)*v.col(j-1);
		b(j+1) = mag(w.col(j));
		v.col(j+1) = w.col(j)/b(j+1);
	}
	w.col(m) = A*v.col(m);
	
	a(m) = cdot(w.col(m),v.col(m));
	//m x m Tridiagonal matrix T to diagonalize exactly
	cx_mat T= zeros<cx_mat>(m,m);
	T.diag(0) = a.subvec(1,m);
	//T.diag(1) = b.subvec(2,m);
	T.diag(1) = conv_to<cx_vec>::from(b.subvec(2,m));
	T.diag(-1) = T.diag(1);
	vec Tvals; cx_mat Tvecs;
	eig_sym(Tvals,Tvecs,T);
	eigsystem out;
	out.eigvals = Tvals;
	out.eigvecs = v.cols(1,m)*Tvecs;
	//out.eigvecs = v.cols(0,m-1)*Tvecs;
	out.eigvecs.col(0) = v.cols(1,m)*Tvecs.col(0);
	//eig_sym(evals,evecs,A);
	//cout << "Lanczos: " << lvals(0) << endl;
	//cout << "Actual: " << evals(0) << endl;
	//sort(lvals).print("L");
	//sort(evals).print("e");
	return out;
}
//get magnitude of sparse vector (matrix)
double spmag(sp_mat A)
{;
	return sqrt(dot(A,A));
}
eigsystem magicEigensolver(sp_mat A, int nevals)
{
	vec evals; mat evecs;
	bool success = eigs_sym(evals,evecs,A,nevals);
	if(!success)
		eig_sym(evals,evecs,fromSparse(A));
	eigsystem out;
	out.eigvals = evals;
	out.eigvecs = zeros<cx_mat>(evecs.n_rows,evecs.n_cols)+evecs;
	return out;
}
//Lanczos method for real, sparse matrix
//IS BROKEN! tridiagonal matrix not constructed correctly
eigsystem Lanczos(sp_mat A, int m, int nevals, vec guess)
{
	int n = A.n_rows;
	sp_mat v(n,m+1);
	sp_mat w(n,m+1);
	//vec randvec = randu<vec>(n);
	//v.col(1) = conv_to<cx_vec>::from(randvec);
	//don't need to explicitly say v.col(0) = 0 vector
	
	v.col(1)= guess;
	//if(true || guess.n_elem == n)
	//	v.col(1) = guess;
	//else
	//	v.col(1) = randu<cx_vec>(n);
		
	//cout << "random: " << endl;
	//v.col(1).print();
	//v.col(1) = randu<cx_vec>(n);
	v.col(1) = v.col(1)/spmag(v.col(1));
	//for checking orthogonality
	w.col(0) = v.col(1);
	vec a(m+1);
	vec b(m+1);
	for(int j=1;j<m-1;j++)
	{
		w.col(j) = A*v.col(j);
		a(j) = dot(w.col(j),v.col(j));
		w.col(j) = w.col(j) - a(j)*v.col(j) - b(j)*v.col(j-1);
		b(j+1) = spmag(w.col(j));
		v.col(j+1) = w.col(j)/b(j+1);
	}
	w.col(m) = A*v.col(m);
	
	a(m) = dot(w.col(m),v.col(m));
	//m x m Tridiagonal matrix T to diagonalize exactly
	sp_mat T(m,m);
	uvec diag0 = uvec(T.n_rows);
	uvec diag1 = uvec(T.n_rows-1);
	uvec diag2 = uvec(T.n_rows-1);
	for(int i=0;i<T.n_rows-1;i++)
	{
		diag0[i] = i; diag1[i] = i; diag2[i] = i+1;
	}
	diag0[T.n_rows-1] = T.n_rows-1;
	uvec diag = join_cols(join_cols(diag0,diag1),diag2);
	diag = join_rows(diag
					,join_cols(join_cols(diag0,diag2),diag1));
	diag = diag.t();
	diag.print("diag");
	vec Tdiags = join_cols(a.subvec(1,m),b.subvec(2,m));
	Tdiags = join_cols(Tdiags,b.subvec(2,m));
	Tdiags.print("Tdiags");
	T = sp_mat(diag,Tdiags,m,m);
	//find lowest eigenvalue
	vec Tvals; mat Tvecs;
	T.print("T");
	eigs_sym(Tvals,Tvecs,T,nevals);
	eigsystem out;
	out.eigvals = Tvals;
	out.eigvecs = zeros<cx_mat>(A.n_rows,nevals) + v.cols(1,m)*Tvecs;
	//eig_sym(evals,evecs,A);
	//cout << "Lanczos: " << lvals(0) << endl;
	//cout << "Actual: " << evals(0) << endl;
	//sort(lvals).print("L");
	//sort(evals).print("e");
	return out;
}
sp_mat toSparse(cx_mat A) //take real part and make sparse
{
	//all indices are unsigned so the vectors can be converted to uvecs
	vector<unsigned int> row_indices; vector<unsigned int> col_indices;
	vector<double> values;
	//int nValues = 0;
	for(unsigned int row=0;row<A.n_rows;row++)
		for(unsigned int col=0;col<A.n_cols;col++)
			{
				double value = real(A(row,col));
				if(value != 0) //here I have assumed <prec have been rounded
				{
					row_indices.push_back(row);
					col_indices.push_back(col);
					values.push_back(value);
					
				}
 			}
 	if(values.size() < 1)
 	{
		return sp_mat(A.n_rows,A.n_cols);
	}
 	umat indices = join_vert(urowvec(row_indices),urowvec(col_indices));
 	vec vals = vec(values);
	sp_mat out(indices,vals,A.n_rows,A.n_cols);
	return out;	
}
sp_mat toSparse(mat A) // make sparse
{
	//all indices are unsigned so the vectors can be converted to uvecs
	vector<unsigned int> row_indices; vector<unsigned int> col_indices;
	vector<double> values;
	//int nValues = 0;
	for(unsigned int row=0;row<A.n_rows;row++)
		for(unsigned int col=0;col<A.n_cols;col++)
			{
				double value = A(row,col);
				if(value != 0) //here I have assumed <prec have been rounded
				{
					row_indices.push_back(row);
					col_indices.push_back(col);
					values.push_back(value);
					
				}
 			}
 	if(values.size() < 1)
 	{
		return sp_mat(A.n_rows,A.n_cols);
	}
 	umat indices = join_vert(urowvec(row_indices),urowvec(col_indices));
 	vec vals = vec(values);
	sp_mat out(indices,vals,A.n_rows,A.n_cols);
	return out;	
}
mat fromSparse(sp_mat A)
{
	mat out = zeros(A.n_rows,A.n_cols);
	int col = 0;
	for(int i=0;i<A.n_nonzero;i++)
	{
		double value = A.values[i];
		int row = A.row_indices[i];
		while(A.col_ptrs[col+1] == i)
			col++;
		out(row,col) = value;
	}
	return out;
}
