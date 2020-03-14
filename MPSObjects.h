#ifndef MPSOBJECTS_H
#define MPSOBJECTS_H

#include <iostream>
#include <fstream>
#include <armadillo>


using namespace arma;
using namespace std;

class Operator {
	//operators can either be a single matrix, zero, a sum of operators
			//or a product of operators
	private:
	//cx_mat kron(cx_mat* Alist, int len);
	string toString(double a);
	public:
		int dim;
		int numOps;
		//integers demarking operator type
		int type; static const int SUM = 2; static const int PROD = 3;
		static const int MAT = 1; static const int ZERO = 0;
		string name;
		//list of operators in sum or product
		Operator* operators;
		//null constructor
		Operator();
		//other constructors
		Operator(Operator oplist[], int len, int dim);
		Operator(Operator oplist[], int len, int dim,int type);
		Operator(Operator oplist[], int len);
		Operator(string name);
		~Operator();
	private:
		//these amount to list operations. could have just extended
			//vector class probably
		static Operator pair(Operator A, Operator B);
		static Operator appendR(Operator A, Operator B);
		static Operator appendL(Operator A,Operator B);
		static Operator join(Operator A, Operator B);
	public:
		//returns product type
		static Operator tensor(Operator A, Operator B);
		//returns sum type
		static Operator sum(Operator A, Operator B);
		
		//compute matrix representation of operator
		cx_mat matrixForm(string names [], cx_mat mats [], int len);
		sp_mat sparseForm(string names [], cx_mat mats [], int len);
		void print();
		void printThis();
		
		Operator copy();
		//infix notation
		inline Operator operator+(Operator rhs)
		{
			return sum(*this,rhs);
		}
		inline Operator operator+=(Operator rhs)
		{
			return sum(*this,rhs);
		}
		inline Operator operator*( Operator rhs)
		{
			return tensor(*this,rhs);
		}
};

class MPO {
	public:
		Operator** W;
		int rows, cols;
		//null constructor
		//MPO(){};
		MPO(int rows, int cols);
		static MPO product(MPO M1, MPO M2);
		inline MPO operator*(MPO M)
		{
			return product(*this,M);
		}
	private:
	
};


#endif
