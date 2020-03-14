#include <iostream>
#include <fstream>
#include <armadillo>
#include "MPSObjects.h"
#include "lalgebra.h"

using namespace arma;
using namespace std;


/*cx_mat Operator::kron(cx_mat* Alist, int len)
{
	if(len == 1)
		return Alist[0];
	Alist[len-2] = arma::kron(Alist[len-2],Alist[len-1]);
	return kron(Alist,len-1);
}*/
string Operator::toString(double a)
{
	ostringstream strs;
	strs << a;
	return strs.str();
}
Operator::Operator()
{
	this->type = ZERO;
	this->name = "";
}
//other constructors
Operator::Operator(Operator oplist[], int len, int dim,int type)
{
	this->numOps = len;
	//this->numTerms = terms;
	this->operators = oplist;
	this->dim = dim;
	this->type = type;
}
Operator::Operator(Operator oplist[], int len, int dim)
{
	this->numOps = len;
	//this->numTerms = terms;
	this->operators = oplist;
	this->dim = dim;
}
Operator::Operator(Operator oplist[], int len)
{
	this->numOps = len;
	//this->numTerms = terms;
	this->operators = oplist;
}
Operator::Operator(string name)
{
	//this->dim = A.n_rows;
	this->type = MAT;
	this->name = name;
	if(name == "0" || name == "")
		this->type = ZERO;
}
Operator::~Operator()
{
	/*cout << "Kill" << endl;
	this->print();
	if(type==SUM || type == PROD)
		delete [] operators;*/
}
//these amount to list operations. could have just extended
	//vector class probably
Operator Operator::pair(Operator A, Operator B)
{
	
	Operator* newList = new Operator[2];
	
	newList[0] = A.copy();
	
	newList[1] = B.copy();
	
	return Operator(newList,2);
}
Operator Operator::appendR(Operator A, Operator B)
{
	//if operator A is tensor product of smaller ones, append B
		//to list
	Operator* newList = new Operator[A.numOps+1];
	for(int i=0; i<A.numOps;i++)
		newList[i] = A.operators[i].copy();
	newList[A.numOps] = B.copy();
	return Operator(newList,A.numOps+1);

}
Operator Operator::appendL(Operator A,Operator B)
{
	//if operator B is tensor product of smaller ones, append A
		//to list
	Operator* newList = new Operator[B.numOps+1];
	newList[0] = A.copy();
	for(int i=0; i<B.numOps;i++)
		newList[i+1] = B.operators[i].copy();
	return Operator(newList,B.numOps+1);

}
Operator Operator::join(Operator A, Operator B)
{
	if(A.operators != 0 && B.operators != 0)
	{
		int length = A.numOps + B.numOps;
		Operator* newList = new Operator[length];
		
		for(int i=0; i< A.numOps;i++)
			newList[i] = A.operators[i].copy();
		
		for(int i=0; i< B.numOps;i++)
			newList[i+A.numOps] = B.operators[i].copy();
		
	return Operator(newList,length);
	}else{
		cout << "Cannot join. ";
		cout << "operators of length " << A.numOps << " and "
			<< B.numOps << " are not both lists" << endl;
		return Operator();
	}
}
//returns product type
Operator Operator::tensor(Operator A, Operator B)
{
	//if either operator is zero, return zero
	if(A.type == ZERO || B.type == ZERO)
		return Operator();
	Operator newOp;
	if(A.type == PROD && B.type == PROD)
	{
		newOp = join(A,B);
	}
	if(A.type == PROD && (B.type == SUM || B.type == MAT))
	{
		newOp = appendR(A,B);
	}
	if((A.type == SUM || A.type == MAT) && B.type == PROD)
	{
		newOp = appendL(A,B);
	}
	if((A.type == SUM && (B.type == SUM || B.type == MAT)) || 
		(A.type == MAT && (B.type == SUM || B.type == MAT)))
	{
		newOp = pair(A,B);
	}
	newOp.dim = A.dim*B.dim;
	newOp.type = PROD;
	return newOp;
}
//returns sum type
Operator Operator::sum(Operator A, Operator B)
{
	//sum identity
	if(A.type == ZERO)
		return B;
	if(B.type == ZERO)
		return A;
	Operator newOp;
	/*
	if(A.type == PROD && B.type == PROD &&
		A.numOps != B.numOps)
	{
		cout << "Cannot sum. Operators have different lengths: ";
		cout << A.numOps << "," << B.numOps << endl;
		return Operator();
	}*/
	/*if(A.dim != B.dim)
	{
		cout << "Cannot sum. Operators have different dimension: ";
		cout << A.dim << "," << B.dim  << endl;
		return Operator();
	}*/
	if(A.type != SUM && B.type != SUM)
	{
		newOp = pair(A,B);
	}
	if(A.type == SUM && (B.type == PROD || B.type == MAT))
	{
		newOp = appendR(A,B);
	}
	if((A.type == PROD || A.type == MAT) && B.type == SUM)
	{
		newOp = appendL(A,B);
	}
	if((A.type == SUM && B.type == SUM))
	{
		newOp = join(A,B);
	}
	newOp.dim = A.dim;
	newOp.type = SUM;
	return newOp;
}

//compute matrix representation of operator
cx_mat Operator::matrixForm(string names [],cx_mat mats [],int len)
{
	if(type == PROD)
	{
		cx_mat matProduct = operators[0].matrixForm(names, mats, len);
		for(int i=1;i<numOps;i++)
		{
			matProduct = kron(matProduct,
				operators[i].matrixForm(names, mats, len));
		}
		return matProduct;
	}
	if(type == SUM)
	{
		cx_mat total = operators[0].matrixForm(names, mats,len);
		for(int i=1;i<numOps;i++)
		{
			total += operators[i].matrixForm(names, mats, len);
		}
		return total;
	}
	if(type == MAT)
	{
		for(int i=0;i<len;i++)
		{
			if(names[i] == name)
			{
				return mats[i];
			}
		}
	}
	//zero type returns dimxdim matrix of 0
	if(type == ZERO)
		return zeros<cx_mat>(dim,dim);
	return zeros<cx_mat>(dim,dim);
}
//compute (real) sparse matrix representation of operator
sp_mat Operator::sparseForm(string names [],cx_mat mats [],int len)
{
	if(type == PROD)
	{
		sp_mat matProduct = operators[0].sparseForm(names, mats, len);
		for(int i=1;i<numOps;i++)
		{
			matProduct = kron(matProduct,operators[i].sparseForm(names,mats,len));
		}
		return matProduct;
	}
	if(type == SUM)
	{
		sp_mat total = operators[0].sparseForm(names, mats,len);
		for(int i=1;i<numOps;i++)
		{
			total += operators[i].sparseForm(names, mats, len);
		}
		return total;
	}
	if(type == MAT)
	{
		for(int i=0;i<len;i++)
		{
			if(names[i] == name)
			{
				//cout << name << "," << mats[i].n_rows << endl;
				return toSparse(mats[i]);
			}
		}
	}
	//zero type returns dimxdim matrix of 0
	if(type == ZERO)
		return sp_mat(dim,dim);
	return sp_mat(dim,dim);
}
//make new operator by copying operator list or matrix
Operator Operator::copy()
{
	if(type == ZERO)
		return Operator();
	if(type == MAT)
		return Operator(name);
	Operator* newList = new Operator[numOps];
	for(int i=0; i<numOps;i++)
		newList[i] = operators[i].copy();
	return Operator(newList,numOps,dim,type);
		
}
void Operator::print()
{
	//puts carriage return at end of print
	printThis(); cout << endl;
}
void Operator::printThis()
{
	if(type == MAT || type == ZERO)
		cout << name;
	if(type == PROD)
	{
		for(int i=0;i<numOps-1;i++)
		{
			operators[i].printThis();
			cout << ".";
		}
		operators[numOps-1].printThis();
	}
	if(type == SUM)
	{
		cout << "(";
		for(int i=0;i<numOps-1;i++)
		{
			operators[i].printThis();
			cout << "+";
		}
		operators[numOps-1].printThis();
		cout << ")";
	}
}


MPO::MPO(int rows, int cols)
{
	this->rows = rows;
	this->cols = cols;
	W = new Operator*[rows];
	for(int i=0;i<rows;i++)
	{
		W[i] = new Operator[cols];
	}
}
MPO MPO::product(MPO M1, MPO M2)
{
	Operator ** W1 = M1.W;
	Operator ** W2 = M2.W;
	if(M1.cols != M2.rows)
	{
		cout << "MPO row/col mismatch " << M1.cols << ","
			<< M2.rows << endl;
	}
	MPO out = MPO(M1.rows,M2.cols);
	for(int i=0;i<M1.rows;i++)
	for(int k=0;k<M2.cols;k++)
	{
		Operator sum = W1[i][0]*W2[0][k];
		for(int j=1;j<M1.cols;j++)
		{
			sum = sum + W1[i][j]*W2[j][k];
		}
		out.W[i][k] = sum;
	}
	return out;	
}

