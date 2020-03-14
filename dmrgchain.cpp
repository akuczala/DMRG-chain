#include <iostream>
#include <fstream>
#include <armadillo>
#include "lalgebra.h"
#include "MPSObjects.h"

using namespace arma;
using namespace std;

const complex<double> I (0,1);

const double prec = 1.0E-7;

class Site {
	public:
	int dim;
	//Operator Z1, Z2, ZZ, X1, X2, H, id;
	Operator id, H, Z, X;
};
/*
class Block {
	public:
	int dim;
	Operator X,Z,P,M,H,id;
	//update of operators given a matrix with which to transform with
	Operator update(Operator A, cx_mat Q, string name)
	{
		cx_mat next = trans(Q)*A.matrixForm()*Q;
		next = roundoff(next,prec);
		return Operator(next,name);
	}
	//appends new site (on right) to system block operators
	Block makeNewSys(Block init, Block site)
	{
		this->Z = init.id*site.Z;
		this->P = init.id*site.P;
		this->M = init.id*site.M;
		this->X = init.id*site.X;
		this->id = Operator(eye<cx_mat>(Z.dim,Z.dim),"1");
		this->dim = Z.dim;
		return *this;
	}
	//appends new site (on left) to environment block operators
	Block makeNewEnv(Block init, Block site)
	{
		this->Z = site.Z*init.id;
		this->P = site.P*init.id;
		this->M = site.M*init.id;
		this->X = site.X*init.id;
		this->id = Operator(eye<cx_mat>(Z.dim,Z.dim),"1");
		this->dim = Z.dim;
		return *this;
	}
	Block updateBlock(cx_mat Q)
	{
		Block next;
		int m = Q.n_cols;
		next.H = update(H,Q,"H");
		next.Z = update(Z,Q,"Z");
		next.P = update(P,Q,"P");
		next.M = update(M,Q,"M");
		next.X = update(X,Q,"X");
		next.id = Operator(eye<cx_mat>(m,m),"1");
		next.dim = m;
		return next;
	}
	void print()
	{
		X.print(); Z.print();
		P.print(); M.print();
		H.print(); id.print();
	}
};*/
class Block {
	public:
	int dim;
	static const int numOps = 4;
	cx_mat mats[numOps];
	//System specific operators -------------------------
	//TFIM
	string names[numOps] = {"id","H","Z","X"};
	void siteInit()
	{
		cx_mat sx, sy, sz, sm, sp, id;
		sx << 0 << 1 << endr << 1 << 0 << endr;
		sy << 0 << -I << endr << I << 0 << endr;
		sz << 1 << 0 << endr << 0 << -1 << endr;
		//sx = sx/2.; sy = sy/2.; sz = sz/2.;
		sp = sx + I*sy; sm = sx - I*sy;
		id = eye<cx_mat>(2,2);
		mats[0] = id;
		mats[1] = zeros<cx_mat>(2,2); 
		mats[2] = sz; mats[3] = sx;
		this->dim = 2;
	}
	Site getOperators()
	{
		Site out;
		out.id = Operator(names[0]);
		out.H = Operator(names[1]);
		out.Z = Operator(names[2]);
		out.X = Operator(names[3]);
		out.dim = dim;
		return out;
	}
	/*//Double coupled TFIM
	string names[numOps] = {"id","H","Z1","Z2","X1","X2","ZZ"};
	void siteInit()
	{
		cx_mat sx, sy, sz, sm, sp, id;
		sx << 0 << 1 << endr << 1 << 0 << endr;
		sy << 0 << -I << endr << I << 0 << endr;
		sz << 1 << 0 << endr << 0 << -1 << endr;
		//sx = sx/2.; sy = sy/2.; sz = sz/2.;
		sp = sx + I*sy; sm = sx - I*sy;
		id = eye<cx_mat>(2,2);
		mats[0] = kron(id,id);
		mats[1] = zeros<cx_mat>(4,4); 
		mats[2] = kron(sz,id); mats[3] = kron(id,sz);
		mats[4] =  kron(sx,id); mats[5] =  kron(id,sx);
		mats[6] = kron(sz,sz);
		this->dim = 4;
	}
	Site getOperators()
	{
		Site out;
		out.id = Operator(names[0]);
		out.H = Operator(names[1]);
		out.Z1 = Operator(names[2]);
		out.Z2 = Operator(names[3]);
		out.X1 = Operator(names[4]);
		out.X2 = Operator(names[5]);
		out.ZZ = Operator(names[6]);
		out.dim = dim;
		return out;
	}*/
	//-------------------------------------------------
	void assignNames(string blockName)
	{
		for(int i=0;i<numOps;i++)
			names[i] = blockName + names[i];
		if(blockName == "" || blockName == "0")
			names[1] = "";
	}
	Block()
	{
		assignNames("");
		siteInit();
	}
	Block(string blockName)
	{
		assignNames(blockName);
		siteInit();
	}
	void setH(cx_mat H)
	{
		mats[1] = H;
	}
	void setId(cx_mat id)
	{
		mats[0] = id;
	}
	cx_mat getId()
	{
		return mats[0];
	}
	cx_mat getH()
	{
		return mats[1];
	}
	//update of operators given a matrix with which to transform with
	cx_mat update(cx_mat A, cx_mat Q)
	{
		cx_mat next = trans(Q)*A*Q;
		return roundoff(next,prec);
	}
	//appends new site (on right) to system block operators
	Block makeNewSys(Block init, Block site)
	{
		for(int i=0;i<numOps;i++)
			this->mats[i] = kron(init.getId(),site.mats[i]);
		this->dim = mats[0].n_rows;
		return *this;
	}
	//appends new site (on left) to environment block operators
	Block makeNewEnv(Block init, Block site)
	{
		for(int i=0;i<numOps;i++)
			this->mats[i] = kron(site.mats[i],init.getId());
		this->dim = mats[0].n_rows;
		return *this;
	}
	Block updateBlock(cx_mat Q, Block last)
	{
		int m = Q.n_cols;
		for(int i=0;i<numOps;i++)
			last.mats[i] = update(mats[i],Q);
		//next.id = eye<cx_mat>(m,m);
		last.dim = m;
		return last;
	}
	void print()
	{
		for(int i=0;i<numOps;i++)
			mats[i].print(names[i]);
	}
};

//global onsite 2x2 matrices
cx_mat sx, sy, sz, sm, sp, id;


void print(string s)
{
	cout << s << endl;
}
void printCx(cx_mat A, string s)
{
	cout << "Re " << s << endl;
	real(A).print();
	cout << "Im " << s << endl;
	imag(A).print();
}
void printDim(cx_mat A, string s)
{
	cout << s << "(" << A.n_rows << "," << A.n_cols << ")" << endl;
}
Operator writeHamiltonian(Block sysb, Block siteb, Block envb, int nSites
	, double Jc, double hc, double Kc)
{
	//constant operators
	Operator J = Operator("J");
	Operator h = Operator("h");
	Operator K = Operator("K");
	//set constant operators to 0 if 0
	if(hc < prec)
		h = Operator();
	if(Kc<prec)
		K = Operator();
	Site sys = sysb.getOperators();
	Site site = siteb.getOperators();
	Site env = envb.getOperators();
	
	//construct MPOs for TFIM
	
	MPO W1 = MPO(1,3); MPO W = MPO(3,3); MPO WL = MPO(3,1);
	
	W1.W[0][0] = sys.H + J*h*sys.X;
	W1.W[0][1] = J*sys.Z;
	W1.W[0][2] = sys.id;
	
	W.W[0][0] = site.id;
	W.W[1][0] = site.Z;
	W.W[2][0] = J*h*site.X;
	W.W[2][1] = J*site.Z; W.W[2][2] = site.id;
	
	WL.W[0][0] = env.id;
	WL.W[1][0] = env.Z;
	WL.W[2][0] = env.H + J*h*env.X;
	

	//construct MPOs for coupled chain TFIM
	
	/*MPO W1 = MPO(1,4); MPO W = MPO(4,4); MPO WL = MPO(4,1);
	W1.W[0][0] = sys.H + K*sys.ZZ+J*(h*(sys.X1+sys.X2));
	W1.W[0][1] = J*sys.Z1;
	W1.W[0][2] = J*sys.Z2;
	W1.W[0][3] = sys.id;

	W.W[0][0] = site.id;
	W.W[1][0] = site.Z1;
	W.W[2][0] = site.Z2;
	W.W[3][0] = K*site.ZZ+J*h*(site.X1+site.X2);
	W.W[3][1] = J*site.Z1; W.W[3][2] = J*site.Z2;
	W.W[3][3] = site.id;
	
	WL.W[0][0] = env.id;
	WL.W[1][0] = env.Z1;
	WL.W[2][0] = env.Z2;
	WL.W[3][0] = env.H + K*env.ZZ+J*h*(env.X1+env.X2);
	*/
	MPO WH = MPO(1,1);
	if(nSites == 0)
		WH = W1*WL;
	else
	{
		WH = W1*W;
		for(int i=1;i<nSites;i++)
		{
			WH = WH*W;
		}
		WH = WH*WL;
	}
	WH.W[0][0].print();
	return WH.W[0][0];
}
Operator writeHamiltonian(Block A, Block B, double Jc, double hc, double gxy)
{
	return writeHamiltonian(A,A,B,0,Jc,hc,gxy);
}
cx_mat buildHamiltonian(Operator H, Block sys, Block site, Block env,
							double Jc, double hc, double Kc)
{
	cx_mat J = Jc*eye<cx_mat>(1,1);
	cx_mat h = hc*eye<cx_mat>(1,1);
	cx_mat K = Jc*eye<cx_mat>(1,1);
	int numOps = sys.numOps + site.numOps + env.numOps+3;
	cx_mat * mats = new  cx_mat[numOps];
	string * names = new string[numOps];
	for(int i=0;i<sys.numOps;i++)
	{
		mats[i] = sys.mats[i];
		names[i] = sys.names[i];
	}
	for(int i=0;i<site.numOps;i++)
	{
		mats[i+sys.numOps] = site.mats[i];
		names[i+sys.numOps] = site.names[i];
	}
	for(int i=0;i<env.numOps;i++)
	{
		mats[i+sys.numOps+env.numOps] = env.mats[i];
		names[i+sys.numOps+env.numOps] = env.names[i];
	}
	mats[numOps-3] = J; names[numOps-3] = "J";
	mats[numOps-2] = h; names[numOps-2] = "h";
	mats[numOps-1] = K; names[numOps-1] = "K";
	cout << endl;
	cx_mat Hmat = H.matrixForm(names,mats,numOps);
	delete [] mats;
	delete [] names;
	return Hmat;
}
sp_mat buildSparseHamiltonian(Operator H, Block sys, Block site, Block env,
							double Jc, double hc, double Kc)
{
	print("start build");
	cx_mat J = Jc*eye<cx_mat>(1,1);
	cx_mat h = hc*eye<cx_mat>(1,1);
	cx_mat K = Jc*eye<cx_mat>(1,1);
	int numOps = sys.numOps + site.numOps + env.numOps+3;
	cx_mat * mats = new  cx_mat[numOps];
	string * names = new string[numOps];
	for(int i=0;i<sys.numOps;i++)
	{
		mats[i] = sys.mats[i];
		names[i] = sys.names[i];
	}
	for(int i=0;i<site.numOps;i++)
	{
		mats[i+sys.numOps] = site.mats[i];
		names[i+sys.numOps] = site.names[i];
	}
	for(int i=0;i<env.numOps;i++)
	{
		mats[i+sys.numOps+env.numOps] = env.mats[i];
		names[i+sys.numOps+env.numOps] = env.names[i];
	}
	mats[numOps-3] = J; names[numOps-3] = "J";
	mats[numOps-2] = h; names[numOps-2] = "h";
	mats[numOps-1] = K; names[numOps-1] = "K";
	sp_mat Hmat = H.sparseForm(names,mats,numOps);
	delete [] mats;
	delete [] names;
	print("end build");
	return Hmat;
}
class dmrglist {
	private:
	
	public:
		int steps;
		int stepsTaken;
		int minSteps;
		int mMax;
		int m, lastm;
		int lsteps, maxlsteps;
		vector <Block> sys, env;
		Operator Hsp, Hep, HSuper;
		//Block sys, env;
		Block site;
		//matrix product state arrays
		vec *Lambda; cx_mat **GammaL; cx_mat **GammaR;
		//ground state
		cx_mat gsMat; cx_vec gs; double gsEnergy;
		//coupling strength
		double J;
		//transverse field strength
		double h;
		//xy coupling strength
		double gxy;
	//initialization
	dmrglist(Block site, double J, double h, double gxy, int steps, int mMax)
	{
		this->site = site;
		this->J = J; this-> h = h; this-> gxy = gxy;
		this->steps = steps; this->mMax = mMax;
		stepsTaken = 0;
		this->minSteps = 5;
		//initialize, this will be input in the future
		m = 2*site.dim;
		lastm = site.dim;
		//number of lanczos steps
		lsteps = 6;
		maxlsteps = 60;
		//printing precision
		cout.precision(10);
		
		//initialize system and environment block lists
		//sys = new Block[steps];
		//env = new Block[steps];
		//MPS lists
		//cx_mat *Lambda; cx_mat *GammaL; cx_mat *GammaR;
		Lambda = new vec[steps];
		GammaL = new cx_mat*[steps];
		GammaR = new cx_mat*[steps];
		gsEnergy = 0;
		//initialize system and enviroment blocks
		//sys[0] = site; env[0] = site;
		sys.push_back(Block("s")); env.push_back(Block("e"));
		
		//two site hamiltonian
		Operator H= writeHamiltonian(site,site,J,h,gxy);
		//sp_mat Htwo = buildSparseHamiltonian(H,sys.back(),site,env.back(),J,h,gxy);
		cx_mat Htwo = buildHamiltonian(H,sys.back(),site,env.back(),J,h,gxy);
		//print(Htwo,"H");
		//exactly diagonalize two site hamiltonian
		eigsystem en;
		//en = magicEigensolver(Htwo,1);
		eig_sym(en.eigvals,en.eigvecs,Htwo);
		cout << "eval " << en.eigvals[0] << endl;
		
		//build gs matrix for 2 sites
		cx_mat gsMatTwo = cx_mat(Htwo.n_rows,Htwo.n_rows);
		gsMatTwo = reshape(en.eigvecs.col(0),gsMatTwo.n_rows,gsMatTwo.n_cols).st();
		//printCx(gsMatTwo,"gsMat");
		//boundary MPS
		cx_mat U, V; vec lamb;
		svd(U,lamb,V,gsMatTwo);
		//keep only Schmidt values > precision
		//rounds values < precision to 0
		lamb = roundoff(lamb,prec);
		lamb.print("lamb0");
		//counts nonzero elements of lambda
		int j;
		for(j=0;j<lamb.n_elem && lamb(j) > 0; j++)
		{} //it's counting
		//set m equal to the number of nonzero schmidt values
		if(j<= mMax)
			m = j;
		//cheating for debug
		//m = 2; lamb = ones(m);
		lamb = lamb.subvec(0,m-1);
		U = U.cols(0,m-1);
		V = V.cols(0,m-1);
		//printCx(U,"U");printCx(V,"V");
		cx_mat Vdag = trans(V); 
		cout << site.dim << endl;
		GammaL[0] = new cx_mat[site.dim];
		GammaR[0] = new cx_mat[site.dim];
		for(int j=0;j<site.dim;j++)
		{
			GammaL[0][j] = U.rows(j*site.dim,j*site.dim);
			//or columns 0, 2 for R
			GammaR[0][j] = Vdag.cols(j,j);
			//Debug
			GammaL[0][j] = eye<cx_mat>(m,m);
			GammaR[0][j] = eye<cx_mat>(m,m);
		}
		Lambda[0] = lamb;
		lastm = m;
		
		//construct hamiltonians
		Block sysp("sp"); Block envp("ep");
		Hsp = writeHamiltonian(sys.back(),site,J,h,gxy);
		Hep = writeHamiltonian(site,env.back(),J,h,gxy);
		HSuper = writeHamiltonian(sys.back(),site,env.back(),2,J,h,gxy);
	}
	~dmrglist(){}

//diagonalize rho, return unitary matrix for RG
cx_mat diagRho( cx_mat rho, int m, int blockType)
{
	vec rhoSpec; cx_mat rhoVecs;
	//exact spectrum
	eig_sym(rhoSpec,rhoVecs,rho);
	//keep (last) m eigenvectors corresponding to largest eigenvalues
	cx_mat U = rhoVecs.cols(rhoSpec.n_elem-m,rhoSpec.n_elem-1);
	//QR decomposition creates orthonormal matrix Q from eigenvectors
	cx_mat Q, R;
	bool suc = qr(Q,R,U);
	//if QR fails for some reason
	if(suc!=true)
		cout << "fail" << endl;
	//keep first m columns
	Q = Q.cols(0,m-1);
	//reverse order of columns for system
	//not exactly sure why, but column order is opposite for system/env
	if(blockType==0) 
		Q = fliplr(Q);
	//round small numbers to 0
	Q = roundoff(Q,prec);
	return Q;
}

void nextMPS(Block sysp, Block envp, int i)
{	
	vec lamb; cx_mat U,V, Vdag;
	
	//RG step by schmidt value decomposition (doesn't work for
	//Heisenberg chain for some reason)	
	svd(U,lamb,V,gsMat);
	//keep only Schmidt values > precision
	//rounds values < precision to 0
	lamb = roundoff(lamb,prec);
	lamb.print("lamb");
	//counts nonzero elements of lambda
	int j;
	for(j=0;j<lamb.n_elem && lamb(j) > 0; j++)
	{} //it's counting
	//set m equal to the number of nonzero schmidt values
	
	m = j;
	if(m > mMax)
		m = mMax;
	//debug: set m=mMax
	//m = mMax;
	lamb = lamb.subvec(0,m-1);
	cout << "m = " << m << endl;
	//lamb = lamb.subvec(0,m-1);
	U = U.cols(0,m-1);
	V = V.cols(0,m-1);
	
	//exactly diagonalize rhoL,R
	
	/*
	cx_mat rhoR = gsMat*trans(gsMat);
	cx_mat rhoL = trans(gsMat)*gsMat;
	rhoR = roundoff(rhoR,prec);
	rhoL = roundoff(rhoL,prec);
	U = diagRho(rhoR,m,0);
	V = diagRho(rhoL,m,1);
	*/

	U = roundoff(U,prec); V = roundoff(V,prec);	
	Vdag = trans(V);
	//U = fliplr(U);
	//printCx(U,"U");
	//printCx(V,"V");
	sys[0] = sysp.updateBlock(U,sys.back());
	env[0] = envp.updateBlock(V,env.back());
	
	Lambda[i+1] = lamb;
	
	GammaL[i+1] = new cx_mat[site.dim];
	GammaR[i+1] = new cx_mat[site.dim];
	
	mat lastLambInv = diagmat(ones(Lambda[i].n_elem) / Lambda[i]);
	//matrices for svd
	for(int s=0;s<site.dim;s++)
	{
		GammaL[i+1][s] = eye<cx_mat>(lastm,m);
		GammaR[i+1][s] = eye<cx_mat>(m,lastm);
		for(int j=0;j<lastm;j++)
		{
			GammaL[i+1][s].row(j) = U.row(2*j+s);
		}
		
		//Vdag = fliplr(Vdag);
		//printCx(U,"U");
		//printCx(Vdag,"Vdag");
		/*for(int j=0;j<lastm;j++)
		{
			GammaR[i+1][0].col(j) = Vdag.col(2*j);
			GammaR[i+1][1].col(j) = Vdag.col(2*j+1);
		}*/
		GammaR[i+1][s] = Vdag.cols(lastm*s,(s+1)*lastm-1);
		
		//Put in canonical form
		GammaL[i+1][s] = lastLambInv*GammaL[i+1][s];
		GammaR[i+1][s] = GammaR[i+1][s]*lastLambInv;
		
		GammaL[i+1][s] = roundoff(GammaL[i+1][s],prec);
		GammaR[i+1][s] = roundoff(GammaR[i+1][s],prec);
	}
	//Lambda[i+1].print("Lambda");
	//printCx(GammaL[i+1][0],"GL0");
	//printCx(GammaL[i+1][1],"GL1");
	//printCx(GammaL[i+1][0],"GR0");
	//printCx(GammaL[i+1][1],"GR1");
}

void iDMRG()
{
	//main loop (iDMRG)
	int i = 0;
	double lastEnergy = 0;
	double dE = 1;
	
	for(i = 0 ; i<steps-1;i++)
	{
		cout << "start loop " << i << endl;
		
		//add a site to sys, env hamiltonians
		Block sysp, envp;
		sysp = sysp.makeNewSys(sys.back(),site); //[i]
		envp = envp.makeNewEnv(env.back(),site);
		
		sysp.setH(buildHamiltonian(Hsp,sys.back(),site,env.back(),J,h,gxy));
		envp.setH(buildHamiltonian(Hep,sys.back(),site,env.back(),J,h,gxy));
		//cout << "SD" << sysp.dim << endl;
		//sysp.dim = sysp.H.dim; envp.dim = envp.H.dim;

		//combine system + environment to make (2m)^2 x (2m)^2 hamiltonian
		cx_mat superH = buildHamiltonian(HSuper,sys.back(),site,env.back(),J,h,gxy);
		/*sp_mat superH = buildSparseHamiltonian(HSuper,
			sys.back(),site,env.back(),J,h,gxy);*/
		cout << "diagonalizing superH, dim= " << superH.n_rows << endl;
		//cout << ", #entries=" << superH.n_nonzero << endl;
		
		//diagonalize superH
		eigsystem energy;
		//exact diagonalization
		//eig_sym(energy.eigvals, energy.eigvecs, superH);
		//generate guess for gs using last MPS
		mat LambdaMat = diagmat(Lambda[i]);
		mat LambdaMatm = diagmat(Lambda[i]);
		
		if(m>lastm)
			LambdaMatm.resize(m,lastm);
		if(m<lastm)
			LambdaMatm = LambdaMatm.submat(0,m-1,0,m-1);
			
		
		int thisM = sys.back().dim; //[i]
		//cout << "thisM " << sys[i].dim << endl;
		LambdaMatm.resize(thisM,thisM);
		int d = site.dim;
		cx_vec guess = cx_vec(thisM*d*d*thisM);
		cout << "m= " << m << endl;
		
		//printDim(GammaL[i][0],"GL");
		//printDim(GammaR[i][0],"GR");
		for(int s=0;s<d;s++)
		{
			GammaL[i][s].resize(thisM,thisM); 
			GammaR[i][s].resize(thisM,thisM);
		}
		//cout << "LM " << LambdaMat.n_rows << endl;
		//cout << "LMm " << LambdaMatm.n_cols << endl;
		for(int s1=0;s1<d;s1++)
		for(int s2=0;s2<d;s2++)
		{
			cx_mat guesspart = LambdaMatm*GammaL[i][s1]*LambdaMatm
							*GammaR[i][s2]*LambdaMatm.t();
			for(int a=0;a<m;a++)
				for(int g=0;g<m;g++)
					guess[g + s2*m + s1*m*d + a*m*d*d]
						= guesspart(a,g);
		}
		
		//cout << "n_elem " << guess.n_elem << endl;
		//guess.print("guess");
		
		//RANDOM GUESS
		//guess = randu<cx_vec>(superH.n_rows);
		
		
		/*real(roundoff(gsMat)).print("gsMat");
		guessMat.print("guess");
		guess.print("guessvec");*/
		//loop until desired precision is reached (lazy man's restarted lanczos)
		//real(superH).print();
		double solprec = 1;
		while(solprec > prec)
		{
			//Lanczos diagonalization
			energy = Lanczos(superH,lsteps, guess);
			gs = energy.eigvecs.col(0);
			//check precision of answer
			solprec = mag((superH - energy.eigvals[0]*eye(superH.n_rows,superH.n_cols))*gs);
			cout << "solprec " << solprec << " with " << lsteps
				<< " steps" << endl;
			//increase # lsteps if precision is not reached
			if(solprec > prec)
			{
				if(lsteps < maxlsteps)
					lsteps += 1;
				else {
					cout << "Cannot reach precision " << prec
						<< " with max Lanczos step " << maxlsteps << endl;
					return;
				}
			}
		}
		cout << "solprec " << solprec << " with " << lsteps
				<< " steps" << endl;
		
		//SPARSE
		//energy = magicEigensolver(superH,1);
		
		gsEnergy = energy.eigvals[0];
		gs = energy.eigvecs.col(0);
		//normalize GS because it doesn't alllways come out of Lanczos
			//normalized (something to look into)
		gs = gs/mag(gs);
		gs = roundoff(gs,prec);
		//exact diagonalization
		//eig_sym(energy.eigvals,energy.eigvecs,superH);
		//gsEnergy = energy.eigvals[0];
		//gs = energy.eigvecs.col(0);
		//construct reduced density matrices
		gsMat = cx_mat(sysp.dim,envp.dim);
		
		//st is a transpose w/o conjugation
		gsMat = reshape(gs,gsMat.n_rows,gsMat.n_cols).st();
		//construct new matrices in MPS from ground state matrix
		nextMPS(sysp,envp,i);
		
		//printCx(roundoff(gsMat),"gsMat");
		//print ground state energy
		
		//terminate if energy increment converges
		/*if(true && abs(dE - (lastEnergy-gsEnergy)/2.) < prec)
		{
			cout << i << " dE " << dE << endl;
			return;
		}*/
		dE = (lastEnergy-gsEnergy)/2.;
		cout << i << " dE " << dE << endl;
		lastEnergy = gsEnergy;
		//store last m value
		lastm = m;
		stepsTaken = i+1;
		//terminate if lambda converges
		if(Lambda[i].n_elem == Lambda[i+1].n_elem && i > minSteps)
		{
			vec dLambda = Lambda[i] - Lambda[i+1];
			dLambda = roundoff(dLambda,prec);
			//join_rows(Lambda[i],Lambda[i+1]).print("Lambders");
			uvec comp = (Lambda[i] - Lambda[i+1] > prec);
			if(sum(comp) == 0)
				return;
		}
		//free up some damn memory you miser
		if(i>4)
		{
			delete[] GammaL[i-3];
			delete[] GammaR[i-3];
			//Lambda[i-3].resize(1,1);
		}
		//increase m if m equals the number eigenvalues of rho
		//if(m<mMax && m == gsMat.n_rows)
			//m = m+1;
	}
	//cout << gs;
	cout << "Did not converge in " << steps << "steps" << endl;
}

};
double calcEntropy(vec lambda)
{
	vec entSpec =  lambda % lambda;
	vec logSpec = log(entSpec);
	return -sum(entSpec % logSpec);
}
int main(int argc, char** argv)
{	
	//check output file
	char* outFileName;
	ofstream outFile;
	if(argc > 1)
	{
		outFileName = argv[1];
		outFile.open(outFileName);
		if(!outFile.is_open())
		{
			cout << "Cannot open " << outFileName << endl;
			return 0;
		}
		outFile.close();
	}
	else
	{
		cout << "No output file" << endl;
	}
	
	Block site;

	//do dmrgw
	double J = 1; double h = 0.5; double gxy = 0;
	int steps = 4; int mMax = 50;
	
	dmrglist dmrg(site,J,h,gxy,steps,mMax);
	dmrg.iDMRG();
	vector<double> entropy;
	vector<double> hlist;
	for(int i=0;i<=dmrg.stepsTaken;i++)
	{
		entropy.push_back(calcEntropy(dmrg.Lambda[i]));
		hlist.push_back(i);
	}
	/*entanglement entropy as function of h
	vector<double> entropy;
	vector<double> hlist;
	vector<vec> speclist;
	double dh = 0.1;
	for(h = 0.5; h < 0.5 + dh; h += dh)
	{
		dmrglist * dmrg = new dmrglist(site,J,h,gxy,steps,mMax);
		dmrg->iDMRG();
		vec lambda = dmrg->Lambda[dmrg->stepsTaken];
		double ent = calcEntropy(lambda);
		entropy.push_back(ent);
		hlist.push_back(h);
		speclist.push_back(lambda);
		delete dmrg;
		
	}*/
	
	/*
	h = 0.1;
	dmrglist * dmrg = new dmrglist(site,J,h,gxy,steps,mMax);
	dmrg->iDMRG();
	for(int s = dmrg->stepsTaken - 1; s <= dmrg->stepsTaken; s++)
	{
		cout << "Lambda " << s << endl;
		dmrg->Lambda[s].print();
		for(int i=0;i<site.dim;i++)
		{
			cout << "Gamma L " << s << "," << i << endl;
			dmrg->GammaL[s][i].print();
			cout << "Gamma R " << s << "," << i << endl;
			dmrg->GammaR[s][i].print();
		}
	}
	delete dmrg;*/
	//check L,R normalization
	/*for(int i=0;i<dmrg.steps;i++)
	{
		cx_mat Lid = 
		trans(dmrg.GammaL[i][0])*dmrg.GammaL[i][0] 
		+ trans(dmrg.GammaL[i][1])*dmrg.GammaL[i][1];
		cx_mat Rid = 
		dmrg.GammaR[i][0]*trans(dmrg.GammaR[i][0])
		 + dmrg.GammaR[i][1]*trans(dmrg.GammaR[i][1]);
		Lid = roundoff(Lid,prec);
		Rid = roundoff(Rid,prec);
		printCx(Lid,"L");
		printCx(Rid,"R");
	}*/
	//print MPS
	/*
	for(int i=0;i<dmrg.steps;i++)
	{
		cout << i << endl;
		dmrg.Lambda[i].print("Lambda");
		printCx(dmrg.GammaL[i][0],"G0L");
		printCx(dmrg.GammaL[i][1],"G1L");
		printCx(dmrg.GammaR[i][0],"G0R");
		printCx(dmrg.GammaR[i][1],"G1R");
	}*/
	//output to file
	if(argc > 1)
	{
		cout << "Writing to " << outFileName << endl;
		outFile.open(outFileName);
		outFile.setf(ios::fixed,ios::floatfield);
		for(int i=0;i<entropy.size();i++)
		{
			cout << "h = " << hlist[i] << endl;
			//speclist[i].print();
			outFile << hlist[i] << "	" << entropy[i] << endl;
		}
	}
	return 0;
}


