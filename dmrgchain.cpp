#include <iostream>
#include <fstream>
#include <armadillo>

using namespace arma;
using namespace std;

const complex<double> I (0,1);
const double prec = 1.0E-10;

//complex vector magnitude
double mag(cx_vec v)
{
	return sqrt(real(cdot(v,v)));
}
//sets elements < prec equal to 0
vec roundoff(vec v)
{
	uvec comp = (abs(real(v)) > (prec)*ones(v.n_elem));
	vec roundv = v % comp;
	// % is element-wise multiplication
	return roundv;
}
cx_vec roundoff(cx_vec v)
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
cx_mat roundoff(cx_mat A)
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
//so functions can output eigenvalues and eigenvectors
struct eigsystem {
	vec eigvals; cx_mat eigvecs;
};
//yields correct g.s. energy but wrong eigenvector
eigsystem Lanczos(cx_mat A, int m, cx_vec guess)
{
	int n = A.n_rows;
	cx_mat v(n,m+1);
	cx_mat w(n,m+1);
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
	cx_mat T(m,m);
	T.diag(0) = a.subvec(1,m);
	//T.diag(1) = b.subvec(2,m);
	T.diag(1) = conv_to<cx_vec>::from(b.subvec(2,m));
	T.diag(-1) = T.diag(1);
	vec Tvals; cx_mat Tvecs;
	eig_sym(Tvals,Tvecs,T);
	eigsystem out;
	out.eigvals = Tvals;
	out.eigvecs = v.cols(0,m-1)*Tvecs;
	out.eigvecs.col(0) = v.cols(1,m)*Tvecs.col(0);
	//eig_sym(evals,evecs,A);
	//cout << "Lanczos: " << lvals(0) << endl;
	//cout << "Actual: " << evals(0) << endl;
	//sort(lvals).print("L");
	//sort(evals).print("e");
	return out;
}
struct block {
	int dim;
	cx_mat sp, sm, sz, H, id;
};

//global onsite 2x2 operators
cx_mat sx, sy, sz, sm, sp, id;
//kronecker product multiple matrices
cx_mat kron(cx_mat* Alist, int len)
{
	if(len == 1)
		return Alist[0];
	Alist[len-2] = kron(Alist[len-2],Alist[len-1]);
	return kron(Alist,len-1);
}
//appends new site to the right of block operator
cx_mat joinSiteR(cx_mat stot, cx_mat idtot, cx_mat ssite)
{
	return kron(idtot,ssite);
}
//appends new site to the left of block operator
cx_mat joinSiteL(cx_mat stot, cx_mat idtot, cx_mat ssite)
{
	return kron(ssite,idtot);
}
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
class dmrglist {
	private:
	
	public:
		const int sites = 200;
		const int mMax = 14;
		int m, lastm;
		int lsteps, maxlsteps;
		block *sys; block *env;
		//matrix product state arrays
		vec *Lambda; cx_mat **GammaL; cx_mat **GammaR;
		//ground state
		cx_mat gsMat; cx_vec gs; double gsEnergy;
		//coupling strength
		double J = 1;
		//transverse field strength
		double h = 0;
		//xy coupling strength
		double gxy = 1;
	//initialization
	dmrglist()
	{
		//initialize, this will be input in the future
		m = 4;
		lastm = 2;
		//number of lanczos steps
		lsteps = 5;
		maxlsteps = 60;
		//printing precision
		cout.precision(6);
		
		//initialize system and environment block lists
		//block *sys; block *env;
		sys = new block[sites];
		env = new block[sites];
		//MPS lists
		//cx_mat *Lambda; cx_mat *GammaL; cx_mat *GammaR;
		Lambda = new vec[sites];
		GammaL = new cx_mat*[sites];
		GammaR = new cx_mat*[sites];
		gsEnergy = 0;
		//initialize system and enviroment blocks
		
		sys[0].H = zeros<cx_mat>(2,2); env[0].H = zeros<cx_mat>(2,2);
		sys[0].sp = sp; sys[0].sm = sm; sys[0].sz = sz;
		env[0].sp = sp; env[0].sm = sm; env[0].sz = sz;
		sys[0].id = id; env[0].id = id;
		sys[0].dim = 4; env[0].dim = 4;
		
		//two site hamiltonian
		cx_mat Htwo = J*kron(sz,sz) + J*gxy*0.5*(kron(sp,sm)+kron(sm,sp))
			+ h*J*0.5*(kron(sp,id)+kron(sm,id)+kron(id,sp)+kron(id,sm));
		//exactly diagonalize two site hamiltonian
		cx_mat evecs; vec evals;
		eig_sym(evals,evecs,Htwo);
		cx_mat gsMatTwo = cx_mat(Htwo.n_rows,Htwo.n_rows);
		gsMatTwo = reshape(evecs.col(0),gsMatTwo.n_rows,gsMatTwo.n_cols).st();
		//printCx(gsMatTwo,"gsMat");
		//boundary MPS
		cx_mat U, V; vec lamb;
		svd(U,lamb,V,gsMatTwo);
		//keep only Schmidt values > precision
		//rounds values < precision to 0
		lamb = roundoff(lamb);
		//counts nonzero elements of lambda
		int j;
		for(j=0;j<lamb.n_elem && lamb(j) > 0; j++)
		{} //it's counting
		//set m equal to the number of nonzero schmidt values
		if(j<= mMax)
			m = j;
		lamb = lamb.subvec(0,m-1);
		U = U.cols(0,m-1);
		V = V.cols(0,m-1);
		//printCx(U,"U");printCx(V,"V");
		cx_mat Vdag = trans(V); 
		GammaL[0] = new cx_mat[2];
		GammaR[0] = new cx_mat[2];
		GammaL[0][0] = U.rows(0,0);
		GammaL[0][1] = U.rows(2,2);
		//or columns 0, 2 for R
		GammaR[0][0] = Vdag.cols(0,0);
		GammaR[0][1] = Vdag.cols(1,1);
		Lambda[0] = lamb;
		lastm = m;
	}
	~dmrglist(){}

//appends new site (on right) to system block operators
block makeNewSys(block init, block prime)
{
	prime.sz = joinSiteR(init.sz,init.id,sz);
	prime.sp = joinSiteR(init.sp,init.id,sp);
	prime.sm = joinSiteR(init.sm,init.id,sm);
	prime.id = eye<cx_mat>(prime.sz.n_rows,prime.sz.n_rows);
	return prime;
}
//appends new site (on left) to environment block operators
block makeNewEnv(block init, block prime)
{
	prime.sz = joinSiteL(init.sz,init.id,sz);
	prime.sp = joinSiteL(init.sp,init.id,sp);
	prime.sm = joinSiteL(init.sm,init.id,sm);
	prime.id = eye<cx_mat>(prime.sz.n_rows,prime.sz.n_rows);
	return prime;
}

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
	Q = roundoff(Q);
	return Q;
}
//update of operators given a matrix with which to transform with
block updateBlock(block prime, cx_mat Q)
{
	block next;
	int m = Q.n_cols;
	next.H = trans(Q)*prime.H*Q;
	next.sz = trans(Q)*prime.sz*Q;
	next.sp = trans(Q)*prime.sp*Q;
	next.sm = trans(Q)*prime.sm*Q;
	next.id = eye<cx_mat>(m,m);
	next.H = roundoff(next.H);
	next.sz = roundoff(next.sz);
	next.sp = roundoff(next.sp);
	next.sm = roundoff(next.sm);
	return next;
}

//si x sj term in hamiltonian (not used)
cx_mat hpart(int i, int j, const int sites, cx_mat si, cx_mat sj)
{
	cx_mat *termList;
	termList = new cx_mat[sites];
	for(int k = 0; k<sites;k++)
	{
		termList[k] = eye<cx_mat>(2,2);
		if(k == i)
			termList[k] = si;
		if(k == j)
			termList[k] = sj;
	}
	return kron(termList,sites);
}
void nextMPS(block sysp, block envp, int i)
{	
	vec lamb; cx_mat U,V, Vdag;
	
	//RG step by schmidt value decomposition (doesn't work for
	//Heisenberg chain for some reason)	
	svd(U,lamb,V,gsMat);
	//keep only Schmidt values > precision
	//rounds values < precision to 0
	lamb = roundoff(lamb);
	//counts nonzero elements of lambda
	int j;
	for(j=0;j<lamb.n_elem && lamb(j) > 0; j++)
	{} //it's counting
	//set m equal to the number of nonzero schmidt values
	if(j<= mMax)
		m = j;
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
	rhoR = roundoff(rhoR);
	rhoL = roundoff(rhoL);
	U = diagRho(rhoR,m,0);
	V = diagRho(rhoL,m,1);
	*/
	
	U = roundoff(U); V = roundoff(V);	
	//U = fliplr(U);
	//printCx(U,"U");
	//printCx(V,"V");
	sys[i+1] = updateBlock(sysp,U);
	env[i+1] = updateBlock(envp,V);
	
	Lambda[i+1] = lamb;
	
	GammaL[i+1] = new cx_mat[2];
	GammaR[i+1] = new cx_mat[2];

	GammaL[i+1][0] = eye<cx_mat>(lastm,m);
	GammaL[i+1][1] = eye<cx_mat>(lastm,m);
	for(int j=0;j<lastm;j++)
	{
		GammaL[i+1][0].row(j) = U.row(2*j);
		GammaL[i+1][1].row(j) = U.row(2*j+1);
	}
	GammaR[i+1][0] = eye<cx_mat>(m,lastm);
	GammaR[i+1][1] = eye<cx_mat>(m,lastm);
	Vdag = trans(V);
	//Vdag = fliplr(Vdag);
	//printCx(U,"U");
	//printCx(Vdag,"Vdag");
	/*for(int j=0;j<lastm;j++)
	{
		GammaR[i+1][0].col(j) = Vdag.col(2*j);
		GammaR[i+1][1].col(j) = Vdag.col(2*j+1);
	}*/
	GammaR[i+1][0] = Vdag.cols(0,lastm-1);
	GammaR[i+1][1] = Vdag.cols(lastm,2*lastm-1);
	
	GammaL[i+1][0] = roundoff(GammaL[i+1][0]);
	GammaL[i+1][1] = roundoff(GammaL[i+1][1]);
	GammaR[i+1][0] = roundoff(GammaR[i+1][0]);
	GammaR[i+1][1] = roundoff(GammaR[i+1][1]);
	Lambda[i+1].print("Lambda");
	//printCx(GammaL[i+1][0],"GL0");
	//printCx(GammaL[i+1][1],"GL1");
	//printCx(GammaL[i+1][0],"GR0");
	//printCx(GammaL[i+1][1],"GR1");
}

void iDMRG()
{
	//main loop (iDMRG)
	int i = 0;
	for(i = 0 ; i<sites-1;i++)
	{
		//add a site to sys, env hamiltonians
		block sysp, envp;
		sysp.H = kron(sys[i].H,id);
		//x,y coupling
		sysp.H += gxy*J*0.5*(kron(sys[i].sp,sm) + kron(sys[i].sm,sp));
		//z coupling
		sysp.H += J*kron(sys[i].sz,sz);
		//transverse field
		sysp.H += h*J*0.5*(kron(sys[i].sp,id)+kron(sys[i].sm,id)
			+ kron(sys[i].id,sp)+kron(sys[i].id,sm));
		
		envp.H = kron(id,env[i].H);
		//x,y coupling
		envp.H += gxy*J*0.5*(kron(sp,env[i].sm) + kron(sm,env[i].sp));
		//z coupling
		envp.H += J*kron(sz,env[i].sz);
		//transverse field
		envp.H += h*J*0.5*(kron(sp,env[i].id)+kron(sm,env[i].id)
			+ kron(id,env[i].sp)+kron(id,env[i].sm));
		//append site to spin operators
		sysp = makeNewSys(sys[i],sysp); 
		envp =  makeNewEnv(env[i],envp);
		
		//combine system + environment to make (2m)^2 x (2m)^2 hamiltonian
		cx_mat superH;
		/*superH = kron(sysp.H,envp.id) + kron(sysp.id,envp.H)
			+ 0.5*(kron(sysp.sp,envp.sm) + kron(sysp.sm,envp.sp))
				+ kron(sysp.sz,envp.sz);*/
		superH = kron(sysp.H,envp.id) + kron(sysp.id,envp.H);
		//xy coupling
		superH +=  gxy*J*0.5*(kron(kron(sys[i].id,sp),kron(sm,env[i].id)) 
			+ kron(kron(sys[i].id,sm),kron(sp,env[i].id)));
		//z coupling
		superH += J*kron(kron(sys[i].id,sz),kron(sz,env[i].id));
		//diagonalize superH
		eigsystem energy;
		eigsystem exact;
		
		//exact diagonalization
		//eig_sym(energy.eigvals, energy.eigvecs, superH);
		
		//generate best guess for ground state by "rotating" MPS
		
		cx_mat UL, VL, UR, VR; vec sL, sR;
		cx_mat A = join_cols(GammaL[i][0],GammaL[i][1]);
		cx_mat B = join_rows(GammaR[i][0],GammaR[i][1]);
		svd(UL,sL,VL,A); svd(UR,sR,VR,B);
		//UL.set_size(A.n_rows,A.n_rows);

		cx_mat A2 = diagmat(Lambda[i])*UR;
		cx_mat B2 = trans(VL)*diagmat(Lambda[i]);
		/*
		printDim(A,"A");printDim(B,"B");
		printDim(UL,"UL");cout << "sL " << sL.n_elem<<endl;printDim(trans(VL),"VLdag");
		printDim(UR,"UR");cout << "sR " << sR.n_elem<<endl;printDim(trans(VR),"VRdag");
		printDim(A2,"A2");printDim(B2,"B2");
		* */
		/*printCx(B,"B");
		Lambda[i].print("L");
		printCx(UL,"UL");
		sL.print("sL");
		printCx(trans(VL),"VLdag");
		printCx(UR,"UR");
		sR.print("sR");
		printCx(trans(VR),"VRdag");
		printCx(A2,"A2");
		printCx(B2,"B2");
		*/
		//Lambda[i].print("L");
		//sL.print("sL");
		//compute Lambda^-1:  / is element wise division

		mat lambInv = diagmat(Lambda[i].ones() / Lambda[i]);
		//sR.resize(m); sL.resize(m);
		sR.resize(lastm); sL.resize(lastm);
		cx_mat guessMat  = A2*diagmat(sR)*lambInv*diagmat(sL)*B2;
		//cout << "dimguess " << guessMat.n_rows << " " << guessMat.n_cols << endl;
		//cout << "ham dim " << superH.n_rows << endl;
		cx_vec guess = randu<cx_vec>(superH.n_rows);
		if(guessMat.n_rows*guessMat.n_rows == superH.n_rows)
		{
			//cx_vec guess = vectorize(guessMat,1); //1 indicates row by row
			for(int j=0;j<guessMat.n_rows;j++)
				for(int k=0;k<guessMat.n_cols;k++)
					guess[k%guessMat.n_rows+j*guessMat.n_cols] = guessMat(j,k);
			//cout << "guess" << endl;
		}
		
		/*real(roundoff(gsMat)).print("gsMat");
		guessMat.print("guess");
		guess.print("guessvec");*/
		//loop until desired precision is reached (lazy man's restarted lanczos)
		/*
		double solprec = 1;
		while(solprec > prec)
		{
			//Lanczos diagonalization
			energy = Lanczos(superH,lsteps, guess);
			
			gsEnergy = energy.eigvals[0]/(2+2*i);
			gs = energy.eigvecs.col(0);
			
			//cout << "gs mag " << mag(gs) << endl;
			//debug
			//check if gs is an eigenvector
			//((superH - energy.eigvals[0]*eye(superH.n_rows,superH.n_cols))*gs).print();
			
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
		*/
		//exact diagonalization
		eig_sym(energy.eigvals,energy.eigvecs,superH);
		gsEnergy = energy.eigvals[0]/(2+2*i);
		gs = energy.eigvecs.col(0);
		//construct reduced density matrices
		gsMat = cx_mat(sysp.H.n_rows,envp.H.n_rows);
		//ground state written as matrix with
		//rows indexing the system and columns indexing the environment
		//(or vice versa)
		
		/*
		for(int j=0;j<gsMat.n_rows;j++)
			for(int k=0;k<gsMat.n_cols;k++)
				gsMat(j,k) = gs[k%gsMat.n_rows+j*gsMat.n_cols];
		*/
		
		//equivalent to above; st is a transpose w/o conjugation
		gsMat = reshape(gs,gsMat.n_rows,gsMat.n_cols).st();
		//construct new matrices in MPS from ground state matrix
		nextMPS(sysp,envp,i);
		
		//check convergence of density matrix
		/*cx_mat checkConv;
		mat rhoA = diagmat(Lambda[i+1]%Lambda[i+1]);
		checkConv = GammaL[i+1][0]*rhoA*trans(GammaL[i+1][0])
		 + GammaL[i+1][1]*rhoA*trans(GammaL[i+1][1])
		 - diagmat(Lambda[i]%Lambda[i]);
		checkConv = roundoff(checkConv);
		printCx(checkConv,"conv");*/
		 
		
		//env[i+1] = updateBlock(sysp,rhoR,m,lsteps,0);
		//env[i+1].H = flipud(fliplr(sys[i+1].H));
		//env[i+1].sm = flipud(fliplr(sys[i+1].sp));
		//env[i+1].sp = flipud(fliplr(sys[i+1].sm));
		//env[i+1].sz = flipud(fliplr(sys[i+1].sz));
		//printCx(roundoff(gsMat),"gsMat");
		//print ground state energy
		cout << i << " energy " << gsEnergy << endl;
		//store last m value
		lastm = m;
		//increase m if m equals the number eigenvalues of rho
		if(m<mMax && m == gsMat.n_rows)
			m = m+1;
		
	}
	//cout << gs;
	
}

};
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
	
	//define (global) spin matrices
	sx << 0 << 1 << endr << 1 << 0 << endr;
	sy << 0 << -I << endr << I << 0 << endr;
	sz << 1 << 0 << endr << 0 << -1 << endr;
	//sx = sx/2.; sy = sy/2.; sz = sz/2.;
	sp = sx + I*sy; sm = sx - I*sy;
	id = eye<cx_mat>(2,2);
	
	dmrglist dmrg;
	dmrg.iDMRG();
	//check L,R normalization
	/*for(int i=0;i<dmrg.sites;i++)
	{
		cx_mat Lid = 
		trans(dmrg.GammaL[i][0])*dmrg.GammaL[i][0] 
		+ trans(dmrg.GammaL[i][1])*dmrg.GammaL[i][1];
		cx_mat Rid = 
		dmrg.GammaR[i][0]*trans(dmrg.GammaR[i][0])
		 + dmrg.GammaR[i][1]*trans(dmrg.GammaR[i][1]);
		Lid = roundoff(Lid);
		Rid = roundoff(Rid);
		printCx(Lid,"L");
		printCx(Rid,"R");
	}*/
	//print MPS
	/*
	for(int i=0;i<dmrg.sites;i++)
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
		int sites = dmrg.sites;
		cout << "Writing to " << outFileName << endl;
		outFile.open(outFileName);
		outFile.setf(ios::fixed,ios::floatfield);
		//outFile.precision(dbl::digits10);
		real(dmrg.sys[sites-1].H).raw_print(outFile,"H");
		outFile << endl;
		real(dmrg.sys[sites-1].sp).raw_print(outFile,"sp");
		outFile << endl;
		real(dmrg.sys[sites-1].sm).raw_print(outFile,"sm");
		outFile << endl;
		real(dmrg.sys[sites-1].sz).raw_print(outFile,"sz");
		outFile.close();
	}
	return 0;
}


