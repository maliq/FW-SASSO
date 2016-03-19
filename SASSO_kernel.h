#ifndef SASSO_KERNEL_H_
#define SASSO_KERNEL_H_

#include "sCache.h" 
#include "mCache.h" 
#include "SASSO_definitions.h" 

// Gram matrix
class SASSO_Q 
{
public:

	SASSO_Q(const sasso_problem* prob_, const sasso_parameters* param_, double *alpha_orig_);

	~SASSO_Q()
	{
		delete kernelCache;	
		if(!param->randomized)
			delete columnsCache;
		delete[] x;
		delete[] LT;
		delete[] Norms;
	}

	Qfloat* get_Q(int idx, int basisNum, int* basisIdx);
	Qfloat* get_KernelColumn(int idx);
	Qfloat kernel_eval(int idx1, int idx2);
	
	Qfloat* LT;//LT[i]=\sum_j \alpha*_j k_ij
	Qfloat* Norms;//k(i,j)

	Qfloat getLT(int idx);
	Qfloat getNorm(int idx);

	double dot(int i, int j);

	double dotUncentered(const data_node *px, const data_node *py);
	double dotCentered(const data_node *px, const data_node *py, double centerx, double centery, double factorx, double factory, int max_idx);
    double distanceSq(const data_node *x, const data_node *y);

	unsigned long int real_kevals;
	unsigned long int requested_kevals;
	unsigned long int get_real_kevals();
	unsigned long int get_requested_kevals();
	void reset_real_kevals();
	void reset_requested_kevals();
	
private:

	const sasso_parameters* param;
	const sasso_problem* prob;	
	double y2norm;
	sCache *kernelCache;
	mCache *columnsCache;
	const data_node **x;
	double *x_square;

	double kernel_linear(int i, int j);
	double kernel_poly(int i, int j);
	double kernel_rbf(int i, int j);
	double kernel_sigmoid(int i, int j);
	double kernel_exp(int i, int j);
    double kernel_normalized_poly(int i, int j);
	double kernel_inv_sqdist(int i, int j);
	double kernel_inv_dist(int i, int j);	    

	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;


	 
protected:

	double (SASSO_Q::*kernel_function)(int i, int j);

};



#endif /*SASSO_KERNEL_H_*/