#ifndef TEST_KERNEL_H_
#define TEST_KERNEL_H_

#include "mCache.h" 
#include "SASSO_definitions.h" 

// Gram matrix
class TEST_Q 
{
public:

	TEST_Q(const dataset* testset_, const sasso_problem* train_problem_, const sasso_parameters* param_,double cache_size);

	~TEST_Q()
	{
		
		if(!param->randomized)
			delete columnsCache;
		delete[] x_train;
		delete[] x_test;
	}

	Qfloat* get_KernelColumn(int train_idx);
	
	int testSVM(data_node* weights, double bias, int& mistakes, int& support_size, double& hinge, double& l1norm);

	double dot(const data_node *x, const data_node *y);

	double distanceSq(const data_node *x, const data_node *y);

	unsigned long int real_kevals;
	unsigned long int requested_kevals;
	unsigned long int get_real_kevals();
	unsigned long int get_requested_kevals();
	void reset_real_kevals();
	void reset_requested_kevals();
	
private:

	const sasso_parameters* param;
	const sasso_problem* train_problem;	
	const dataset* testset;
	
	mCache *columnsCache;
	const data_node **x_test;
	const data_node **x_train;
	double *x_test_square;
	
	double kernel_linear(int test_idx, int train_idx);
	double kernel_poly(int test_idx, int train_idx);
	double kernel_rbf(int test_idx, int train_idx);
	double kernel_sigmoid(int test_idx, int train_idx);
	double kernel_exp(int test_idx, int train_idx);
    double kernel_normalized_poly(int test_idx, int train_idx);
	double kernel_inv_sqdist(int test_idx, int train_idx);
	double kernel_inv_dist(int test_idx, int train_idx);	    

	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

protected:

	double (TEST_Q::*kernel_function)(int test_idx, int train_idx);

};



#endif /*TEST_KERNEL_H_*/