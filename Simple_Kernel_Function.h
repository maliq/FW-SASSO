#ifndef SIMPLE_KERNEL_FUNCTION_H_
#define SIMPLE_KERNEL_FUNCTION_H_

#include "sCache.h" 
#include "SASSO_definitions.h" 

// Gram matrix
class Simple_Kernel_Function 
{
public:

	Simple_Kernel_Function(data_node **x_support_, sasso_parameters* param_, int raw_ss_);

	~Simple_Kernel_Function(){ }

	Qfloat eval_SVM(double *weights, double bias, const data_node *x);
	Qfloat eval_SVM(data_node *weights, double bias, const data_node *x);
	Qfloat eval_SVM(data_node *weights, double bias, const data_node *x, int& ss, double& l1norm);

	Qfloat kernel_eval(int i, int j);
	Qfloat kernel_eval(int i, const data_node *other_x);

	double dot(const data_node *support_x, const data_node *other_x);
	
	unsigned long int real_kevals;
	unsigned long int requested_kevals;
	unsigned long int get_real_kevals();
	unsigned long int get_requested_kevals();
	void reset_real_kevals();
	void reset_requested_kevals();
	
private:

	sasso_parameters* param;
	data_node **x_support;	
	int raw_ss;

	//sCache *kernelCache;
	double *x_square;

	double kernel_linear(int i, int j);
	double kernel_poly(int i, int j);
	double kernel_rbf(int i, int j);
	double kernel_sigmoid(int i, int j);
	double kernel_exp(int i, int j);
    double kernel_normalized_poly(int i, int j);
	double kernel_inv_sqdist(int i, int j);
	double kernel_inv_dist(int i, int j);	    

	double kernel_linear_(int i, const data_node *other_x);
	double kernel_poly_(int i, const data_node *other_x);
	double kernel_rbf_(int i, const data_node *other_x);
	double kernel_sigmoid_(int i, const data_node *other_x);
	double kernel_exp_(int i, const data_node *other_x);
    double kernel_normalized_poly_(int i, const data_node *other_x);
	double kernel_inv_sqdist_(int i, const data_node *other_x);
	double kernel_inv_dist_(int i, const data_node *other_x);	

	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

protected:

	double (Simple_Kernel_Function::*kernel_function)(int i, int j);
	double (Simple_Kernel_Function::*kernel_function_)(int i, const data_node *other_x);
	double dotUncentered(const data_node *px, const data_node *py);

};



#endif /*SIMPLE_KERNEL_FUNCTION_H_*/