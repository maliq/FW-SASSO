#include "Simple_Kernel_Function.h" 
#include <limits>
#include <cmath> 

Qfloat Simple_Kernel_Function::eval_SVM(double *weights, double bias, const data_node *x){

	double dot = 0.0;

	for(int i=0; i < raw_ss; i++){
		dot += weights[i]*kernel_eval(i,x);
	}

	dot += bias;

	return (Qfloat)dot;
}

Qfloat Simple_Kernel_Function::eval_SVM(data_node *weights, double bias, const data_node *x){

	double dot = 0.0;

	while(weights->index != -1)
	{
		int idx = weights->index;
		double weight = weights->value;
		dot += weight*kernel_eval(idx,x);
		++weights;
				
	}

	dot += bias;
	return (Qfloat)dot;
}

Qfloat Simple_Kernel_Function::eval_SVM(data_node *weights, double bias, const data_node *x, int& support_size, double& l1norm){

	double dot = 0.0;
	l1norm = 0.0;
	support_size=0;
	while(weights->index != -1)
	{
		int idx = weights->index;
		double weight = weights->value;
		dot += weight*kernel_eval(idx,x);
		++weights;
		support_size+=1;
		l1norm+=std::abs(weight);		
	}

	dot += bias;
	return (Qfloat)dot;
}


Simple_Kernel_Function::Simple_Kernel_Function(data_node **x_support_, sasso_parameters* param_, int raw_ss_)
:kernel_type(param_->kernel_type), degree(param_->degree),gamma(param_->gamma), coef0(param_->coef0), raw_ss(raw_ss_)
{

		x_support = x_support_;
		param = param_;
	
		real_kevals = 0;
		requested_kevals = 0;

		switch(kernel_type)
		{
			case LINEAR:
				kernel_function = &Simple_Kernel_Function::kernel_linear;
				kernel_function_ = &Simple_Kernel_Function::kernel_linear_;
				break;
			case POLY:
				kernel_function = &Simple_Kernel_Function::kernel_poly;
				kernel_function_ = &Simple_Kernel_Function::kernel_poly_;
				break;
			case RBF:
				kernel_function = &Simple_Kernel_Function::kernel_rbf;
				kernel_function_ = &Simple_Kernel_Function::kernel_rbf_;
				break;
			case SIGMOID:
				kernel_function = &Simple_Kernel_Function::kernel_sigmoid;
				kernel_function_ = &Simple_Kernel_Function::kernel_sigmoid_;
	        case EXP:
	            kernel_function = &Simple_Kernel_Function::kernel_exp;
	            kernel_function_ = &Simple_Kernel_Function::kernel_exp_;
				break;
	        case NORMAL_POLY:
	            kernel_function = &Simple_Kernel_Function::kernel_normalized_poly;
	            kernel_function_ = &Simple_Kernel_Function::kernel_normalized_poly_;
				break;
			case INV_DIST:
				kernel_function = &Simple_Kernel_Function::kernel_inv_dist;
				kernel_function_ = &Simple_Kernel_Function::kernel_inv_dist_;
				break;
			case INV_SQDIST:
				kernel_function = &Simple_Kernel_Function::kernel_inv_sqdist;
				kernel_function_ = &Simple_Kernel_Function::kernel_inv_sqdist_;
				break;
		}

		int i;

		//kernelCache = new sCache(param_, raw_ss);
		
		if(kernel_type == RBF || kernel_type == NORMAL_POLY || kernel_type == EXP || kernel_type == INV_DIST || kernel_type == INV_SQDIST){
			x_square = new double[raw_ss];
			for(int i=0;i<raw_ss;i++)
				x_square[i] = dot(x_support[i],x_support[i]);
		}
		else
			x_square = 0;
}

Qfloat Simple_Kernel_Function::kernel_eval(int idx1, int idx2){
		
		requested_kevals++;
		Qfloat Q;

		Q = (Qfloat)((this->*kernel_function)(idx1, idx2));			
		
		return Q;
}

Qfloat Simple_Kernel_Function::kernel_eval(int idx, const data_node *other_x){
		
		requested_kevals++;
		Qfloat Q;

		Q = (Qfloat)((this->*kernel_function_)(idx, other_x));			
		
		return Q;
}


double Simple_Kernel_Function::dot(const data_node *support_x, const data_node *other_x)
{
	return dotUncentered(support_x,other_x);
}


double Simple_Kernel_Function::dotUncentered(const data_node *px, const data_node *py)
{
	double sum = 0;
	
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += (double)px->value * (double)py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}


unsigned long int Simple_Kernel_Function::get_real_kevals(){
		return real_kevals;
	} 

unsigned long int Simple_Kernel_Function::get_requested_kevals(){
		return requested_kevals;
	} 
	
void Simple_Kernel_Function::reset_real_kevals(){
		real_kevals = 0;
	}

void Simple_Kernel_Function::reset_requested_kevals(){
		requested_kevals = 0;
	}


double Simple_Kernel_Function::kernel_linear(int i, int j)
{
	real_kevals++;
	return dot(x_support[i],x_support[j]);
}

double Simple_Kernel_Function::kernel_poly(int i, int j)
{
	real_kevals++;
	return powi(gamma*dot(x_support[i],x_support[j])+coef0,degree);
}

double Simple_Kernel_Function::kernel_normalized_poly(int i, int j)
{
	real_kevals++;
	return pow((gamma*dot(x_support[i],x_support[j])+coef0) / sqrt((gamma*x_square[i]+coef0)*(gamma*x_square[j])+coef0),degree);
}

double Simple_Kernel_Function::kernel_rbf(int i, int j)
{
	real_kevals++;
	return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x_support[i],x_support[j])));
}

double Simple_Kernel_Function::kernel_sigmoid(int i, int j)
{
	real_kevals++;
	return tanh(gamma*dot(x_support[i],x_support[j])+coef0);
}

double Simple_Kernel_Function::kernel_exp(int i, int j)
{
	real_kevals++;
	double temp=gamma*(x_square[i]+x_square[j]-2*dot(x_support[i],x_support[j]));
	return exp(-sqrt(temp));
}

double Simple_Kernel_Function::kernel_inv_sqdist(int i, int j)
{	
	real_kevals++;
	return 1.0/(gamma*(x_square[i]+x_square[j]-2*dot(x_support[i],x_support[j]))+1.0);
}

double Simple_Kernel_Function::kernel_inv_dist(int i, int j)
{
	real_kevals++;
	double temp=gamma*(x_square[i]+x_square[j]-2*dot(x_support[i],x_support[j]));
	return 1.0/(sqrt(temp)+1.0);
}

double Simple_Kernel_Function::kernel_linear_(int i, const data_node *other_x)
{
	real_kevals++;
	return dot(x_support[i],other_x);
}

double Simple_Kernel_Function::kernel_poly_(int i, const data_node *other_x)
{
	real_kevals++;
	return powi(gamma*dot(x_support[i],other_x)+coef0,degree);
}

double Simple_Kernel_Function::kernel_normalized_poly_(int i, const data_node *other_x)
{
	real_kevals++;
	double other_square = dot(other_x,other_x);
	return pow((gamma*dot(x_support[i],other_x)+coef0) / sqrt((gamma*x_square[i]+coef0)*(gamma*other_square)+coef0),degree);
}

double Simple_Kernel_Function::kernel_rbf_(int i, const data_node *other_x)
{
	real_kevals++;
	double other_square = dot(other_x,other_x);
	return exp(-gamma*(x_square[i]+other_square-2*dot(x_support[i],other_x)));
}

double Simple_Kernel_Function::kernel_sigmoid_(int i, const data_node *other_x)
{
	real_kevals++;
	return tanh(gamma*dot(x_support[i],other_x)+coef0);
}

double Simple_Kernel_Function::kernel_exp_(int i, const data_node *other_x)
{
	real_kevals++;
	double other_square = dot(other_x,other_x);
	double temp=gamma*(x_square[i]+other_square-2*dot(x_support[i],other_x));
	return exp(-sqrt(temp));
}

double Simple_Kernel_Function::kernel_inv_sqdist_(int i, const data_node *other_x)
{	
	real_kevals++;
	double other_square = dot(other_x,other_x);
	return 1.0/(gamma*(x_square[i]+other_square-2*dot(x_support[i],other_x))+1.0);
}

double Simple_Kernel_Function::kernel_inv_dist_(int i, const data_node *other_x)
{
	real_kevals++;
	double other_square = dot(other_x,other_x);
	double temp=gamma*(x_square[i]+other_square-2*dot(x_support[i],other_x));
	return 1.0/(sqrt(temp)+1.0);
}

