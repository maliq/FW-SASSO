#include "TEST_kernel.h" 
#include <limits>
#include <cmath> 


TEST_Q::TEST_Q(const dataset* testset_, const sasso_problem* train_problem_, const sasso_parameters* param_, double cache_size)
:kernel_type(param_->kernel_type), degree(param_->degree),gamma(param_->gamma), coef0(param_->coef0)
{

	
		train_problem  = train_problem_;
		testset = testset_;
		param = param_;
	
		real_kevals = 0;
		requested_kevals = 0;

		switch(kernel_type)
		{
			case LINEAR:
				kernel_function = &TEST_Q::kernel_linear;
				break;
			case POLY:
				kernel_function = &TEST_Q::kernel_poly;
				break;
			case RBF:
				kernel_function = &TEST_Q::kernel_rbf;
				break;
			case SIGMOID:
				kernel_function = &TEST_Q::kernel_sigmoid;
	        case EXP:
	            kernel_function = &TEST_Q::kernel_exp;
				break;
	        case NORMAL_POLY:
	            kernel_function = &TEST_Q::kernel_normalized_poly;
				break;
			case INV_DIST:
				kernel_function = &TEST_Q::kernel_inv_dist;
				break;
			case INV_SQDIST:
				kernel_function = &TEST_Q::kernel_inv_sqdist;
				break;
		}

		int elements = max((int)((cache_size*(1<<20))/sizeof(Qfloat)),10);
		columnsCache = new mCache(elements,testset->l);
		printf("TEST CACHE can allocate %d columns of size %d\n",elements,testset->l);
		clone(x_train,train_problem->x,train_problem->l);
		clone(x_test,testset->x,testset->l);
		
		if(kernel_type == RBF || kernel_type == NORMAL_POLY || kernel_type == EXP || kernel_type == INV_DIST || kernel_type == INV_SQDIST){
			x_test_square = new double[testset->l];
			for(int i=0;i<testset->l;i++)
				x_test_square[i] = dot(x_test[i],x_test[i]);
		}
		else
			x_test_square = 0;


}


int TEST_Q::testSVM(data_node* weights, double bias, int& mistakes, int& support_size, double& hinge, double& l1norm){
	
	double* activations = new double[testset->l];
	mistakes=0;
	support_size=0;
	hinge=0.0;
	l1norm=0.0;

	for(int test_idx=0; test_idx<testset->l; test_idx++)
			activations[test_idx] = bias;

	while(weights->index!=-1){
		Qfloat* kprods = get_KernelColumn(weights->index);
		support_size+=1;
		l1norm+=std::abs(weights->value);
		for(int test_idx=0; test_idx<testset->l; test_idx++)
			activations[test_idx] += (double)kprods[test_idx]*weights->value;
		weights++;
	}

	int predicted_class;

	for(int test_idx=0; test_idx<testset->l; test_idx++){
			double discriminant = activations[test_idx];
			predicted_class = (discriminant>=0) ? 1 : -1;
			if(predicted_class!=testset->y[test_idx])
				mistakes += 1;
			hinge += (double)testset->y[test_idx]*discriminant;
	}

	delete[] activations;
	return mistakes;
}

Qfloat* TEST_Q::get_KernelColumn(int train_idx)
	{	
		
		requested_kevals += testset->l;
		bool has_to_fill = false;

		Qfloat *Q = columnsCache->get_data(train_idx, has_to_fill);

		if (has_to_fill && (Q != NULL))
		{	

			for(int test_idx = 0; test_idx < testset->l; test_idx++)
			{
				Q[test_idx] = (Qfloat)((this->*kernel_function)(test_idx, train_idx));	
						
			}						
		}

		return Q;
	}


double TEST_Q::dot(const data_node *px, const data_node *py)
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


unsigned long int TEST_Q::get_real_kevals(){
		return real_kevals;
	} 

unsigned long int TEST_Q::get_requested_kevals(){
		return requested_kevals;
	} 
	
void TEST_Q::reset_real_kevals(){
		real_kevals = 0;
	}

void TEST_Q::reset_requested_kevals(){
		requested_kevals = 0;
	}

double TEST_Q::distanceSq(const data_node *x, const data_node *y)
{
	double sum = 0.0;
	
    while(x->index != -1 && y->index !=-1)
	{
		if(x->index == y->index)
		{
			double d = (double)x->value - (double)y->value;
			sum += d*d;
			
			++x;
			++y;
		}
		else
		{
			if(x->index > y->index)
			{
				sum += ((double)y->value) * (double)y->value;
				++y;
			}
			else
			{
				sum += ((double)x->value) * (double)x->value;
				++x;
			}
		}
	}

	while(x->index != -1)
	{
		sum += ((double)x->value) * (double)x->value;
		++x;
	}

	while(y->index != -1)
	{
		sum += ((double)y->value) * (double)y->value;
		++y;
	}
	
	return (double)sum;
}

double TEST_Q::kernel_linear(int test_idx, int train_idx)
{
	real_kevals++;
	return dot(x_test[test_idx],x_train[train_idx]);
}

double TEST_Q::kernel_poly(int test_idx, int train_idx)
{
	real_kevals++;
	return powi(gamma*dot(x_test[test_idx],x_train[train_idx])+coef0,degree);
}

double TEST_Q::kernel_normalized_poly(int test_idx, int train_idx)
{
	real_kevals++;
	double x_train_squared = dot(x_train[train_idx],x_train[train_idx]);
	return pow((gamma*dot(x_test[test_idx],x_train[train_idx])+coef0) / sqrt((gamma*x_test_square[test_idx]+coef0)*(gamma*x_train_squared)+coef0),degree);
}

double TEST_Q::kernel_rbf(int test_idx, int train_idx)
{
	real_kevals++;
	double x_train_squared = dot(x_train[train_idx],x_train[train_idx]);
	return exp(-gamma*(x_test_square[test_idx]+x_train_squared-2*dot(x_test[test_idx],x_train[train_idx])));
}

double TEST_Q::kernel_sigmoid(int test_idx, int train_idx)
{
	real_kevals++;
	return tanh(gamma*dot(x_test[test_idx],x_train[train_idx])+coef0);
}

double TEST_Q::kernel_exp(int test_idx, int train_idx)
{
	real_kevals++;
	double x_train_squared = dot(x_train[train_idx],x_train[train_idx]);
	double temp=gamma*(x_test_square[test_idx]+x_train_squared-2*dot(x_test[test_idx],x_train[train_idx]));
	return exp(-sqrt(temp));
}

double TEST_Q::kernel_inv_sqdist(int test_idx, int train_idx)
{	
	real_kevals++;
	double x_train_squared = dot(x_train[train_idx],x_train[train_idx]);
	return 1.0/(gamma*(x_test_square[test_idx]+x_train_squared-2*dot(x_test[test_idx],x_train[train_idx]))+1.0);
}

double TEST_Q::kernel_inv_dist(int test_idx, int train_idx)
{
	real_kevals++;
	double x_train_squared = dot(x_train[train_idx],x_train[train_idx]);
	double temp=gamma*(x_test_square[test_idx]+x_train_squared-2*dot(x_test[test_idx],x_train[train_idx]));
	return 1.0/(sqrt(temp)+1.0);
}

