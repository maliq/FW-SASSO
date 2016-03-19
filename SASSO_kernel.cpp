#include "SASSO_kernel.h" 
#include <limits>
#include <cmath> 


SASSO_Q::SASSO_Q(const sasso_problem* prob_, const sasso_parameters* param_, double *alpha_orig_)
:kernel_type(param_->kernel_type), degree(param_->degree),gamma(param_->gamma), coef0(param_->coef0)
{

		int i;
		prob  = prob_;
		param = param_;
	

		real_kevals = 0;
		requested_kevals = 0;

		switch(kernel_type)
		{
			case LINEAR:
				kernel_function = &SASSO_Q::kernel_linear;
				break;
			case POLY:
				kernel_function = &SASSO_Q::kernel_poly;
				break;
			case RBF:
				kernel_function = &SASSO_Q::kernel_rbf;
				break;
			case SIGMOID:
				kernel_function = &SASSO_Q::kernel_sigmoid;
	        case EXP:
	            kernel_function = &SASSO_Q::kernel_exp;
				break;
	        case NORMAL_POLY:
	            kernel_function = &SASSO_Q::kernel_normalized_poly;
				break;
			case INV_DIST:
				kernel_function = &SASSO_Q::kernel_inv_dist;
				break;
			case INV_SQDIST:
				kernel_function = &SASSO_Q::kernel_inv_sqdist;
				break;
		}

		if(param->randomized){
			kernelCache = new sCache(param_->cache_size, prob->l);
			columnsCache = NULL;
		} else {
			kernelCache = new sCache(100, prob->l);
			int elements = max((int)((param_->cache_size*(1<<20))/sizeof(Qfloat)),10);
			columnsCache = new mCache(elements, prob->l, prob->l);
			printf("Elements is %d\n",elements);
		}
			
		clone(x,prob_->x,prob_->l);
		
		if(kernel_type == RBF || kernel_type == NORMAL_POLY || kernel_type == EXP || kernel_type == INV_DIST || kernel_type == INV_SQDIST){
			x_square = new double[prob_->l];
			for(int i=0;i<prob_->l;i++)
				x_square[i] = dot(i,i);
		}
		else
			x_square = 0;

		LT = new Qfloat[prob_->l];
		Norms = new Qfloat[prob_->l];

		for(int i=0; i < prob_->l; i++){
			LT[i]=0.0; 	
		}

		for(int j=0; j < prob_->l; j++){
			for(int i=0; i <= j; i++){
				double kij=kernel_eval(i,j);
				LT[j] += alpha_orig_[i]*kij;
				if(j!=i)
					LT[i] += alpha_orig_[j]*kij;
				if(j==i)
					Norms[i] = kij;
			}
		}

}

Qfloat SASSO_Q::getLT(int idx){
		
		return LT[idx];

}


Qfloat SASSO_Q::getNorm(int idx){

		return Norms[idx];

}

Qfloat SASSO_Q::kernel_eval(int idx1, int idx2){
		
		requested_kevals++;
		Qfloat Q;

		Q = (Qfloat)((this->*kernel_function)(idx1, idx2));			
		
		return Q;
	}

Qfloat* SASSO_Q::get_Q(int idx, int basisNum, int* basisIdx)
	{	
		
		requested_kevals += basisNum;
		
		int numRet;
		Qfloat *Q = kernelCache->get_data(idx, basisNum, numRet);
		if (Q != NULL)
		{	

			for(int i = numRet; i < basisNum; i++)
			{
				int idx2 = basisIdx[i];
				Q[i] = (Qfloat)((this->*kernel_function)(idx, idx2));	
						
			}						
		}
		return Q;
	}

Qfloat* SASSO_Q::get_KernelColumn(int idx)
	{	
		
		requested_kevals += prob->l;
		bool has_to_fill = false;

		Qfloat *Q = columnsCache->get_data(idx, has_to_fill);

		if (has_to_fill && (Q != NULL))
		{	

			for(int i = 0; i < prob->l; i++)
			{
				Q[i] = (Qfloat)((this->*kernel_function)(i, idx));	
						
			}						
		}

		return Q;
	}


double SASSO_Q::dot(int i, int j)
{
	
	if((param->normalize) && (prob->normalized))
		return dotCentered(x[i],x[j],prob->mean_predictors[i],prob->mean_predictors[j],prob->inv_std_predictors[i],prob->inv_std_predictors[j],prob->input_dim);
	else
		return dotUncentered(x[i],x[j]);
}


double SASSO_Q::dotCentered(const data_node *px, const data_node *py, double centerx, double centery, double factorx, double factory, int max_idx)
{
	double sum = 0.0;
	
	int previous_idx = 0;
	const data_node *pmin;
	double scale = factorx*factory;

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

	sum = scale*(sum-(prob->input_dim*centerx*centery));
	return sum;
}

double SASSO_Q::dotUncentered(const data_node *px, const data_node *py)
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


unsigned long int SASSO_Q::get_real_kevals(){
		return real_kevals;
	} 

unsigned long int SASSO_Q::get_requested_kevals(){
		return requested_kevals;
	} 
	
void SASSO_Q::reset_real_kevals(){
		real_kevals = 0;
	}

void SASSO_Q::reset_requested_kevals(){
		requested_kevals = 0;
	}

double SASSO_Q::distanceSq(const data_node *x, const data_node *y)
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

double SASSO_Q::kernel_linear(int i, int j)
{
	real_kevals++;
	return dot(i,j);
}

double SASSO_Q::kernel_poly(int i, int j)
{
	real_kevals++;
	return powi(gamma*dot(i,j)+coef0,degree);
}

double SASSO_Q::kernel_normalized_poly(int i, int j)
{
	real_kevals++;
	return pow((gamma*dot(i,j)+coef0) / sqrt((gamma*x_square[i]+coef0)*(gamma*x_square[j])+coef0),degree);
}

double SASSO_Q::kernel_rbf(int i, int j)
{
	real_kevals++;
	return exp(-gamma*(x_square[i]+x_square[j]-2*dot(i,j)));
}

double SASSO_Q::kernel_sigmoid(int i, int j)
{
	real_kevals++;
	return tanh(gamma*dot(i,j)+coef0);
}

double SASSO_Q::kernel_exp(int i, int j)
{
	real_kevals++;
	double temp=gamma*(x_square[i]+x_square[j]-2*dot(i,j));
	return exp(-sqrt(temp));
}

double SASSO_Q::kernel_inv_sqdist(int i, int j)
{	
	real_kevals++;
	return 1.0/(gamma*(x_square[i]+x_square[j]-2*dot(i,j))+1.0);
}

double SASSO_Q::kernel_inv_dist(int i, int j)
{
	real_kevals++;
	double temp=gamma*(x_square[i]+x_square[j]-2*dot(i,j));
	return 1.0/(sqrt(temp)+1.0);
}

