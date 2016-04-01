#ifndef SASSO_DEFINITIONS_H_
#define SASSO_DEFINITIONS_H_

#include <stdio.h>
#include <stdlib.h>     
#include <string.h>

#define TAU	1e-12

typedef double Qfloat;
typedef signed char schar;												

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct data_node;
struct sasso_problem;
struct sasso_parameters;
struct sasso_model;

enum { BC, FW, MODFW, PARTAN, SWAP}; //training algorithms
enum { ZERO };//initialization methods
enum { PRIMAL, DUAL };//data representation model
enum { EXP_SINGLE_TRAIN, EXP_REGULARIZATION_PATH, TEST_INPUT_MODEL, TEST_REG_PATH, SYNC_B, TEST_ALL_REG_PATH};
enum { STOPPING_WITH_DUAL_GAP, STOPPING_WITH_INF_NORM, STOPPING_WITH_OBJECTIVE};
enum { UNIFORM, BLOCKS};
enum { LINEAR, POLY, RBF, SIGMOID, EXP, NORMAL_POLY, INV_DIST, INV_SQDIST };   /* kernel_type */

struct data_node
{
	int index;
	double value;
};

struct sasso_stats
{
	double n_iterations;
	double n_performed_dot_products;
	double n_requested_dot_products;
	double physical_time;
	
	double time_towards_random;
	double time_towards_active;
	double time_cycle_weights_FW;

};

struct sasso_problem
{
	int l;//number of original support points
	int input_dim ;//dimensionality of the support points
	double *alpha_orig;//weights of the original support points
	struct data_node **x;
	data_node* x_space;
	int elements;
	int type;//data format:sparse or dense

	bool normalized;
	double* mean_predictors;
	double* inv_std_predictors;
	double bias;
	double const_scaling_weights;
};

struct dataset
{
	int l;//number of original support points
	int input_dim ;//dimensionality of the support points
	double *y;//weights of the original support points
	struct data_node **x;
	data_node* x_space;
	int elements;
	int type;//data format:sparse or dense

};

struct sasso_parameters
{
	int exp_type;

	int kernel_type;
	int degree;         /* for poly */
	double gamma;       /* for poly/rbf/sigmoid */
	double coef0;       /* for poly/sigmoid */

	double reg_param;
	double reg_param_min;
	double reg_param_max;
	double reg_param_step;
	double eps_regularization_path;
	double n_steps_reg_path;

	bool computing_regularization_path;
	bool quick_stop_regularization_path;
	bool print_regularization_path;
	bool print_optimization_path;
	bool save_models_along_the_path;

	bool normalize;

	bool BORDER_WARM_START;
	bool ACTIVE_SET_HEURISTIC;
	double cache_size; 
	double eps;	

	int stopping_criterion;        
	bool safe_stopping;
	int nsamplings_safe_stopping; 
	int nsamplings_iterations; 
	int randomization_strategy;

	int training_algorithm;
	bool cooling;
	bool randomized;
	int sample_size;   
	int initialization_method;
	int max_iterations;

	int repetitions;
	int nfold;
	bool fixed_test;
	bool fixed_train;
	bool save_model;
	
	char* input_file_name;
	char* model_file_name;
	char* results_file_name;
	char* path_file_name;
	char* summary_exp_file_name;
	int frecuency_messages;
	char* test_data_file_name;

	bool save_file_testing_reg_path;
	char* file_testing_reg_path;
	char* file_validation_set;
	char* file_new_reg_path;
	bool syntB;

};


struct sasso_model
{
	sasso_parameters* params;	
	sasso_problem* input_problem;
	data_node *weights;		
	double bias;
	
	double delta;
	int ss;
	double l1norm;
	double l2norm_cotter;
	double obj_val;

	double *obj; 
	unsigned long int *smo_it; //for solvers using SMO
	int *greedy_it;//for greedy algorithms
	double training_time;

	bool normalized;
	double* mean_predictors;
	double* inv_std_predictors;
	
};

template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
inline double powi(double base, int times)
{
        double tmp = base, ret = 1.0;

    for(int t=times; t>0; t/=2)
	{
        if(t%2==1) ret*=tmp;
        tmp = tmp * tmp;
    }
    return ret;
}

template <class T> inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}


#endif /*SASSO_DEFINITIONS_H_*/
