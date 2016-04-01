#include "SASSO.h"
#include "SASSO_definitions.h" 
#include "Simple_Kernel_Function.h" 
#include "TEST_kernel.h" 
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <map>

class SASSO_train
{

public:


	SASSO_train(){
		FW_SASSO_solver=NULL;
	}

	sasso_model* train(sasso_problem* prob, sasso_parameters* params,sasso_stats* stats);
	sasso_problem* readSASSOProblem(const char *filename);
	dataset* readClassificationDataSet(const char *filename);

	sasso_problem* normalize(sasso_problem* prob, sasso_model* model);
	void destroyProblem(sasso_problem* prob);
	void destroyModel(sasso_model* model);

	sasso_model** compute_regularization_path(sasso_problem* prob, sasso_parameters* params,sasso_stats* stats);
	void compute_delta_range(sasso_problem* problem, sasso_parameters* params);
	

	void parse_command_line(sasso_parameters* params, int argc, char **argv, char *input_file_name, char *model_file_name, char* results_file_name, char* path_file_name);
	void exit_with_help();
	const char* getTextTrainingAlgorithm(int code);
	const char* getModalityAlgorithm(sasso_parameters* params);

	void printParams(FILE* file, sasso_parameters* params);
	void printStats(FILE* file, sasso_stats* stats);
	void showData(sasso_problem* prob);
	char* give_me_the_time();
	
	void test_input_SVM(sasso_problem* prob,sasso_parameters* params,char* testset_file_name);
	void test_simplified_SVM(sasso_model* model, sasso_parameters* params, char* test_file_name){
		test_model(model,params,test_file_name);
	}
	void test_regularization_path(sasso_model** models, sasso_problem* input_problem, sasso_parameters* params, char* testset_file_name);
	double chooseB(double* SV_activations, TEST_Q* testQ,double best_b_now);
	void syntonizeB(sasso_model **models, sasso_problem *input_problem, sasso_parameters *params, char *path_file_name);

	void test_all_regularization_path(sasso_model** models, sasso_problem* input_problem,
									  sasso_parameters* params, std::string testset_file_name,
									  int init_step, int step_size, std::map<int, double>& model_missclass,
									  std::map<int, int>& map_support_size, std::map<int, double>& map_hinge,
									  std::map<int, double>& map_l1norm);

	double l1norm_input_SVM(sasso_problem* prob);
	void test_model(sasso_model* model, sasso_parameters* params, dataset* testset);
	void test_model(sasso_model* model, sasso_parameters* params, char* test_file_name);

	sasso_model** load_models_from_regularization_path(sasso_problem* problem, char* reg_path_filename, sasso_parameters* params, bool read_b){

		
		printf("Reading Regularization Path from = %s\n",reg_path_filename);

		std::ifstream myfile_ (reg_path_filename);
		std::string line;
		std::istringstream iss;
		double delta,l1norm,obj,value,l2norm_cotter;
		int support_size;
		int idx;
		int count_models=0;

		while (std::getline(myfile_, line))
		{
		    std::istringstream iss(line);
		    if (iss >> delta >> support_size >> l1norm >> obj >> l2norm_cotter)
		    	count_models++;
		 
		}

		printf("DETECTED %d MODELS\n",count_models);

		myfile_.close();
		sasso_model** models =  new sasso_model*[count_models];

		std::ifstream myfile (reg_path_filename);
		count_models=0;

		while (std::getline(myfile, line))
		{
		    std::istringstream iss(line);
		    std::string components;
		    std::string component1;
		    std::string component2;
		    sasso_model* model = new sasso_model();

		    if (iss >> delta >> support_size >> l1norm >> obj >> l2norm_cotter){

		    	model->delta=delta;
		    	model->ss=support_size;
		    	model->l1norm = l1norm;
		    	model->l2norm_cotter = l2norm_cotter;
		    	model->obj_val = obj;

		    	model->bias=problem->bias;
				model->params = params;
				model->input_problem = problem;		
		    	model->weights = new data_node[support_size+1];
		   		int count_features=0;
		    	while(iss >> components){
			    	std::istringstream ss(components);
			    	std::getline(ss, component1, ':');
			    	std::getline(ss, component2, ':');
			    	idx=std::stoi(component1);
			    	value=std::stod(component2);
			    	if(idx>=0){
			    		model->weights[count_features].index = idx;
			    		model->weights[count_features].value = value;
			    	} else {
			    		printf("NEGATIVE INDEX DETECTED, IDX=%d, VALUE=%g\n",idx,value);
			    		if(read_b){
			    			if(idx == -1){
			    				model->bias = value;
			    			}
			    		}
			    	}
			    	count_features++;
		    	}
		    	model->weights[support_size].index=-1;
		    	model->weights[support_size].value=0.0;

		    	models[count_models]=model;
		    	count_models++;
		    }
		 
		}

		myfile.close();
 		return models;
	}

private:

	SASSO* FW_SASSO_solver;
	void count_pattern(FILE *fp, dataset* prob, int &elements, int &type, int &dim);
	void count_pattern_SASSO(FILE *fp, sasso_problem* prob, int &elements, int &type, int &dim);

};



