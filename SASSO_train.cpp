#include "SASSO_train.h"
#include <stdio.h>
#include <cmath> 
#include <ctype.h>
#include <stdexcept>
#include <string.h>
#include <math.h>
#include <limits>
#include <stdexcept>
#include <map>

#define N_STEPS_PATH 10.0
#define FRAC_MIN_DELTA 10000.0
#define N_BES_SYNC 1000

double SASSO_train::l1norm_input_SVM(sasso_problem* prob){

double norm = 0.0;

for(int i=0; i < prob->l; i++)
	norm += std::abs(prob->alpha_orig[i]);

printf("l1norm input SVM is %f\n",norm);

return norm;

}

void SASSO_train::test_input_SVM(sasso_problem* prob, sasso_parameters* params, char* testset_file_name){

	dataset* testset = this->readClassificationDataSet(testset_file_name);
	
	l1norm_input_SVM(prob);

	Simple_Kernel_Function* kernel = new Simple_Kernel_Function(prob->x, params, prob->l);

	double discriminant;
	int predicted_class;
	double mistakes = 0.0;
	double hinge_loss = 0.0;

	clock_t time_s = clock ();
	int size_block = 1000;
	for(int i=0; i < testset->l; i++){

		if((i%size_block==0) & (i>0)){
			double thetime = ((float)(clock() -time_s)/CLOCKS_PER_SEC);
			printf("TESTING ... %d done, average testing time is: %lf (s) \n",i,thetime/(double)size_block);
			time_s = clock ();
		}
		discriminant = kernel->eval_SVM(prob->alpha_orig, prob->bias, testset->x[i]);
		predicted_class = (discriminant>=0) ? 1 : -1;
		if(predicted_class!=testset->y[i])
			mistakes += 1.0;
		hinge_loss += testset->y[i]*discriminant;
	}

	printf("Average Hinge Loss = %f\n",hinge_loss/(double)testset->l);
	printf("Misclassification Rate = %f\n",mistakes/(double)testset->l);
	printf("Number of Support Vectors (prob->l) = %d\n",prob->l);
}

void SASSO_train::test_model(sasso_model* model, sasso_parameters* params, dataset* testset){

	Simple_Kernel_Function* kernel = new Simple_Kernel_Function(model->input_problem->x, params, model->input_problem->l);

	double discriminant;
	int predicted_class;
	double mistakes = 0.0;
	double hinge_loss = 0.0;

	//printf("STARTING TESTING ...\n");
	int ss=0;
	double l1norm=0.0;
	for(int i=0; i < testset->l; i++){

		discriminant = kernel->eval_SVM(model->weights, model->bias, testset->x[i], ss, l1norm);
		predicted_class = (discriminant>=0) ? 1 : -1;
		if(predicted_class!=testset->y[i])
			mistakes += 1.0;
		hinge_loss += testset->y[i]*discriminant;
	}

	printf("Number of Support Vectors = %d\n",ss);
	//printf("Average Hinge Loss = %f\n",hinge_loss/(double)testset->l);
	printf("Misclassification Rate = %f\n",mistakes/(double)testset->l);
	printf("L1 norm = %g\n",l1norm);

}

double SASSO_train::chooseB(double* SV_activations, TEST_Q* testQ, double best_b_now){

	double b_ = 0.0;
	double act_max = SV_activations[0];
	double act_min = SV_activations[0];

	int l = testQ->getNtest();

	bool verbose = false;

	for(int test_idx=0; test_idx<l; test_idx++){
		if(SV_activations[test_idx]>act_max)
			act_max = SV_activations[test_idx];
		if(SV_activations[test_idx]<act_min)
			act_min = SV_activations[test_idx];
	}

	double b_min  = -1.0*act_max;
	double b_max  = -1.0*act_min;
	printf("TEST N = %d\n", l);
	printf("B_MIN = %g\n",b_min);
	printf("B_MAX = %g\n",b_max);

	int mistakes__=0;
	double hinge__ = 0.0;

	int n_b = N_BES_SYNC;
	double step = (b_max - b_min)/((double)n_b);
	printf("B_STEP = %g\n",step);

	double b_candidate = b_min - step;
	double best_b = best_b_now;
	int best_error = testQ->testSVM(SV_activations,best_b,mistakes__,hinge__);

	printf("CURRENT B = %g\n",best_b);
	printf("CURRENT ERROR = %d\n",best_error);
	
	for(int b_idx=0;b_idx<n_b;b_idx++){
		b_candidate = b_candidate + step;
		int error = testQ->testSVM(SV_activations,b_candidate,mistakes__,hinge__);
		if(error < best_error){
			if(verbose)
				printf("BETTER B ... B= %g, ERROR=%d, CURRENT_BEST_ERR=%d, CURRENT_BEST_B=%g\n",b_candidate,error,best_error,best_b);
			best_error = error;
			best_b = b_candidate;
		}
	}
	printf("LAST B EXPLORED = %g\n",b_candidate);
	printf("LAST ERROR = %d\n",best_error);

	b_ =  best_b;
	printf("BEST_B = %g\n",b_);

	return b_;

}

void SASSO_train::syntonizeB(sasso_model **models, sasso_problem* input_problem, sasso_parameters* params, char* path_file_name){

	dataset* testdata;
	char* tunning_file_name;
	if( params->file_validation_set!=NULL ){
		printf("READING VALIDATION DATA ...\n");
		tunning_file_name = params->file_validation_set;
	} else {
		printf("READING TEST DATA ...\n");
		tunning_file_name = params->file_testing_reg_path;
	}
	printf("%s\n", tunning_file_name);
	testdata = this->readClassificationDataSet(tunning_file_name);
	
	printf("DONE ...\n");

	char new_results_filename[8192];
	sprintf(new_results_filename,"%s.---SYNC-B---.txt",path_file_name);
	FILE* new_results_file = fopen(new_results_filename,"w");   
 	
	TEST_Q* testQ = new TEST_Q(testdata,input_problem,params,params->cache_size);
	
	FILE* save_testing=NULL;
	if(params->save_file_testing_reg_path)
		save_testing=fopen(params->file_testing_reg_path,"w");

	int support_size;
	int mistakes;
	double hinge_loss;
	double l1norm;

	printf("EXPLORING PATH ... N STEPS=%d\n",(int)params->n_steps_reg_path);
	

	clock_t time_start = clock ();

	for(int i=0; i < (int)params->n_steps_reg_path; i++){

				sasso_model* model = models[i];
				data_node* weights = model->weights;
				printf("SYNTONIZING B FOR MODEL %d ...\n",i);
	
				double* SV_activations = testQ->getSVActivations(weights,support_size,l1norm);

				double b_synt = chooseB(SV_activations,testQ,model->bias);
				model->bias = b_synt;

				if(save_testing!=NULL)
					mistakes = testQ->testSVM(SV_activations,b_synt,mistakes,hinge_loss);

				delete(SV_activations);
				
				fprintf(new_results_file,"%g", model->delta);
				fprintf(new_results_file," %d",model->ss);
				fprintf(new_results_file," %g",model->l1norm);
				fprintf(new_results_file," %g",model->obj_val);
				fprintf(new_results_file," %g",model->l2norm_cotter);
				int j=0;

				while(weights[j].index!=-1){
					fprintf(new_results_file," %d:%g",weights[j].index,weights[j].value);
					j++;
				}

				fprintf(new_results_file," -1:%g",model->bias);
				fprintf(new_results_file,"\n");

				if(save_testing!=NULL){
					fprintf(save_testing,"%d %g %g %g\n",support_size,(double)mistakes/(double)testdata->l,hinge_loss,l1norm);
				}
	}

	clock_t time_end = clock ();
	double physical_time = (double)(time_end - time_start)/CLOCKS_PER_SEC;

    std::basic_ifstream<char> exists_file = std::ifstream(params->summary_exp_file_name);

	FILE* summary_exp = fopen(params->summary_exp_file_name,"a");
    if(!exists_file)
        fprintf(summary_exp,"TUNNING DATASET;TIME B SYNTONIZATION;NUMBER OF CANDIDATES TESTED;\n");
	fprintf(summary_exp,"%s;%f;%d\n", tunning_file_name,physical_time,N_BES_SYNC);
	
	fclose(new_results_file);
	if(save_testing!=NULL)
		fclose(save_testing);
	fclose(summary_exp);

	free(testdata->y);
	free(testdata->x);
	free(testdata->x_space);
	free(testdata);
	delete(testQ);

}

void SASSO_train::test_regularization_path(sasso_model** models, sasso_problem* input_problem, sasso_parameters* params, char* testset_file_name){

	printf("READING TEST DATA ...\n");
	dataset* testdata = this->readClassificationDataSet(testset_file_name);
	printf("READING TEST DATA DONE.\n");

	TEST_Q* testQ = new TEST_Q(testdata,input_problem,params,params->cache_size);
	
	FILE* save_testing=NULL;
	if(params->save_file_testing_reg_path)
		save_testing=fopen(params->file_testing_reg_path,"w");

	int support_size;
	int mistakes;
	double hinge_loss;
	double l1norm;

	for(int i=0; i < (int)params->n_steps_reg_path; i++){
				printf("START -- Model Number = %d\n",i+1);
				sasso_model* model = models[i];
				data_node* weights = model->weights;
				mistakes = testQ->testSVM(weights,model->bias,mistakes,support_size,hinge_loss,l1norm);
				printf("Number of Support Vectors = %d\n",support_size);
				printf("Misclassification Rate = %f\n",(double)mistakes/(double)testdata->l);
				printf("Hinge Loss = %g\n",hinge_loss);
				printf("L1 norm = %g\n",l1norm);
				if(save_testing!=NULL){
					fprintf(save_testing,"%d %g %g %g\n",support_size,(double)mistakes/(double)testdata->l,hinge_loss,l1norm);
				}
	}

	if(save_testing!=NULL)
		fclose(save_testing);

	printf("END TESTING\n");	
}

void SASSO_train::test_all_regularization_path(sasso_model** models, sasso_problem* input_problem,
                                               sasso_parameters* params, std::string testset_file_name,
                                               int step_init, int step_size, std::map<int, double>& model_missclass,
											   std::map<int, int>& map_support_size, std::map<int, double>& map_hinge,
											   std::map<int, double>& map_l1norm){

	std::cout << "READING TEST DATA from " << testset_file_name << std::endl;
    dataset* testdata = this->readClassificationDataSet(testset_file_name.c_str());
    printf("READING TEST DATA DONE.\n");

    TEST_Q* testQ = new TEST_Q(testdata,input_problem,params,params->cache_size);

	int support_size;
	int mistakes;
	double hinge_loss;
	double l1norm;


    for(int i=step_init-1; i < (int)params->n_steps_reg_path; i+= step_size){
        printf("START -- Model Number = %d\n",i+1);
        sasso_model* model = models[i];
        data_node* weights = model->weights;
		printf("Model Number = %d loaded\n",i+1);
        mistakes = testQ->testSVM(weights,model->bias,mistakes,support_size,hinge_loss,l1norm);
        double misclass = (double)mistakes/(double)testdata->l;
		printf("Misclassification Rate = %f\n",misclass);

        model_missclass[i] += misclass;
		map_hinge[i] += hinge_loss;
		map_support_size[i] = support_size;
		map_l1norm[i] = l1norm;
    }

    free(testdata->y);
    free(testdata->x);
    free(testdata->x_space);
    free(testdata);
    delete(testQ);

    printf("END TESTING\n");
}

void SASSO_train::test_model(sasso_model* model, sasso_parameters* params, char* testset_file_name){

	printf("READING DATA ...\n");
	dataset* data = this->readClassificationDataSet(testset_file_name);
	printf("READING DATA DONE.\n");
	this->test_model(model, params, data);	

}


void SASSO_train::compute_delta_range(sasso_problem* problem, sasso_parameters* params){
	
	printf("----- DETERMINING DELTA RANGE ----- \n");


	double origina_l1_norm = this->l1norm_input_SVM(problem);
	
	params->n_steps_reg_path = N_STEPS_PATH;
	params->reg_param_max = origina_l1_norm;
	params->reg_param_min = params->reg_param_max/params->frac_min_delta;
	params->reg_param = params->reg_param_min;
	params->reg_param_step = params->reg_param_max/params->n_steps_reg_path;
	
	printf("----- END DETERMINING DELTA RANGE ----- \n");
	

}

sasso_model** SASSO_train::compute_regularization_path(sasso_problem* prob, sasso_parameters* params,sasso_stats* stats){

	params->quick_stop_regularization_path = false;
	
	//CHECK PARAMETERS
	if(params->n_steps_reg_path <= 0)
		params->n_steps_reg_path = N_STEPS_PATH;
	if((params->reg_param_max < 0.0) || (params->reg_param_min < 0.0))
		compute_delta_range(prob, params);
	if(params->reg_param_min < 0.0)
		params->reg_param_min = params->eps;

	params->reg_param = params->reg_param_min;

	FW_SASSO_solver = new SASSO(prob,params);

	data_node** weights_path = new data_node*[(int)params->n_steps_reg_path];

	clock_t time_start = clock ();
	
	// TRAINING ..
	FW_SASSO_solver->Solve(params->eps, params->training_algorithm, params->cooling, params->randomized, weights_path);
	// END TRAINING

	clock_t time_end = clock ();
	
	FW_SASSO_solver->getStats(stats);
	stats->physical_time = (double)(time_end - time_start)/CLOCKS_PER_SEC;


	sasso_model** models = new sasso_model*[(int)params->n_steps_reg_path];
	for(int i=0; i<(int)params->n_steps_reg_path; i++){
		sasso_model* model = new sasso_model();
		model->bias=prob->bias;
		model->params = params;
		model->input_problem = prob;
		model->weights = weights_path[i];
		models[i]=model;
	}

	return models;
}


sasso_model* SASSO_train::train(sasso_problem* prob, sasso_parameters* params, sasso_stats* stats){

	printf("Training the model ... \n");
	printf("Problem is of size %d and dimension %d ... \n",prob->l, prob->input_dim);

	//CHECK PARAMETERS
	if(params->reg_param < 0.0)
		throw std::invalid_argument( "ERROR. Regularization parameter cannot be negative\n" );

	sasso_model* model = new sasso_model();

	model->bias=prob->bias;
	printf("BIAS MODEL IS %f\n",model->bias);
	model->params = params;
	model->input_problem = prob;

	// if(params->normalize){
	// 	prob = normalize(prob, model);
	// 	prob->normalized = true;
	// } else
	// 	prob->normalized = false;

	model->weights = NULL;

	//TRAINING ...
	if(FW_SASSO_solver==NULL)
		FW_SASSO_solver = new SASSO(prob,params);

	clock_t time_start = clock();
		
	printf("Training ...\n");

	FW_SASSO_solver->Solve(params->eps, params->training_algorithm, params->cooling, params->randomized);

	clock_t time_end = clock();
		
	printf("Training DONE in %g (secs)\n",(double)(time_end - time_start)/CLOCKS_PER_SEC);

	FW_SASSO_solver->ComputeLASSOSolution(model->weights, 0.0);
	
	printf("Solution COMPUTED!!.\n");

	FW_SASSO_solver->getStats(stats);
	stats->physical_time = (double)(time_end - time_start)/CLOCKS_PER_SEC;

	// int i=0;
	// while(model->weights[i].index!=-1){

	// 	printf("WEIGHT: IDX=%d, VAL=%g\n",model->weights[i].index,model->weights[i].value);
	// 	i++;
	
	// }

	//END TRAINING ...

	return model;
}


//WARNING: it overrides the previous problem
sasso_problem* SASSO_train::normalize(sasso_problem* prob, sasso_model* model){
	
	model->mean_predictors = new double[prob->l];
	model->inv_std_predictors = new double[prob->l];
	model->normalized = true;

	prob->mean_predictors = model->mean_predictors;
	prob->inv_std_predictors = model->inv_std_predictors;
	prob->normalized = true;

	for(int i=0; i<prob->l; i++){
		
		data_node* x = prob->x[i];
		double sum_x = 0.0;
		double sum_x2 = 0.0;
		while(x->index != -1){
			sum_x +=  x->value;
			sum_x2 +=  (x->value)*(x->value);
			++x;
		}

		double av_x=sum_x/(double)prob->input_dim;
		double av_x2=sum_x2/(double)prob->input_dim;
		double std_x = std::sqrt((double)prob->input_dim*(av_x2 - (av_x*av_x)));	
		
		prob->mean_predictors[i] = av_x;
		if(std_x > 0.0){
			prob->inv_std_predictors[i] = (1.0/std_x);
		} else {
			prob->inv_std_predictors[i] = 1.0/std::sqrt((double)prob->input_dim);
		}
	}

 return prob;

}

sasso_problem* SASSO_train::readSASSOProblem(const char *filename){

	FILE *fp = fopen(filename,"r");
	sasso_problem* prob = Malloc(struct sasso_problem,1);
	
	prob->l = 0;

	int elements, i, j;
	int type, dim;
	int max_index;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	
	elements = 0;
	type     = 0; // sparse format
	dim      = 0;
    
    
    count_pattern_SASSO(fp, prob, elements, type, dim);
   
	prob->alpha_orig  = Malloc(double,(unsigned long)prob->l);
	prob->x  = Malloc(struct data_node *,(unsigned long)prob->l);
	prob->x_space  = Malloc(struct data_node,(unsigned long)(elements+prob->l));
	prob->elements = elements;
	prob->type = type;
	double THE_CONSTANT=10000;
	prob->const_scaling_weights=THE_CONSTANT;
	prob->bias = prob->bias/THE_CONSTANT;

	if (!prob->alpha_orig || !prob->x || !prob->x_space) {
		fprintf(stdout, "ERROR: not enough memory!\n");
		prob->l = 0;
		return NULL;
	}

	max_index = 0;
	j         = 0;

 	for(i=0; i<prob->l; i++)
	{	
		if(i%50000==0){
			printf("%i datos cargados...\n",i);
		}
		double label;
		prob->x[i] = &prob->x_space[j];

		// load the weight for dense or sparse
		fscanf(fp,"%lf",&label);
		prob->alpha_orig[i] = label/THE_CONSTANT;

		int elementsInRow = 0;
		while(1)
		{	
			int c;
			
			do {
				c = getc(fp);	
				if(c=='\n') break;
			} while(isspace(c));
			
			if((c=='\n') || (c==EOF)) break;
			
			ungetc(c,fp);

			if (type == 0) // sparse format
			{
				fscanf(fp,"%d:%lf",&(prob->x_space[j].index),&(prob->x_space[j].value));
				++j;
			}
			else if ((type == 1) && (elementsInRow < dim-1)) // dense format, read a feature
			{
				prob->x_space[j].index = elementsInRow+1;
				elementsInRow++;
				fscanf(fp, "%lf,", &(prob->x_space[j].value));
				++j;
			}
			else if ((type == 1) && (elementsInRow == dim-1)) // dense format, read a feature
			{
				prob->x_space[j].index = elementsInRow+1;
				elementsInRow++;
				fscanf(fp, "%lf", &(prob->x_space[j].value));
				++j;
			}
			else if ((type == 1) && (elementsInRow >= dim)) // dense format, read the label
			{
                fscanf(fp,"%lf",&label);
				prob->alpha_orig[i] = label/THE_CONSTANT;
			}
		}	

		if(j>=1 && prob->x_space[j-1].index > max_index)
			max_index = prob->x_space[j-1].index;
		prob->x_space[j++].index = -1;
		
	}

	printf("problem is of dimension: %d\n",max_index);
	printf("prob->l is: %d\n",prob->l);
	prob->input_dim = max_index;
	return prob;

}

dataset* SASSO_train::readClassificationDataSet(const char *filename){

	FILE *fp = fopen(filename,"r");
	dataset* data = Malloc(struct dataset,1);
	
	data->l = 0;

	int elements, i, j;
	int type, dim;
	int max_index;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	
	elements = 0;
	type     = 0; // sparse format
	dim      = 0;
    
    count_pattern(fp, data, elements, type, dim);
   	rewind(fp);
    
	data->y  = Malloc(double,(unsigned long)data->l);
	data->x  = Malloc(struct data_node *,(unsigned long)data->l);
	data->x_space  = Malloc(struct data_node,(unsigned long)(elements+data->l));
	data->elements = elements;
	data->type = type;

	if (!data->y || !data->x || !data->x_space) {
		fprintf(stdout, "ERROR: not enough memory!\n");
		data->l = 0;
		return NULL;
	}

	max_index = 0;
	j         = 0;

 	for(i=0; i<data->l; i++)
	{	
		if(i%50000==0){
			printf("%i datos cargados...\n",i);
		}
		double label;
		data->x[i] = &data->x_space[j];
		if (type == 0) // sparse format
		{
			fscanf(fp,"%lf",&label);
			data->y[i] = label;
		}

		int elementsInRow = 0;
		while(1)
		{	
			int c;
			
			do {
				c = getc(fp);	
				if(c=='\n') break;
			} while(isspace(c));
			
			if((c=='\n') || (c==EOF)) break;
			
			ungetc(c,fp);

			if (type == 0) // sparse format
			{

				fscanf(fp,"%d:%lf",&(data->x_space[j].index),&(data->x_space[j].value));
				++j;
			}
			else if ((type == 1) && (elementsInRow < dim)) // dense format, read a feature
			{
				data->x_space[j].index = elementsInRow+1;
				elementsInRow++;
				fscanf(fp, "%lf,", &(data->x_space[j].value));
				++j;
			}
			else if ((type == 1) && (elementsInRow >= dim)) // dense format, read the label
			{
                fscanf(fp,"%lf",&label);
				data->y[i] = label;
			}
		}	

		if(j>=1 && data->x_space[j-1].index > max_index)
			max_index = data->x_space[j-1].index;
		data->x_space[j++].index = -1;
		
	}

	data->input_dim = max_index;
	return data;

}

void SASSO_train::destroyProblem(sasso_problem* prob){

	free(prob->x_space);
	free(prob->alpha_orig);
	free(prob->x);

}
	
void SASSO_train::destroyModel(sasso_model* mod){
	
	// if(mod->mean_predictors!=NULL){
	// 	delete [] mod->mean_predictors;
	// } 

	// if(mod->std_predictors!=NULL){
	// 	delete [] mod->std_predictors;
	// } 

}

//detects data format and counts number of patterns and dimensions
void SASSO_train::count_pattern_SASSO(FILE *fp, sasso_problem* prob, int &elements, int &type, int &dim)
{
	char str1[10], str2[10];
	double bias;
 	fscanf(fp, "%s %s %lf\n", str1, str2, &bias);
    prob->bias=bias;

    int c;
    while ( (c=fgetc(fp)) != EOF )
    {
	    switch(c)
	    {
		    case '\n':
		    	prob->l += 1;
		    	if ((type == 1) && (dim == 0)) // dense format
					dim = ++elements;
		        else if (type == 1) 
		        	++elements;			    
			    break;

		    case ':':
			    ++elements;
			    break;

		    case ',':
			    ++elements;
			    type = 1;
			    break;

		    default:
			    ;
	    }
    }
    
    rewind(fp);
    fscanf(fp, "%s %s %lf\n", str1, str2, &bias);
    
    printf(">>> READING SASSO PROBLEM. INITIAL SUPPORT SET IS: l = %d, elements = %d\n",prob->l,elements);

}

void SASSO_train::showData(sasso_problem* prob){

 FILE * pFile;
 pFile = fopen ("testData.txt","w");

	for(int i=0; i<prob->l; i++){
		data_node* old_x = prob->x[i];
		printf("\ndato %d\n",i);
		while(old_x->index != -1){
			int feature_idx = old_x->index;
			double feature_value = old_x->value;	
			fprintf(pFile, "%d:%f, ",feature_idx,feature_value);
			printf("%d:%f, ",feature_idx,feature_value);
			old_x++;
		}
		fprintf(pFile,"\n");
	}

}

//detects data format and counts number of patterns and dimensions
void SASSO_train::count_pattern(FILE *fp, dataset* prob, int &elements, int &type, int &dim)
{

    int c;
    do
    {
	    c = fgetc(fp);
	    switch(c)
	    {
		    case '\n':
		    	prob->l += 1;
		    	if ((type == 1) && (dim == 0)) // dense format
			        dim = elements;				    
			    break;

		    case ':':
			    ++elements;
			    break;

		    case ',':
			    ++elements;
			    type = 1;
			    break;

			case EOF:			    
			    break;

		    default:
			    ;
	    }
    } while  (c != EOF);
    
    rewind(fp);
    printf(">>> LOADING DATASET, NUMBER OF EXAMPLES IS: l = %d, elements = %d\n",prob->l,elements);

}


void SASSO_train::parse_command_line(sasso_parameters* params, int argc, char **argv, char *input_file_name, char *model_file_name, char* results_file_name, char* path_file_name)
{
	int i;

	params->exp_type = EXP_SINGLE_TRAIN;
	params->cache_size   = 2000;	
	params->eps          = 1e-5;
	params->stopping_criterion = STOPPING_WITH_OBJECTIVE;
	params->sample_size  = 139;
	params->training_algorithm = FW;
	params->cooling = false;
	params->randomized = true;
	params->initialization_method = ZERO;
    params->save_model = false;
	params->max_iterations =  std::numeric_limits<int>::max();
	params->normalize = false;

	params->reg_param = -1.0;
	params->reg_param_min = -1.0;
	params->reg_param_max = -1.0;
	params->reg_param_step = -1.0;
	params->computing_regularization_path = false;
	params->quick_stop_regularization_path = false;
	params->print_regularization_path = false;
	params->print_optimization_path = false;

	params->safe_stopping = false;
	params->nsamplings_safe_stopping = 1;
	params->nsamplings_iterations = 1;
	params->n_steps_reg_path = N_STEPS_PATH;
	params->frac_min_delta = FRAC_MIN_DELTA;
	params->BORDER_WARM_START = false;
	params->ACTIVE_SET_HEURISTIC = false;
	params->kernel_type = -1;
	params->gamma = -1.0;
	params->degree = -1;
	params->coef0 = 0.0;
	params->test_data_file_name = NULL;
	params->file_validation_set = NULL;
    params->syntB = false;
	params->save_models_along_the_path = false;
	params->save_file_testing_reg_path = false;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();

		switch(argv[i-1][1])
		{
			case 'E':
				params->exp_type = atoi(argv[i]);
				if(params->exp_type == EXP_REGULARIZATION_PATH)
					params->computing_regularization_path = true;
					params->print_regularization_path = true;
				break;
			case 'I':
				if(argv[i-1][2]=='M')
					params->initialization_method =  ZERO;
				if(argv[i-1][2]=='L')
					params->max_iterations = atoi(argv[i]);
				break;
			case 'N':
				if(argv[i-1][2]=='F'){
					params->nfold = atoi(argv[i]);
				} else if (argv[i-1][2]=='S'){
					params->n_steps_reg_path = atof(argv[i]);
				} else if (argv[i-1][2]=='M'){
					if(atof(argv[i])>0.0)
						params->normalize = true;
					else
						params->normalize = false;
				} 
				break;	
			case 'M':
				if(argv[i-1][2]=='A'){
					params->training_algorithm = atoi(argv[i]);
				}
				break;
			case 'B':
				if(argv[i-1][2]=='W'){
					if(atoi(argv[i])>0){
						params->BORDER_WARM_START  = true;
					} else {
						params->BORDER_WARM_START  = false;
					}
				}
				break;
			case 'F':
				if(argv[i-1][2]=='D'){
					params->frac_min_delta = atof(argv[i]);
				} else if(argv[i-1][2]=='M'){
					params->frecuency_messages = atoi(argv[i]);
					printf(">> FM:%d\n",params->frecuency_messages);
				}
				break;
			case 'T':
				params->test_data_file_name = new char[8192];
				strcpy(params->test_data_file_name, argv[i]);
				break;
			case 'V':
				params->file_validation_set = new char[8192];
				strcpy(params->file_validation_set, argv[i]);
				break;
			case 'C':
				if (argv[i-1][2]=='O'){
					if(atoi(argv[i]) == 0)
						params->cooling = false;
					else 	
						params->cooling = true;
				}
				break;
			case 'm':
				params->cache_size = atof(argv[i]);
				break;
			case 'e':
				params->eps = atof(argv[i]);
				break;
			case 'R':
				if (argv[i-1][2]=='S'){
					if(atoi(argv[i]) == 0)
						params->randomized = false;
					else 	
						params->randomized = true;
				} else if(argv[i-1][2]=='P'){
						params->reg_param = atof(argv[i]);
				} else if(argv[i-1][2]=='M'){
						params->reg_param_min = atof(argv[i]);
				} else if(argv[i-1][2]=='X'){
						params->reg_param_max = atof(argv[i]);
				} else if(argv[i-1][2]=='D'){
						params->reg_param_step = atof(argv[i]);
				}
				break;	
			case 'k':
				params->kernel_type = atoi(argv[i]);
				break;
			//LINEAR, POLY, RBF, SIGMOID, EXP, NORMAL_POLY, INV_DIST, INV_SQDIST
			case 'g':
				params->gamma = atof(argv[i]);
				break;	
			case 'd':
				params->degree = atoi(argv[i]);
				break;	
			case 'c':
				params->coef0 = atof(argv[i]);
				break;		
			case 'X':
				params->save_file_testing_reg_path = true;
				params->file_testing_reg_path = argv[i];
				break;				
			case 'S':
				if (argv[i-1][2]=='C'){
					if(atoi(argv[i]) == 1){
						params->stopping_criterion = STOPPING_WITH_INF_NORM;
					} else { 
						if (atoi(argv[i]) == 2)	
							params->stopping_criterion = STOPPING_WITH_DUAL_GAP;
						else
							params->stopping_criterion = STOPPING_WITH_OBJECTIVE;
					}	 
				} else if(argv[i-1][2]=='S'){
					//safe stopping
					params->safe_stopping = true;
					params->nsamplings_safe_stopping = atoi(argv[i]);
				} else if(argv[i-1][2]=='T'){
					if(atoi(argv[i]) == 0){
						params->randomization_strategy = UNIFORM;
					} else { 
						params->randomization_strategy = BLOCKS;
					}
				} else if(argv[i-1][2]=='B'){
						params->file_new_reg_path = new char[8192];
						strcpy(params->file_new_reg_path, argv[i]);
				}  
				break;				 
			case 'a':
				params->sample_size = atoi(argv[i]);
				break;
            case 'b':
                if(atoi(argv[i]) == 0)
                    params->syntB = false;
                else
                    params->syntB = true;
                break;
			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
		}
	}



	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(params->kernel_type < 0){
		throw std::invalid_argument( "ERROR. You need to specify a kernel\n" );
	}

	if(params->kernel_type == EXP){
		if(params->gamma < 0.0){
			throw std::invalid_argument( "ERROR. For this kernel you need to specify gamma\n" );
		}
	}

	if(params->kernel_type == POLY){
		if(params->degree < 0){
			throw std::invalid_argument( "ERROR. For this kernel you need to specify degree\n" );
		}
	}

	if(params->exp_type != TEST_REG_PATH && params->exp_type != TEST_ALL_REG_PATH  && params->exp_type != SYNC_B){

		if(i<argc-1) 
		{
			strcpy(model_file_name,argv[i+1]);
			params->save_model = true;
		}

		char stamp[64];
		sprintf(stamp,"%s",give_me_the_time());

		sprintf(results_file_name,"%s.SASSO.RESULTS.%s.%.1E.%s.%s.txt",input_file_name,getTextTrainingAlgorithm(params->training_algorithm),params->eps,getModalityAlgorithm(params),stamp);
		sprintf(path_file_name,"%s.SASSO.ALG-PATH.%s.%.1E.%s.%s.txt",input_file_name,getTextTrainingAlgorithm(params->training_algorithm),params->eps,getModalityAlgorithm(params),stamp);
		params->summary_exp_file_name = new char[8192];
		sprintf(params->summary_exp_file_name,"%s.SASSO.SUMMARY.%s.%.1E.%s.%s.txt",input_file_name,getTextTrainingAlgorithm(params->training_algorithm),params->eps,getModalityAlgorithm(params),stamp);
	
	} else if(params->exp_type == SYNC_B || params->exp_type == TEST_ALL_REG_PATH){

		char stamp[64];
		sprintf(stamp,"%s",give_me_the_time());

		if(i<argc-1) 
		{
			strcpy(path_file_name,argv[i+1]);
			
		} 
		
		params->summary_exp_file_name = new char[8192];
		sprintf(params->summary_exp_file_name,"%s.SUMMARY-B.SYNTONIZATION-%s.txt",path_file_name,stamp);

	} else {

		if(i<argc-1) 
		{
			strcpy(path_file_name,argv[i+1]);
			
		} else
			exit_with_help();
	}

	params->input_file_name = input_file_name;
	params->model_file_name = model_file_name;
	params->path_file_name = path_file_name;
	params->results_file_name = results_file_name;
	

}

void SASSO_train::printParams(FILE* file, sasso_parameters* params){
 
	if(params->stopping_criterion == STOPPING_WITH_DUAL_GAP)
		fprintf(file,"Stopping Criterion: STOPPING_WITH_DUAL_GAP\n");
	else if (params->stopping_criterion == STOPPING_WITH_INF_NORM)
		fprintf(file,"Stopping Criterion: STOPPING_WITH_INF_NORM\n");
	else if (params->stopping_criterion == STOPPING_WITH_OBJECTIVE)
		fprintf(file,"Stopping Criterion: STOPPING_WITH_OBJECTIVE\n");
	else
		fprintf(file,"Stopping Criterion: UNKNOWN\n");

	if(params->BORDER_WARM_START)
		fprintf(file,"Warm Start: BORDER (SCALING PREVIOUS SOLUTION)\n");
	else
		fprintf(file,"Warm Start: PREVIOUS SOLUTION (i.e. NOTHING SPECIAL)\n");

	if(params->ACTIVE_SET_HEURISTIC)
		fprintf(file,"Active Set Exploration in Toward Search: YES\n");
	else
		fprintf(file,"Active Set Exploration in Toward Search: NO\n");

	fprintf(file,"EPS Stopping: %g\n",params->eps);
	fprintf(file,"Cache Size (MB): %f\n",params->cache_size);

	fprintf(file,"Training Method: %s\n",getTextTrainingAlgorithm(params->training_algorithm));
	
	if(params->exp_type == EXP_SINGLE_TRAIN){
		fprintf(file,"Experiment Type: Single Trainining ...\n");
		fprintf(file,"Regularization Parameter (delta): %g\n",params->reg_param);
	} else if (params->exp_type == EXP_REGULARIZATION_PATH){
		fprintf(file,"Experiment Type: Regularization Path ...\n");
		fprintf(file,"MIN - Regularization Parameter (delta_min): %g\n",params->reg_param_min);
		fprintf(file,"MAX - Regularization Parameter (delta_max): %g\n",params->reg_param_max);
		fprintf(file,"STEP - Regularization Parameter (delta_step): %g\n",params->reg_param_step);
		fprintf(file,"NUMBER STEPS IN PATH: %g\n",params->n_steps_reg_path);
		fprintf(file,"FRACTION TO CALCULATE DELTA MIN: %g\n",params->frac_min_delta);		
	}

	if(params->cooling)
		fprintf(file,"Cooling: YES\n");
	else
		fprintf(file,"Cooling: NO\n");

	if(params->randomized){
		
			fprintf(file, "Randomization: YES (%d points)\n", params->sample_size);
			
			if(params->randomization_strategy == UNIFORM){
				fprintf(file, "Randomization Strategy: SIMPLE RANDOM SAMPLE\n");
			} else if (params->randomization_strategy == BLOCKS) {
				fprintf(file, "Randomization Strategy: BLOCKS\n");
			}

			if(params->safe_stopping)
				fprintf(file, "Safe Stopping: YES (-SS %d)\n", params->nsamplings_safe_stopping);
			else
				fprintf(file, "Safe Stopping: NO\n");
	} else {
			fprintf(file, "Randomization: NO\n");
	}


	if(params->normalize)
		fprintf(file,"Normalize: YES\n");
	else
		fprintf(file,"Normalize: NO\n");

	if(params->max_iterations < std::numeric_limits<int>::max())
		fprintf(file,"Max Iterations: %d\n",params->max_iterations);

}

void SASSO_train::printStats(FILE* file, sasso_stats* stats){
 
 	fprintf(file,"** Performance ** \n");
	fprintf(file,"Iterations: %g\n",stats->n_iterations);
	fprintf(file,"Performed Dot Products: %g\n",stats->n_performed_dot_products);
	fprintf(file,"Requested Dot Products: %g\n",stats->n_requested_dot_products);
	fprintf(file,"Running Time (Secs): %g\n",stats->physical_time);


}

const char* SASSO_train::getModalityAlgorithm(sasso_parameters* params){
	if(params->randomized)
		return "RANDOMIZED";
	else
		return "DETERMINISTIC";
}

const char* SASSO_train::getTextTrainingAlgorithm(int code){
	switch(code)
		{
			case FW:
				return "FW";
				break;
			case FULLYFW:
				return "FULLY CORRECTIVE FW";
				break;
		}
	return "OTHER (CHECK)";
}

char* SASSO_train::give_me_the_time(){

  time_t rawtime = time(0);
  tm *now = localtime(&rawtime);
  char timestamp[32];
  if(rawtime != -1){
     strftime(timestamp,sizeof(timestamp),"%Y-%m-%d-%Hhrs-%Mmins-%Ssecs",now);
  }
  return(timestamp);
}

void SASSO_train::exit_with_help()
{
	printf(
	"SASSO: Sparsification of SVMs using the LASSO\n"
	"Usage: SASSO-train [options] input_model_file output_file\n"
	"Version 04.08.15\n"
	"Options:\n"
	"-E : experiment-type (default 0)\n"
	"     0 -- SASSO with known regularization parameter\n"
	"     1 -- SASSO regularization path\n"
	"     2 -- test single model\n"
	"     3 -- test SASSO regularization path\n"
	"     4 -- Syntonization of b\n"
	"     5 -- mean test SASSO regularization path of 10 validation files\n"
	"-k : kernel type (default 2)\n"
	"     0: LINEAR, 1: POLY, 2: RBF, 3: SIGMOID\n"
	"-g : gamma parameter for RBF kernels\n" 
	"-c : coef0 parameter for POLY and SIGMOID kernels\n" 
	"-d : degree parameter for POLY and SIGMOID kernels\n" 
	"-RS : RANDOMIZED?: (0) no (1) yes - default\n" 
	"-RP value: Regularization Parameter\n" 
	"-RM value: Min value - Regularization Parameter (for EXP_TYPE = 1)\n" 
	"-RX value: Max value - Regularization Parameter (for EXP_TYPE = 1)\n" 
	"-RD value: Step value for chaning the Regularization Parameter (for EXP_TYPE = 1)\n" 
	"-IL : Maximum number of Iterations\n" 
	"-SC : Stopping Criterion: (0) INFINITE NORM OF DIFFERENCE BETWEEN ITERATES (default), (1) DUALITY GAP, (2) IMPROVEMENT IN THE OBJECTIVE\n" 
	"-e value: epsilon tolerance of termination criterion\n"
	"-NS n : number of differents trainings (requires differentes files train/test) is set to n\n"
	"-SM: save the model(s) after training\n"
	"-m cachesize: set cache memory size in MB (default 200)\n"
	"-a size: sample size for probabilistic sampling (default 60)\n"
    "-b : synchronize B\n"
	);
	exit(1);
}

int main(int argc, char **argv){

	SASSO_train* fw = new SASSO_train();
	sasso_problem* problem = new sasso_problem();
	sasso_parameters* params = new sasso_parameters();
	char input_file_name[4096]; 
	char model_file_name[4096]; 
	char results_file_name[8192]; 
	char path_file_name[8192]; 
	sasso_stats* stats = new sasso_stats();

	printf("PARSING COMMAND LINE ...\n");
	int imax = std::numeric_limits<int>::max();
	printf("MAX INT IS =%d\n", imax);
	fw->parse_command_line(params,argc,argv,input_file_name,model_file_name,results_file_name,path_file_name);
	printf("END PARSING ...\n");
	printf("1<<20 %d ...\n",1<<20);

	problem = fw->readSASSOProblem(input_file_name);


	if(params->exp_type == EXP_SINGLE_TRAIN){

		FILE* summary_exp = fopen(params->summary_exp_file_name,"w");
		fw->printParams(stdout,params);
		sasso_model* trained_model = fw->train(problem,params,stats);
		fw->printParams(summary_exp,params);
		fw->printStats(summary_exp,stats);
		if(params->test_data_file_name!=NULL){
			fw->test_simplified_SVM(trained_model,params,params->test_data_file_name);
		}
 		fw->destroyProblem(problem);
 		fclose(summary_exp);
	}
	
	if(params->exp_type == EXP_REGULARIZATION_PATH){
		fw->printParams(stdout,params);
		params->save_models_along_the_path = true;
		sasso_model** models = fw->compute_regularization_path(problem,params,stats);
		if(params->test_data_file_name!=NULL){
			fw->test_regularization_path(models,problem,params,params->test_data_file_name);
		}

		//fw->printParams(summary_exp,params);
		//fw->printStats(summary_exp,stats);
	}	
	
	if(params->exp_type == TEST_INPUT_MODEL){
		fw->test_input_SVM(problem,params,params->test_data_file_name);
	}

	if(params->exp_type == TEST_REG_PATH){
		printf("Reading SASSO Problem from = %s\n",input_file_name);
		sasso_problem* problem = fw->readSASSOProblem(input_file_name);
		sasso_model** models_path = fw->load_models_from_regularization_path(problem,path_file_name, params, true);
		if(params->test_data_file_name!=NULL){
			fw->test_regularization_path(models_path,problem,params,params->test_data_file_name);
		}
	}

	if(params->exp_type == SYNC_B){

		printf("Reading SASSO Problem from = %s\n",input_file_name);
		sasso_problem* problem = fw->readSASSOProblem(input_file_name);
		sasso_model** models_path = fw->load_models_from_regularization_path(problem,path_file_name, params, false);
		if((params->file_validation_set==NULL) && (params->file_testing_reg_path==NULL)){
			printf("ERROR: Validation OR Test Data is Required ...\n");
		} else {
			printf("SYNTONIZING B .... \n");
			fw->syntonizeB(models_path,problem,params,path_file_name);
		}

		printf("END SYNC B.\n");
	}

	if(params->exp_type == TEST_ALL_REG_PATH){
		printf("Readed SASSO Problem from = %s for test ALL\n",input_file_name);
        //Load the path sasso model
		sasso_model** models = fw->load_models_from_regularization_path(problem,path_file_name, params, false);
        std::map<int, double> model_missclass;
		std::map<int, int> support_size;
		std::map<int, double> hinge_loss;
		std::map<int, double> l1norm;
        std::string file_validation_set_wildcard;
        int step_init = 1;
        int step_size = 1;
        for(int i=step_init-1; i < (int)params->n_steps_reg_path; i+= step_size){
            model_missclass[i] = 0;
			hinge_loss[i] = 0;
        }

		if (params->file_validation_set == NULL) {
			printf("ERROR: Validation is Required ...\n");
		} else {
            file_validation_set_wildcard = std::string(params->file_validation_set);
		}


        for(int i = 1; i<=10; i++) {
			if (params->syntB) {
				//reload model
				if(i>1){
					delete(models);
					models = fw->load_models_from_regularization_path(problem,path_file_name, params, false);
				}

                // set validation_file
                std::string file_validation_set_str = file_validation_set_wildcard + std::to_string(i);
                params->file_validation_set = new char [file_validation_set_str.length()+1];
                strcpy (params->file_validation_set, file_validation_set_str.c_str());

                printf("SYNTONIZING B .... \n");
                fw->syntonizeB(models, problem, params, path_file_name);
			}

            //test data test_data_file_name
            if (params->test_data_file_name != NULL) {
                // models_path = path of sasso.
                // problem train/target from input_file_name.
                // params:
                // test_data_file_name = test file.
                fw->test_all_regularization_path(models, problem, params,
                                                 std::string(params->test_data_file_name) + std::to_string(i),
                                                 step_init, step_size, model_missclass,
												 support_size, hinge_loss, l1norm);
            }
        }

		FILE* save_testing=NULL;
		if(params->save_file_testing_reg_path)
			save_testing=fopen(params->file_testing_reg_path,"w");

        std::cout << "Printing miss class error: " << std::endl;
        for(int i=step_init-1; i < (int)params->n_steps_reg_path; i+= step_size){
			if(save_testing!=NULL){
				fprintf(save_testing,"%d %g %g %g\n",support_size[i],model_missclass[i]/10,hinge_loss[i]/10,l1norm[i]);
			}
			printf("%d %g %g %g\n",support_size[i],model_missclass[i]/10,hinge_loss[i]/10,l1norm[i]);
        }
		if(save_testing!=NULL)
			fclose(save_testing);
		delete(models);

	}

	//test_input_SVM(sasso_problem* prob, sasso_parameters* params, char* testset_file_name){

	delete(fw);
	delete(problem);

	return 0;

}


