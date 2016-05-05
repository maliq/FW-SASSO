#include "SASSO.h"
#include <stdio.h>
#include <ctype.h>
#include <stdexcept>
#include <string.h>
#include <math.h>
#include <cmath> 
#include <stdlib.h>
#include <time.h>
#include <stdexcept>


#define NUM_SAMPLINGS 1
#define EPS_SCALING 0.5
#define INITIAL_EPS 0.01
#define DUAL_GAP_TOL -10*TAU
#define THRESHOLD_WEIGHTS TAU

enum { INIT_STEP, FW_STEP, MFW_STEP, SWAP_STEP, PARTAN_STEP};	/* step_type */

int compare_data_node(const void *a,const void *b) {
data_node *x = (data_node *) a;
data_node *y = (data_node *) b;
   if (x->index < y->index)
      return -1;
   else if (x->index > y->index)
      return 1;
   else
      return 0;
}

int SASSO::ChooseRandomIndex(){//choose a random vertex from the m-dimensional simplex
	int idx = 0;
	int rand32bit = std::rand();
	idx	= rand32bit%prob->l; //** LASSO: NOT LONGER BALANCED SAMPLING **//
	return idx;
}


double SASSO::ComputeGradientCoordinate(int idx, Qfloat** Qcolumn){


	double grad = 0.0;
	
	if(!param->randomized){

		grad = gradientALLPoints[idx];
	
	} else {

		Qfloat* Qm = sassoQ->get_Q(idx,coreNum,coreIdx);

		for (int j=0; j<coreNum; j++){
				grad += (Qfloat)Qm[j]*outAlpha[j]; 
		}

		grad -= sassoQ->getLT(idx);
	}

	return grad;

}

bool SASSO::AllocateMemoryForFW(int initial_size){ 

		allocated_size_for_alpha= initial_size;
		double* temp_array_weights =  Malloc(double,initial_size);
		for(int m=0; m<coreNum; m++)
			temp_array_weights[m] = outAlpha[m];
		
		outAlpha = Malloc(double,initial_size);
		
		for(int m=0; m<coreNum; m++){
			outAlpha[m] = temp_array_weights[m];
		}

		if(param->randomized)
			gradientActivePoints = Malloc(Qfloat,initial_size);
		
		if(!param->randomized)
			gradientALLPoints = Malloc(Qfloat,prob->l);
		
		if(param->randomized){//initialize array for caching gradient of ACTIVE points
			for(int m=0; m<coreNum; m++){
				gradientActivePoints[m] = 0.0;
				Qfloat* Qm = sassoQ->get_Q(coreIdx[m],coreNum, coreIdx);
				if(Qm!=NULL) //** LASSO: IF CORENUM=0, NEXT 2 LINES SHOULD NOT BE COMPUTED **//
					for (int j=0; j<coreNum; j++)
						gradientActivePoints[m] += (Qfloat)Qm[j]*outAlpha[j]; 
				gradientActivePoints[m] -= sassoQ->getLT(coreIdx[m]); //** LASSO: WE NEED TO SUBSTRACT SIGMA_i **//
			}
		} else {

			for(int m=0; m<prob->l; m++){//initialize array for caching gradient of ALL points
				gradientALLPoints[m] = 0.0;
				Qfloat* Qm = sassoQ->get_Q(m,coreNum,coreIdx);
				if(Qm!=NULL)//** LASSO: IF CORENUM=0, NEXT 2 LINES SHOULD NOT BE COMPUTED **//
					for (int j=0; j<coreNum; j++)
						gradientALLPoints[m] += (Qfloat)Qm[j]*outAlpha[j]; 
				gradientALLPoints[m] -= sassoQ->getLT(m);	//** LASSO: WE NEED TO ADD SIGMA_i **//
			}
		}

		for(int m=coreNum; m<initial_size; m++){
			outAlpha[m] = 0.0;
		}

		free(temp_array_weights);
		return true;

}


bool SASSO::CheckMemoryForFW(){

	if (this->coreNum >= allocated_size_for_alpha) {	
		allocated_size_for_alpha = (int)(1.5*allocated_size_for_alpha);
		outAlpha = (double*)realloc(outAlpha,allocated_size_for_alpha*sizeof(double));
		if(param->randomized)
			gradientActivePoints = (Qfloat*)realloc(gradientActivePoints,allocated_size_for_alpha*sizeof(Qfloat));
		for(int k=this->coreNum;k<allocated_size_for_alpha; k++){
			outAlpha[k] = 0.0;
			if(param->randomized)
				gradientActivePoints[k] = 0.0;
		}
	}

	return true;
}

bool SASSO::Clean(){

	printf("Cleaning SASSO Solver ...\n");
	if(param->randomized){
			free(gradientActivePoints); 	
	} else{
			free(gradientALLPoints);
	}
	delete sassoQ;
	printf("Cleaning done.\n");
	return true;
}

//Problem is: min f(alpha) = 0.5(alpha^T Q alpha) s.t. 1^Talpha = 1, alpha >= 0
int SASSO::Solve(double FW_eps, int method, bool cooling, bool randomized){
	
	 int status = -1;

	 this->Initialize();

	 switch(method)
	 {
	 	case FW:
			status = this->StandardFW(FW_eps, cooling, randomized);
	 		break;
	 	case FULLYFW:
			status = this->FullyCorrectiveFW(FW_eps, cooling, randomized);
	 		break;	 		
	 	default:
	 		throw std::invalid_argument( "FW-based-LASSO: selected algorithm is not implemented ..." );
	 		break;
	 }

	 this->Clean();
	 return status;
}

int SASSO::Solve(double FW_eps, int method, bool cooling, bool randomized, data_node** models_found_in_the_path_){
	
	 models_found_in_the_path = models_found_in_the_path_;
	 return Solve(FW_eps, method, cooling, randomized);

}

//Searches the coordinate of the gradient with largest absolute value
//Returns the value of the gradient coordinate with SIGN!
double SASSO::TowardVertex(int &towardIdx, double Sk, double Fk, double delta){

    int randomCoordinate = -1;
    double randomCoordinateGrad = INFINITY;
    double currentTowardGradient = 0.0;//** LASSO: CHECK IF THIS INITIALIZATION IS OK **//
    Qfloat *Qcolumn = NULL;

	if(param->randomized){//randomized search of toward vertex
		previousQcolumn = Q_actives_dot_toward;
		if(param->randomization_strategy == BLOCKS){
			int block = sampler->setRandomBlock();
			for(int count_repetitions=0; count_repetitions< NUM_SAMPLINGS; count_repetitions++){
				for(int k=sampler->getStartCurrentBlock(); k<sampler->getEndCurrentBlock(); k++){
				 	randomCoordinate = k;
				 	//printf("RANDOM COORD=%d\n",k);
				 	randomCoordinateGrad = ComputeGradientCoordinate(randomCoordinate, &Qcolumn);
				 	if(std::abs(randomCoordinateGrad) >= std::abs(currentTowardGradient)){//** LASSO: WE NEED TO COMPUTE ABSOLUTE VALUE OF GRAD COORDINATES **//
				 		towardIdx = randomCoordinate;
				 		currentTowardGradient = randomCoordinateGrad;	
				 		Q_actives_dot_toward = Qcolumn; 
				 	}
				}
			}
		} else {
			clock_t time_toward_random_part = clock ();
			int count_sampling = 0;
			for(int count_repetitions=0; count_repetitions< NUM_SAMPLINGS; count_repetitions++){
				while(count_sampling < param->sample_size){
				 	randomCoordinate = ChooseRandomIndex();
				 	randomCoordinateGrad = ComputeGradientCoordinate(randomCoordinate, &Qcolumn);
				 	if(std::abs(randomCoordinateGrad) >= std::abs(currentTowardGradient)){//** LASSO: WE NEED TO COMPUTE ABSOLUTE VALUE OF GRAD COORDINATES **//
				 		towardIdx = randomCoordinate;
				 		currentTowardGradient = randomCoordinateGrad;	
				 		Q_actives_dot_toward = Qcolumn; 
				 	}
				 	count_sampling++;
				}
			}

			time_toward_random_part = clock () - time_toward_random_part;
			time_towards_1 += ((float)time_toward_random_part)/CLOCKS_PER_SEC;

			clock_t time_toward_active_points = clock ();
			
			double tilde_delta = (currentTowardGradient < 0.0) ? delta:-1.0*delta;
			double dual_gap = (Sk-Fk) - tilde_delta*currentTowardGradient;
	
			if((dual_gap<DUAL_GAP_TOL) && param->ACTIVE_SET_HEURISTIC){
				for(int k=0; k<coreNum; k++){
					randomCoordinate = coreIdx[k];
					randomCoordinateGrad = ComputeGradientCoordinate(randomCoordinate, &Qcolumn);
					if(std::abs(randomCoordinateGrad) >= std::abs(currentTowardGradient)){//** LASSO: WE NEED TO COMPUTE ABSOLUTE VALUE OF GRAD COORDINATES **//
				 		towardIdx = randomCoordinate;
				 		currentTowardGradient = randomCoordinateGrad;	
				 		Q_actives_dot_toward = Qcolumn; 
				 	}
				}
			}
			time_toward_active_points = clock () - time_toward_active_points;
			time_towards_2 += ((float)time_toward_active_points)/CLOCKS_PER_SEC;

		}
							
	} else {//not randomized search of toward vertex
			for (int m=0; m<prob->l; m++){
				 	double tempGradient = ComputeGradientCoordinate(m,&Qcolumn);
				 	//printf("G%d:%g\n",m,tempGradient);
					if(std::abs(tempGradient) > std::abs(currentTowardGradient)){//** LASSO: WE NEED TO COMPUTE ABSOLUTE VALUE OF GRAD COORDINATES **//
				 		currentTowardGradient = tempGradient;
				 		towardIdx = m;
				 	}	
			}
			previousQcolumn = NULL;
			Q_actives_dot_toward = NULL;
	}

	return currentTowardGradient;//** RETURNS THE SIGNED GRADIENT COORDINATE **//
}

double SASSO::ActiveTowardVertex(int &towardIdx, double Sk, double Fk, double delta){

    Qfloat *Qcolumn = NULL;

    towardIdx = coreIdx[0];
    double currentTowardGradient = ComputeGradientCoordinate(towardIdx,&Qcolumn);
    
	for(int m=1; m<coreNum; m++){
		int idx = coreIdx[m];
		double tempGradient = ComputeGradientCoordinate(idx,&Qcolumn);
		if(std::abs(tempGradient) > std::abs(currentTowardGradient)){//** LASSO: WE NEED TO COMPUTE ABSOLUTE VALUE OF GRAD COORDINATES **//
		 		currentTowardGradient = tempGradient;
		 		towardIdx = idx;
		}	
	}

	previousQcolumn = NULL;
	Q_actives_dot_toward = NULL;
	
	return currentTowardGradient;//** RETURNS THE SIGNED GRADIENT COORDINATE **//
}


double SASSO::safe_stopping_check(double Sk, double Fk, double delta, double &tilde_delta, double &dualGap, int &towardIdx, double &towardGrad){
	
	printf("** SAFE STOPPING CHECK FOR RANDOMIZED ITERATIONS ...\n");
	printf("** .. CURRENT DUAL GAP: %g (toward-gradient=%g, tilde-delta=%g)...\n",dualGap, towardGrad, tilde_delta);
	nsamplings_randomized_iterations = param->nsamplings_safe_stopping;
	int toward_vertex_check = toward_vertex;
	double toward_gradient_check = TowardVertex(toward_vertex_check,Sk,Fk,delta);
	if(std::abs(toward_gradient_check) > std::abs(toward_gradient)){
		towardGrad = toward_gradient_check;
		towardIdx = toward_vertex_check;
	}
	tilde_delta = (towardGrad< 0.0) ? delta:-1.0*delta;
	dualGap = (Sk-Fk) - tilde_delta*towardGrad;
	nsamplings_randomized_iterations = param->nsamplings_iterations;
	printf("** .. FINAL DUAL GAP: %g (toward-gradient=%g, tilde-delta=%g) ...\n",dualGap, towardGrad, tilde_delta);
	return dualGap;	
}

int SASSO::StandardFW(double convergence_eps, bool cooling, bool randomized){

	printf("FW Solver: Standard FW (with line search), ");
	cooling ? printf("cooling : YES, ") : printf("cooling : NO, ");
	randomized ? printf("randomized : YES\n") : printf("randomized : NO\n");


    FILE* pathFile;
    if(param->print_optimization_path)
	    pathFile = fopen(param->path_file_name,"w");   
 	
 	FILE* regPathFile = NULL;
    		
 	AllocateMemoryForFW(1000);
	
	greedy_it = 0;

	double Sk = 0.0;
	double Fk = 0.0;
	double Gk_istar = 0.0;
	double tilde_delta = 0;
	double sigma_istar;
	double gap = INFINITY;
	double delta_objective = INFINITY;
	double dual_gap = INFINITY;
	double max_diff_iterates = INFINITY;
	double objective = this->objective;
	double l1norm_solution = 0.0;
	double mdata = (double)prob->input_dim;

	//objective = (1.0/mdata)*objective;
	int fremessages = param->frecuency_messages;
	if(fremessages <= 0)
		fremessages = 500;
	printf("FM:%d\n",fremessages);

	double delta, delta_min, delta_max, delta_step;

	if(param->computing_regularization_path){
		delta_min = param->reg_param_min;
		delta_max = param->reg_param_max;
		delta_step = param->reg_param_step;
		printf(">>>> COMPUTING REGULARIZATION PATH \n");
	} else {
		delta_min = param->reg_param;
		delta_max = param->reg_param;
		delta_step = param->reg_param;

	}

    double nsteps = param->n_steps_reg_path; 

	if(delta_min<TAU)
		delta_min = delta_max/nsteps;

	double logdmax = std::log(delta_max);
    double logdmin = std::log(delta_min);
    double logrange = logdmax - logdmin;

    delta_step = logrange/((double)nsteps-1.0);
	printf(">>>> DELTA MIN=%g, DELTA_MAX=%g, DELTA_STEP=%g, logrange=%g,nsteps=%g\n",delta_min,delta_max,delta_step,logrange,param->n_steps_reg_path);
	if(param->randomized)
		printf(">> USING RANDOMIZED VERSION!!\n");
	else
		printf(">> USING DETERMINISTIC VERSION!!\n");

	data_node* previousSolution = NULL;//sparse format
	data_node* newSolution = NULL;//sparse format
	
	if(std::abs(delta_max-delta_min)<TAU)
		nsteps = 1;

	for(int delta_counter = 0; delta_counter < nsteps; delta_counter++){

		delta = std::exp(logdmin+((double)delta_counter*delta_step));
	   
	   	//*** SCALING HEURISTIC ***/
	   	if((param->BORDER_WARM_START) && (delta>0.0) && (l1norm_solution>0.0)){
	   		printf("################# STARTING WITH BORDER WARM START \n");
	   		double scale_factor = delta/l1norm_solution;
	   		double new_Sk = scale_factor*scale_factor*Sk;
			double new_Fk = scale_factor*Fk;
			double new_objective = 0.5*new_Sk - new_Fk;
			if(new_objective<objective){
				l1norm_solution = 0.0;
			   	for(int i=0; i < coreNum; i++){
					outAlpha[i] = scale_factor*outAlpha[i];
					l1norm_solution += std::abs(outAlpha[i]);
				}
				objective = new_objective;
				Sk = new_Sk;
				Fk = new_Fk;	
			}
		} else{
			printf("################# STARTING WITHOUT BORDER WARM START \n");
		}

		printf("***** OBJECTIVE FOR NEW DELTA=%g\n",objective);
		printf("\n *************** REGULARIZATION WITH : %g ******************* \n", delta);
		
		double epsilonFactor = EPS_SCALING;
		double currentEpsilon = cooling ? INITIAL_EPS : convergence_eps/epsilonFactor;
	
		//iterate until desired precision 
		while((delta>TAU) && (currentEpsilon > convergence_eps)){// solve problem with current epsilon (warm start from the previous solution)
			currentEpsilon *= epsilonFactor;		
			currentEpsilon = (currentEpsilon < convergence_eps) ? convergence_eps: currentEpsilon;
			printf("EPS-Iteration: Iterating to achieve EPS = %g Current.EPS = %g N.Active.Points = %d\n",convergence_eps,currentEpsilon,coreNum);
			
			toward_vertex = -1;
			
			dual_gap = -INFINITY;

			sampler->reset();
			while(dual_gap<DUAL_GAP_TOL){
				toward_gradient = TowardVertex(toward_vertex,Sk,Fk,delta);
				tilde_delta = (toward_gradient < 0.0) ? delta:-1.0*delta;
				dual_gap = (Sk-Fk) - tilde_delta*toward_gradient;
			}

			//dual_gap = (1.0/mdata)*dual_gap;

			if((dual_gap <= currentEpsilon) && (param->randomized) && (param->safe_stopping) && (param->stopping_criterion == STOPPING_WITH_DUAL_GAP))
				safe_stopping_check(Sk,Fk,delta,tilde_delta,dual_gap,toward_vertex,toward_gradient);
	
			Gk_istar = toward_gradient + sassoQ->getLT(toward_vertex);
			max_diff_iterates = INFINITY;
			delta_objective = INFINITY;

			if(param->stopping_criterion == STOPPING_WITH_INF_NORM){
				gap = max_diff_iterates;
			} else {
				if(param->stopping_criterion == STOPPING_WITH_OBJECTIVE)
					gap = delta_objective;
				else
					gap = dual_gap;
			}

			printf("** INITIAL OBJECTIVE = %g\n",objective);
			printf("** INITIAL COMPUTED DUAL GAP = %g\n",dual_gap);
		
			while((gap > currentEpsilon) && (greedy_it <= param->max_iterations)){

					if(inverted_coreIdx[toward_vertex] < 0){
					//there is a new active point ...
						this->CheckMemoryForFW();
						coreIdx[coreNum] = toward_vertex;
						inverted_coreIdx[toward_vertex] = coreNum;
						coreNum++;
					}

					double normToward = sassoQ->getNorm(toward_vertex);

					if((coreNum > 0) && (std::abs(outAlpha[inverted_coreIdx[toward_vertex]] - tilde_delta) < TAU))
						break;

					greedy_it++; 
					/* Toward Step */

				
					double denominator_step = (Sk-(2.0*tilde_delta*Gk_istar)+(tilde_delta*tilde_delta*normToward));
					double step_size = dual_gap/(Sk-(2.0*tilde_delta*Gk_istar)+(tilde_delta*tilde_delta*normToward));
					

					if(step_size > 1.0){
						step_size = 1.0;
					}

					if(step_size < -TAU){
						printf("ERROR. STEP SIZE should not be negative: %g\n",step_size);
						//throw std::invalid_argument( "ERROR. STEP SIZE should not be negative\n");
					}

				
					if(greedy_it%fremessages==0){
						if(param->print_optimization_path){
							fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,-1.0);
						}
						printf("Iterations=%d, Objective = %.12f, GAP=%.12f, DUAL-GAP=%.12f, Active Features=%d, toward gradient=%g, toward index=%d, step-size=%g\n",greedy_it,objective,gap,dual_gap,coreNum,toward_gradient,toward_vertex,step_size);
						printf("Denom-step=%g, Sk=%g, Fk=%g, tilde*tilde=%g, normTow=%g\n",denominator_step,Sk,Fk,tilde_delta*tilde_delta,normToward);
					}
					
					
					double new_Sk = (1.0-step_size)*(1.0-step_size)*Sk + 2*tilde_delta*step_size*(1.0-step_size)*Gk_istar + tilde_delta*tilde_delta*step_size*step_size*normToward;
					double new_Fk = (1.0-step_size)*Fk + tilde_delta*step_size*sassoQ->getLT(toward_vertex);
					double new_objective = 0.5*new_Sk - new_Fk;
					//new_objective = (1.0/mdata)*new_objective;
					double old_objective = objective;
					double improvement = objective - new_objective;
					delta_objective = improvement;
					
					
					objective = new_objective;//update the objective function value
					Sk = new_Sk;
					Fk = new_Fk;

					double diff_alpha = 0.0;
					
					
					max_diff_iterates = -INFINITY;
					l1norm_solution = 0.0;

					
					for(int i=0; i < coreNum; i++){
						if(inverted_coreIdx[toward_vertex]!=i){
							diff_alpha = std::abs(-step_size*outAlpha[i]);
						} else {
							diff_alpha = std::abs(step_size*(tilde_delta-outAlpha[i]));
						}
						outAlpha[i] = outAlpha[i]*(1.0-step_size);

						if(inverted_coreIdx[toward_vertex]!=i){
							l1norm_solution += std::abs(outAlpha[i]);
						} else {
							l1norm_solution += std::abs(outAlpha[i]+tilde_delta*step_size);
						}

						if(max_diff_iterates < diff_alpha) {
							max_diff_iterates = diff_alpha;
						}

					}

					outAlpha[inverted_coreIdx[toward_vertex]] += tilde_delta*step_size;
		   			
		   			clock_t time_cycle_ = clock ();
					
		   			if(!param->randomized){//UPDATE GRADIENTS

		   	// 			for(int m=0; m<prob->l; m++){
		   	// 				double old_gradient = gradientALLPoints[m];
		   	// 				double kernel_product = sassoQ->kernel_eval(m,toward_vertex);
				 	// 		gradientALLPoints[m] = (1.0-step_size)*old_gradient + step_size*tilde_delta*kernel_product - step_size*sassoQ->getLT(m);
						// }
		   				Qfloat* kcol = sassoQ->get_KernelColumn(toward_vertex);
						for(int m=0; m<prob->l; m++){
		   					double old_gradient = gradientALLPoints[m];
		   					double kernel_product = (double)kcol[m];
				 			gradientALLPoints[m] = (1.0-step_size)*old_gradient + step_size*tilde_delta*kernel_product - step_size*sassoQ->getLT(m);
						}

						
		   			}

		   			time_cycle_ = clock() - time_cycle_;
		   			time_cycle_weights_FW += ((float)time_cycle_)/CLOCKS_PER_SEC;
		   			if(greedy_it%fremessages==0)
			   			printf("*********** TIME GRADIENTS = %g*************************\n", ((float)time_cycle_)/CLOCKS_PER_SEC);
				
					//toward_gradient = TowardVertex(toward_vertex);
					
	
					dual_gap = -INFINITY;
					int mcounter=0;
					sampler->reset();
					
					while(dual_gap<DUAL_GAP_TOL){
						toward_gradient = TowardVertex(toward_vertex,Sk,Fk,delta);
						tilde_delta = (toward_gradient < 0.0) ? delta:-1.0*delta;
						dual_gap = (Sk-Fk) - tilde_delta*toward_gradient;
						//printf("IN WHILE. Counter=%d\n",mcounter);
						//printf("Iterations=%d, Objective = %.12f, GAP=%.12f, DUAL-GAP=%.12f, Active Features=%d, toward gradient=%g, toward index=%d, step-size=%g\n",greedy_it,objective,gap,dual_gap,coreNum,toward_gradient,toward_vertex,step_size);
						mcounter++;
						if(mcounter%2000==0){
							throw std::invalid_argument("Loop to find the toward vertex is becoming too long. Aborting to avoid infinite loop ...\n" );
						}
					}
					
					tilde_delta = (toward_gradient < 0.0) ? delta:-1.0*delta;
					dual_gap = (Sk-Fk) - tilde_delta*toward_gradient;
					//dual_gap = (1.0/mdata)*dual_gap;

					if((dual_gap <= currentEpsilon) && (param->randomized) && (param->safe_stopping)){
						safe_stopping_check(Sk,Fk,delta,tilde_delta,dual_gap,toward_vertex,toward_gradient);
					}
	
				
					Gk_istar = toward_gradient + sassoQ->getLT(toward_vertex);
					
					if(param->stopping_criterion == STOPPING_WITH_INF_NORM){
						gap = max_diff_iterates;
					} else {
						if(param->stopping_criterion == STOPPING_WITH_OBJECTIVE) {
							gap = std::abs(delta_objective/old_objective);
						} else {
							gap = dual_gap;
						}
					}
		
					if(greedy_it%fremessages==0){
						printf("NEW TOWIDX=%d, CORRESP. COREIDX=%d, GAP=%g\n",toward_vertex, inverted_coreIdx[toward_vertex],gap);
					}

					
			}//end eps

		}//end eps cycle

		printf("** FINAL OBJECTIVE = %g\n",objective);
		printf("** FINAL GAP = %g\n",gap);
		printf("** FINAL DUAL GAP = %g\n",dual_gap);
		printf("** ITERATIONS = %d\n",greedy_it);
		printf("** L1-NORM SOLUTION = %.10f\n",l1norm_solution);
		printf("** SK (L2-NORM COTTER) = %.10f\n",Sk);
		printf("** Active Features=%d\n",coreNum);
		

		if(param->print_optimization_path){
		    fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,-1.0);
		}

	    if((!param->save_models_along_the_path) && previousSolution!=NULL){
	    	free(previousSolution);
	    	previousSolution = NULL;
	    }

	    previousSolution = newSolution;
	 
	    if(param->save_models_along_the_path){
	    	printf("writting model %d\n",delta_counter);
	    	ComputeLASSOSolution(models_found_in_the_path[delta_counter],0.0);
			newSolution=models_found_in_the_path[delta_counter];

		} else {
			ComputeLASSOSolution(newSolution,0.0);
		}

	    if(param->print_regularization_path){
 			//Sk is L2norm2Cotter
 			this->printLASSOSolution(newSolution, objective, delta, Sk);
		}
 	
		bool difference = true;
		bool intern = false;

	    if((param->quick_stop_regularization_path) && (previousSolution!=NULL) && (newSolution!=NULL)){ //compare solutions
	    	if(std::abs(delta-l1norm_solution)/delta > 0.01){
	    		//printf("INTERIOR SOLUTION\n");
	    		intern = true;
	    	} else {
	   	 		difference = compareLASSOSolutions(previousSolution,newSolution);  	
	   	 	}
	    }

	    if(!difference){
	    	printf("NO Difference between sucessive delta iterations\n");
	    	break;
	    }

	    if(intern){
	    	printf("Solution in the strict interior of the feasible region, i.e., already optimal.\n");
	    	break;
	    }
	} //end delta cycle	

	if(param->print_optimization_path)
		fclose(pathFile);

	return 1;
}


int SASSO::FullyCorrectiveFW(double convergence_eps, bool cooling, bool randomized){

	printf("FW Solver: Standard FW (with line search), ");
	cooling ? printf("cooling : YES, ") : printf("cooling : NO, ");
	randomized ? printf("randomized : YES\n") : printf("randomized : NO\n");


    FILE* pathFile;
    if(param->print_optimization_path)
	    pathFile = fopen(param->path_file_name,"w");   
 	
 	FILE* regPathFile = NULL;
    		
 	AllocateMemoryForFW(1000);
	
	greedy_it = 0;

	double Sk = 0.0;
	double Fk = 0.0;
	double Gk_istar = 0.0;
	double tilde_delta = 0;
	double sigma_istar;
	double gap = INFINITY;
	double delta_objective = INFINITY;
	double dual_gap = INFINITY;
	double max_diff_iterates = INFINITY;
	double objective = this->objective;
	double l1norm_solution = 0.0;
	double mdata = (double)prob->input_dim;
	bool did_last_optimization_exploring =  false;
	bool vertex_candidate_is_active = false;

	//objective = (1.0/mdata)*objective;
	int fremessages = param->frecuency_messages;
	if(fremessages <= 0)
		fremessages = 500;
	printf("FM:%d\n",fremessages);

	double delta, delta_min, delta_max, delta_step;

	if(param->computing_regularization_path){
		delta_min = param->reg_param_min;
		delta_max = param->reg_param_max;
		delta_step = param->reg_param_step;
		printf(">>>> COMPUTING REGULARIZATION PATH \n");
	} else {
		delta_min = param->reg_param;
		delta_max = param->reg_param;
		delta_step = param->reg_param;

	}

    double nsteps = param->n_steps_reg_path; 

	if(delta_min<TAU)
		delta_min = delta_max/nsteps;

	double logdmax = std::log(delta_max);
    double logdmin = std::log(delta_min);
    double logrange = logdmax - logdmin;

    delta_step = logrange/((double)nsteps-1.0);
	printf(">>>> DELTA MIN=%g, DELTA_MAX=%g, DELTA_STEP=%g, logrange=%g,nsteps=%g\n",delta_min,delta_max,delta_step,logrange,param->n_steps_reg_path);
	if(param->randomized)
		printf(">> USING RANDOMIZED VERSION!!\n");
	else
		printf(">> USING DETERMINISTIC VERSION!!\n");

	data_node* previousSolution = NULL;//sparse format
	data_node* newSolution = NULL;//sparse format
	
	if(std::abs(delta_max-delta_min)<TAU)
		nsteps = 1;

	for(int delta_counter = 0; delta_counter < nsteps; delta_counter++){

		delta = std::exp(logdmin+((double)delta_counter*delta_step));
	   
	   	//*** SCALING HEURISTIC ***/
	   	if((param->BORDER_WARM_START) && (delta>0.0) && (l1norm_solution>0.0)){
	   		printf("################# STARTING WITH BORDER WARM START \n");
	   		double scale_factor = delta/l1norm_solution;
	   		double new_Sk = scale_factor*scale_factor*Sk;
			double new_Fk = scale_factor*Fk;
			double new_objective = 0.5*new_Sk - new_Fk;
			if(new_objective<objective){
				l1norm_solution = 0.0;
			   	for(int i=0; i < coreNum; i++){
					outAlpha[i] = scale_factor*outAlpha[i];
					l1norm_solution += std::abs(outAlpha[i]);
				}
				objective = new_objective;
				Sk = new_Sk;
				Fk = new_Fk;	
			}
		} else{
			printf("################# STARTING WITHOUT BORDER WARM START \n");
		}

		printf("***** OBJECTIVE FOR NEW DELTA=%g\n",objective);
		printf("\n *************** REGULARIZATION WITH : %g ******************* \n", delta);
		
		double epsilonFactor = EPS_SCALING;
		double currentEpsilon = cooling ? INITIAL_EPS : convergence_eps/epsilonFactor;
	
		//iterate until desired precision 
		while((delta>TAU) && (currentEpsilon > convergence_eps)){// solve problem with current epsilon (warm start from the previous solution)
			currentEpsilon *= epsilonFactor;		
			currentEpsilon = (currentEpsilon < convergence_eps) ? convergence_eps: currentEpsilon;
			printf("EPS-Iteration: Iterating to achieve EPS = %g Current.EPS = %g N.Active.Points = %d\n",convergence_eps,currentEpsilon,coreNum);
			
			toward_vertex = -1;
			
			dual_gap = -INFINITY;

			sampler->reset();
			while(dual_gap<DUAL_GAP_TOL){
				toward_gradient = TowardVertex(toward_vertex,Sk,Fk,delta);
				tilde_delta = (toward_gradient < 0.0) ? delta:-1.0*delta;
				dual_gap = (Sk-Fk) - tilde_delta*toward_gradient;
			}

			//dual_gap = (1.0/mdata)*dual_gap;

			if((dual_gap <= currentEpsilon) && (param->randomized) && (param->safe_stopping) && (param->stopping_criterion == STOPPING_WITH_DUAL_GAP))
				safe_stopping_check(Sk,Fk,delta,tilde_delta,dual_gap,toward_vertex,toward_gradient);
	
			Gk_istar = toward_gradient + sassoQ->getLT(toward_vertex);
			max_diff_iterates = INFINITY;
			delta_objective = INFINITY;

			if(param->stopping_criterion == STOPPING_WITH_INF_NORM){
				gap = max_diff_iterates;
			} else {
				if(param->stopping_criterion == STOPPING_WITH_OBJECTIVE)
					gap = delta_objective;
				else
					gap = dual_gap;
			}

			printf("** INITIAL OBJECTIVE = %g\n",objective);
			printf("** INITIAL COMPUTED DUAL GAP = %g\n",dual_gap);
		
			did_last_optimization_exploring =  false;
			vertex_candidate_is_active = false;

			while(((gap > currentEpsilon) || !did_last_optimization_exploring) && (greedy_it <= param->max_iterations)){

					if(inverted_coreIdx[toward_vertex] < 0){
					//there is a new active point ...
						this->CheckMemoryForFW();
						coreIdx[coreNum] = toward_vertex;
						inverted_coreIdx[toward_vertex] = coreNum;
						coreNum++;
					}

					double normToward = sassoQ->getNorm(toward_vertex);

					if((coreNum > 0) && (std::abs(outAlpha[inverted_coreIdx[toward_vertex]] - tilde_delta) < TAU))
						break;

					greedy_it++; 
					/* Toward Step */

				
					double denominator_step = (Sk-(2.0*tilde_delta*Gk_istar)+(tilde_delta*tilde_delta*normToward));
					double step_size = dual_gap/(Sk-(2.0*tilde_delta*Gk_istar)+(tilde_delta*tilde_delta*normToward));
					

					if(step_size > 1.0){
						step_size = 1.0;
					}

					if(step_size < -TAU){
						printf("ERROR. STEP SIZE should not be negative: %g\n",step_size);
						//throw std::invalid_argument( "ERROR. STEP SIZE should not be negative\n");
					}

				
					if(greedy_it%fremessages==0){
						if(param->print_optimization_path){
							fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,-1.0);
						}
						printf("Iterations=%d, Objective = %.12f, GAP=%.12f, DUAL-GAP=%.12f, Active Features=%d, toward gradient=%g, toward index=%d, step-size=%g\n",greedy_it,objective,gap,dual_gap,coreNum,toward_gradient,toward_vertex,step_size);
						printf("Denom-step=%g, Sk=%g, Fk=%g, tilde*tilde=%g, normTow=%g\n",denominator_step,Sk,Fk,tilde_delta*tilde_delta,normToward);
					}
					
					
					double new_Sk = (1.0-step_size)*(1.0-step_size)*Sk + 2*tilde_delta*step_size*(1.0-step_size)*Gk_istar + tilde_delta*tilde_delta*step_size*step_size*normToward;
					double new_Fk = (1.0-step_size)*Fk + tilde_delta*step_size*sassoQ->getLT(toward_vertex);
					double new_objective = 0.5*new_Sk - new_Fk;
					//new_objective = (1.0/mdata)*new_objective;
					double old_objective = objective;
					double improvement = objective - new_objective;
					delta_objective = improvement;
					
					
					objective = new_objective;//update the objective function value
					Sk = new_Sk;
					Fk = new_Fk;

					double diff_alpha = 0.0;
					
					
					max_diff_iterates = -INFINITY;
					l1norm_solution = 0.0;

					
					for(int i=0; i < coreNum; i++){
						if(inverted_coreIdx[toward_vertex]!=i){
							diff_alpha = std::abs(-step_size*outAlpha[i]);
						} else {
							diff_alpha = std::abs(step_size*(tilde_delta-outAlpha[i]));
						}
						outAlpha[i] = outAlpha[i]*(1.0-step_size);

						if(inverted_coreIdx[toward_vertex]!=i){
							l1norm_solution += std::abs(outAlpha[i]);
						} else {
							l1norm_solution += std::abs(outAlpha[i]+tilde_delta*step_size);
						}

						if(max_diff_iterates < diff_alpha) {
							max_diff_iterates = diff_alpha;
						}

					}

					outAlpha[inverted_coreIdx[toward_vertex]] += tilde_delta*step_size;
		   			
		   			clock_t time_cycle_ = clock ();
					
		   			if(!param->randomized){//UPDATE GRADIENTS

		   	// 			for(int m=0; m<prob->l; m++){
		   	// 				double old_gradient = gradientALLPoints[m];
		   	// 				double kernel_product = sassoQ->kernel_eval(m,toward_vertex);
				 	// 		gradientALLPoints[m] = (1.0-step_size)*old_gradient + step_size*tilde_delta*kernel_product - step_size*sassoQ->getLT(m);
						// }
		   				Qfloat* kcol = sassoQ->get_KernelColumn(toward_vertex);
						for(int m=0; m<prob->l; m++){
		   					double old_gradient = gradientALLPoints[m];
		   					double kernel_product = (double)kcol[m];
				 			gradientALLPoints[m] = (1.0-step_size)*old_gradient + step_size*tilde_delta*kernel_product - step_size*sassoQ->getLT(m);
						}

						
		   			}

		   			time_cycle_ = clock() - time_cycle_;
		   			time_cycle_weights_FW += ((float)time_cycle_)/CLOCKS_PER_SEC;
		   			if(greedy_it%fremessages==0)
			   			printf("*********** TIME GRADIENTS = %g*************************\n", ((float)time_cycle_)/CLOCKS_PER_SEC);
				
					if(vertex_candidate_is_active)
						did_last_optimization_exploring = false;
					else
						did_last_optimization_exploring = true;

					if(param->stopping_criterion == STOPPING_WITH_INF_NORM){
						gap = max_diff_iterates;
					} else {
						if(param->stopping_criterion == STOPPING_WITH_OBJECTIVE) {
							gap = std::abs(delta_objective/old_objective);
						} else {
							gap = dual_gap;
						}
					}

					if((gap < currentEpsilon) && !did_last_optimization_exploring){
						dual_gap = -INFINITY;
						int mcounter=0;
						sampler->reset();
						
						while(dual_gap<DUAL_GAP_TOL){
							toward_gradient = TowardVertex(toward_vertex,Sk,Fk,delta);
							tilde_delta = (toward_gradient < 0.0) ? delta:-1.0*delta;
							dual_gap = (Sk-Fk) - tilde_delta*toward_gradient;
							//printf("IN WHILE. Counter=%d\n",mcounter);
							//printf("Iterations=%d, Objective = %.12f, GAP=%.12f, DUAL-GAP=%.12f, Active Features=%d, toward gradient=%g, toward index=%d, step-size=%g\n",greedy_it,objective,gap,dual_gap,coreNum,toward_gradient,toward_vertex,step_size);
							mcounter++;
							if(mcounter%2000==0){
								throw std::invalid_argument("Loop to find the toward vertex is becoming too long. Aborting to avoid infinite loop ...\n" );
							}
						}
						printf("EXPLORED ... \n");
						vertex_candidate_is_active = false;

					} else {
						toward_gradient = ActiveTowardVertex(toward_vertex,Sk,Fk,delta);
						tilde_delta = (toward_gradient < 0.0) ? delta:-1.0*delta;
						dual_gap = (Sk-Fk) - tilde_delta*toward_gradient;
						printf("ACTIVE ... \n");
						vertex_candidate_is_active = true;
					}

					tilde_delta = (toward_gradient < 0.0) ? delta:-1.0*delta;
					dual_gap = (Sk-Fk) - tilde_delta*toward_gradient;
					//dual_gap = (1.0/mdata)*dual_gap;

					if((param->stopping_criterion == STOPPING_WITH_DUAL_GAP) && (dual_gap <= currentEpsilon) && (param->randomized) && (param->safe_stopping)){
						safe_stopping_check(Sk,Fk,delta,tilde_delta,dual_gap,toward_vertex,toward_gradient);
					}
	
					Gk_istar = toward_gradient + sassoQ->getLT(toward_vertex);
							
					if(greedy_it%fremessages==0){
						printf("NEW TOWIDX=%d, CORRESP. COREIDX=%d, GAP=%g\n",toward_vertex, inverted_coreIdx[toward_vertex],gap);
					}


			}//end eps

		}//end eps cycle

		printf("** FINAL OBJECTIVE = %g\n",objective);
		printf("** FINAL GAP = %g\n",gap);
		printf("** FINAL DUAL GAP = %g\n",dual_gap);
		printf("** ITERATIONS = %d\n",greedy_it);
		printf("** L1-NORM SOLUTION = %.10f\n",l1norm_solution);
		printf("** SK (L2-NORM COTTER) = %.10f\n",Sk);
		printf("** Active Features=%d\n",coreNum);
		

		if(param->print_optimization_path){
		    fprintf(pathFile,"\n%d, %g, %g, %g",greedy_it,2.0*objective,dual_gap,-1.0);
		}

	    if((!param->save_models_along_the_path) && previousSolution!=NULL){
	    	free(previousSolution);
	    	previousSolution = NULL;
	    }

	    previousSolution = newSolution;
	 
	    if(param->save_models_along_the_path){
	    	printf("writting model %d\n",delta_counter);
	    	ComputeLASSOSolution(models_found_in_the_path[delta_counter],0.0);
			newSolution=models_found_in_the_path[delta_counter];

		} else {
			ComputeLASSOSolution(newSolution,0.0);
		}

	    if(param->print_regularization_path){
 			//Sk is L2norm2Cotter
 			this->printLASSOSolution(newSolution, objective, delta, Sk);
		}
 	
		bool difference = true;
		bool intern = false;

	    if((param->quick_stop_regularization_path) && (previousSolution!=NULL) && (newSolution!=NULL)){ //compare solutions
	    	if(std::abs(delta-l1norm_solution)/delta > 0.01){
	    		//printf("INTERIOR SOLUTION\n");
	    		intern = true;
	    	} else {
	   	 		difference = compareLASSOSolutions(previousSolution,newSolution);  	
	   	 	}
	    }

	    if(!difference){
	    	printf("NO Difference between sucessive delta iterations\n");
	    	break;
	    }

	    if(intern){
	    	printf("Solution in the strict interior of the feasible region, i.e., already optimal.\n");
	    	break;
	    }
	} //end delta cycle	

	if(param->print_optimization_path)
		fclose(pathFile);

	return 1;
}



bool SASSO::Initialize(){
	
	//** LASSO: INITILIZATION WITH WEIGHTS=0 **//
	//** LASSO: IN THIS CASE THE OBJETIVE FUNCTION VALUE IS 0.5*Y'Y **//

	sassoQ  = new SASSO_Q(prob, param, prob->alpha_orig);
	coreNum  = 0; 
	allocated_size_for_alpha = 0;
	this->objective = 0.0;


return true;

}


data_node* SASSO::printLASSOSolution(data_node* tempSolution, double objective, double delta, double L2norm2Cotter){
	
	printf("******** FILE NAME: %s ***********************\n",param->results_file_name);

	FILE* resultsFile = fopen(param->results_file_name,"a");   
 	
	double L1norm = 0.0;	
	double obj_value = objective;
	int count_actives = 0;
	
	int i=0;
	while(tempSolution[i].index!=-1){
		double value = tempSolution[i].value;
		L1norm += std::abs(value);

		if(std::abs(value) > TAU)
			count_actives++;

		i++;
	}

	fprintf(resultsFile,"%g",delta);
	fprintf(resultsFile," %d",count_actives);
	fprintf(resultsFile," %g",L1norm);
	fprintf(resultsFile," %g",obj_value);
	fprintf(resultsFile," %g",L2norm2Cotter);
	
	i=0;
	while(tempSolution[i].index!=-1){
		fprintf(resultsFile," %d:%g",tempSolution[i].index,tempSolution[i].value);
		i++;
	}

	fprintf(resultsFile,"\n");
	fclose(resultsFile);
	
	return tempSolution;
}

data_node* SASSO::showLASSOSolution(data_node* &tempSolution){
	
	ComputeLASSOSolution(tempSolution,0.0);
	int i=0;
	double L1norm=0.0;
	
	//while(tempSolution[i].index!=-1){
		//printf("... COMPUTING ... WEIGHT: IDX=%d, VAL=%g\n",tempSolution[i].index,tempSolution[i].value);
		//L1norm+=std::abs(tempSolution[i].value); 
		//i++;
	//}
	//printf("... L1 NORM =%g\n",L1norm);
	
	return tempSolution;
}

bool SASSO::compareLASSOSolutions(data_node* previousSolution, data_node* newSolution){
    
		    while((newSolution->index!=-1) && (previousSolution->index!=-1)){
				if(newSolution->index == previousSolution->index){
					if(std::abs(newSolution->value - previousSolution->value) > param->eps_regularization_path + TAU){
						//printf("DIFFERENCE %.10f %.10f\n",newSolution->value,previousSolution->value);
						//printf("DIFFERENCE %.10f %.10f\n",std::abs(newSolution->value - previousSolution->value), param->eps_regularization_path);
						return true;
					}
				} else {
					 return true;
				}
				previousSolution++;
				newSolution++;
			}
	   

	    printf("NO DIFFERENCE\n");
	    return false; 
}

double SASSO::ComputeLASSOSolution(data_node* &weights, double Threshold)
{
	int count_actives = 0;

	for(int i = 0; i < coreNum; i++)
	{  
		 if(std::abs(outAlpha[i]) > THRESHOLD_WEIGHTS)
				count_actives++;
	}

	weights = Malloc(struct data_node, count_actives+1);
	data_node* weightso = weights;

	count_actives = 0;
	for(int i = 0; i < coreNum; i++)
	{  
		 if(std::abs(outAlpha[i]) > THRESHOLD_WEIGHTS){
			int ii    = coreIdx[i];
			weights[count_actives].index = ii;
			weights[count_actives].value = outAlpha[i];
			count_actives++;
		}
	}

	weights[count_actives].index = INT_MAX;
	weights[count_actives].value = 0;

	qsort(weights,count_actives+1, sizeof(data_node), compare_data_node);

	weights[count_actives].index = -1;
	weights[count_actives].value = 0;
	printf("ACTIVES=%d\n",count_actives);
	return 0.0;
}

