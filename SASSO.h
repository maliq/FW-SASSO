#include <stdio.h>
#include <stdlib.h>        
#include "SASSO_definitions.h" 
#include "SASSO_kernel.h" 
#include <limits.h>
#include "BlockSampler.h"

class SASSO
{

	private:

		const sasso_parameters *param;//LASSO parameters
		const sasso_problem   *prob;//data

		double delta_reg;

		SASSO_Q *sassoQ; //class managing dot products for the LASSO problem
		
		int init_method;//initialization method
	
		double 	objective;//objective function value
	
		int    	toward_vertex;//idx of the training point to move the solution towards 
		int    	away_vertex;//idx of the training point to move the solution away
		double  toward_gradient;
		double  away_gradient;

		int  	*coreIdx; //contains the "data ids" of active points, i.e. coreIdx[i]=j means that the i-th active point is the j-th example
		int     *inverted_coreIdx;//-1 if the point is not active, otherwise inverted_coreIdx[j]=i means that the j-th active point is the i-th example
		int   	coreNum; //number of active points

		int     maxNumBasis; //maximum allowed number of active points

		double intercept;

		double  *outAlpha;//stores the last solution to the optimization problem
		
		data_node** models_found_in_the_path;

 		int greedy_it; //FW iterations
    	
		int allocated_size_for_alpha;
		int NPRINT;
		//cached things for computations
		Qfloat *gradientALLPoints; //gradient coordinates corresponding to the ALL points, always updated
		Qfloat *gradientActivePoints; //gradient coordinates corresponding to the active points (coreset points), always updated
	
		Qfloat *Q_actives_dot_toward; 
		Qfloat *Q_actives_dot_away; 
		Qfloat *previousQcolumn;

		double time_upt_residuals;
		double time_towards_1;
		double time_towards_2;
		double time_cycle_weights_FW;

		BlockSampler* sampler;

		int StandardFW(double convergence_eps, bool cooling, bool randomized);
		int FullyCorrectiveFW(double convergence_eps, bool cooling, bool randomized);

		double TowardVertex(int &towardIdx,double Sk, double Fk, double delta); 
		double ActiveTowardVertex(int &towardIdx, double Sk, double Fk, double delta);

		double AwayVertex(int &awayIdx); 
		
		double ActiveTowardVertex(int &towardIdx);

		double ComputeGradientCoordinate(int idx, Qfloat** Qcolumn);
		int ChooseRandomIndex();

		bool Initialize();
		int Yildirim_Initialization();
		int RandomSet_Initialization();
		int FullSet_Initialization();

		bool AllocateMemoryForFW(int initial_size);
		bool CheckMemoryForFW();
		bool Clean();

		double safe_stopping_check(double Sk, double Fk, double delta, double &tilde_delta, double &Gap, int &towardIdx, double &towardGrad);
		int nsamplings_randomized_iterations;
	
public:

	SASSO(const sasso_problem *_prob, const sasso_parameters* _param){

		NPRINT = 0;

		std::srand(std::time(0));
//		std::srand(0);

		prob  = _prob;
		param = _param;
		delta_reg = param->reg_param;
	 	inverted_coreIdx = new int[prob->l];
		coreIdx = new int[prob->l];
		
		allocated_size_for_alpha = 0;
		
		for(int i = 0; i < prob->l; i++)
		{	
			inverted_coreIdx[i] = -1;
			coreIdx[i] = -1;
		}

		outAlpha = Malloc(double,1);
	
		sassoQ  = NULL;		
		
		greedy_it = 0;

		sampler = new BlockSampler(prob->l,param->sample_size);

		time_upt_residuals=0.0;
		time_towards_1=0.0;
		time_towards_2=0.0;
		time_cycle_weights_FW=0.0;
		intercept = 0.0;
	}

	~SASSO(){

		delete [] coreIdx;
		delete [] inverted_coreIdx;
		free(outAlpha);
		
		
	}


	int Solve(double FW_eps, int method, bool cooling, bool randomized);
	int Solve(double FW_eps, int method, bool cooling, bool randomized, data_node** models_found_in_the_path_);


	double ComputeLASSOSolution(data_node* &weights, double Threshold);
	double GetObjective(){
		return objective;
	}

	double GetFWIterations(){
		return (double)greedy_it;
	}

	double getIntercept(){
		return intercept;
	}

	void getStats(sasso_stats* stats){
			stats->n_iterations = (double)GetFWIterations();
			stats->n_performed_dot_products = (double)sassoQ->get_real_kevals();
			stats->n_requested_dot_products = (double)sassoQ->get_requested_kevals();
			stats->time_towards_random=time_towards_1;
			stats->time_towards_active=time_towards_2;
			stats->time_cycle_weights_FW=time_cycle_weights_FW;
	}

	data_node* showLASSOSolution(data_node* &tempSolution);
	data_node* printLASSOSolution(data_node* tempSolution, double objective, double delta, double L2norm2Cotter);

	bool compareLASSOSolutions(data_node* previousSolution, data_node* newSolution);

};	
