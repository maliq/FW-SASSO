#ifndef MCACHE_H_
#define MCACHE_H_

#include <stdio.h>
#include <stdlib.h>
#include "SASSO_definitions.h" 

class mCache
{
public:
	mCache(int size_, int depth_, int max_index){
		size=size_;
		depth=depth_;

		positions = new int[max_index];
		use_stats = new int[size];
		inverted_positions = new int[size];
		pointers_to_data = new Qfloat*[size];

		for(int i=0; i< max_index; i++)
			positions[i]=-1;
		
		for(int i=0; i< size; i++){
			use_stats[i]=0;
			inverted_positions[i]=-1;
		}
		
		used=0;
	}

	~mCache(){

		for(int i=0; i< used; i++)
			delete[] pointers_to_data[i];
		
		delete[] positions;
		delete[] use_stats; 
		delete[] pointers_to_data; 
		delete[] inverted_positions;

	}

	Qfloat* get_data(int idx, bool& has_to_fill);
	bool isCached(int idx) { return (positions[idx]>=0); } 

protected:		

	int size;
	int used;
	int depth;
	int *positions;
	int *inverted_positions;
	int *use_stats;
	Qfloat** pointers_to_data;

};


#endif /*MCACHE_H_*/