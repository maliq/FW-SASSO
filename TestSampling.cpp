#include "BlockSampler.h"
#include <stdio.h>

int main(int argc, char **argv){

	int ndata = 1299;
	int sample_size = 100;

	BlockSampler* sampler = new BlockSampler(ndata,sample_size);

	
	printf("\n##################### START ##########################\n");

	int counter=0;
	while(counter<sampler->getNBlocks()){
		int block = sampler->setRandomBlock();
		printf("\n##########################################################\n");
		printf("Block=%d\n",block);
		for(int k=sampler->getStartCurrentBlock(); k<sampler->getEndCurrentBlock(); k++){
			int idx = k;
			printf("IDX=%d, ",idx);
		}
		counter++;
		if(counter%5==0){
			sampler->reset();
			printf("\n###################### RESETTING ##########################\n");
		}
		printf("\n");
	}
	return 1;

}