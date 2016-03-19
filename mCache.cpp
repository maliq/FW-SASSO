
#include "mCache.h"
#include <cstdlib>

Qfloat* mCache::get_data(int idx, bool& has_to_fill){
		if(isCached(idx)){
			has_to_fill = false;
			use_stats[positions[idx]]+=1;
			return pointers_to_data[positions[idx]];
		} else {
			has_to_fill = true;
			if(used<size){//there is space
				Qfloat* new_column = new Qfloat[depth];
				positions[idx] = used;
				inverted_positions[used] = idx;
				pointers_to_data[used] = new_column;
				use_stats[used]=1;
				used++;
				return new_column;
			} else {
				//make space
				int min_use = 260144000;
				int idx_min_use = std::rand()%size;

				// for(int i=0; i< size; i++){
				// 	if(use_stats[i]<min_use){
				// 		idx_min_use=i;
				// 		min_use=use_stats[i];
				// 		if(use_stats[i]>0)
				// 			use_stats[i]-=1;
				// 	}
				// }
				
				int pos_to_use_in_cache = idx_min_use;
				positions[inverted_positions[pos_to_use_in_cache]] = -1;
				positions[idx] = pos_to_use_in_cache;
				inverted_positions[pos_to_use_in_cache] = idx;
				use_stats[pos_to_use_in_cache] = 1;

				return pointers_to_data[pos_to_use_in_cache];
		
			}
		}
}