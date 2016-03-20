 
# The SASSO implemented via FRANK-WOLFE.

## Features
* Deterministic version fully tested and implemented with recursive gradient updates and a special cache data structure (mcache) designed to store entire columns of the kernel matrix. this results much better than the old scache in this setting (deterministic).
* The models computed along the path are saved, as usual, to the results file, but are also saved for further processing.
* Testing of the models computed along the path is efficiently implemented using an special data structure (test kernel).
* The latter can be applied immediately after computing the path, or from scratch providing the name of the results file.

## TODO
* Randomized version works but has not been optimized in terms of efficiency.
* Probably, it is a good idea to add the recursive computation of the gradients corresponding to active points.

