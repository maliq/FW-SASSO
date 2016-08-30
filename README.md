 
# The SASSO implemented via FRANK-WOLFE.

## Features
* Deterministic version fully tested and implemented with recursive gradient updates and a special cache data structure (mcache) designed to store entire columns of the kernel matrix. this results much better than the old scache in this setting (deterministic).
* The models computed along the path are saved, as usual, to the results file, but are also saved for further processing.
* Testing of the models computed along the path is efficiently implemented using an special data structure (test kernel).
* The latter can be applied immediately after computing the path, or from scratch providing the name of the results file.


## Datasets
* Adult https://raw.githubusercontent.com/maliq/SASSO_DATASETS/master/w8a.tar.gz
* IJCNN https://raw.githubusercontent.com/maliq/SASSO_DATASETS/master/IJCNN.tar.gz
* Web https://raw.githubusercontent.com/maliq/SASSO_DATASETS/master/w8a.tar.gz
* TIMIT https://raw.githubusercontent.com/maliq/SASSO_DATASETS/master/TIMIT.tar.gz
* MNIST https://raw.githubusercontent.com/maliq/SASSO_DATASETS/master/MNIST.tar.gz

## Usage

### General usage.

    ./sasso-train -E option [params] file1 file2

### Syntonization of b, Option  -E 4

Reads the original SASSO result and determines the value of b for each model in the regularization path.

	-V filename indicates in which dataset the syntonization is performed, i.e.,  the data used to measure classification errors associated to each value of b
	-k 2 -g 0.025 kernel parameters as usual
	-X filename: filename to save the errors on the validation set obtained after syntonizing b, optional
	file1: file with the original SVM, i.e., original b, support vectors and weights
	file2: file with the original SASSO result

> new sasso result (with new b's) are saved adding the string ".---SYNC-B---." to the name of the file containing the original sasso result.

#### Example:

	./sasso-train -E 4 -k 2 -g 0.025 -V experiment-b-timit/TIMIT.3.binary.test.01.txt experiment-b-timit/TIMIT_3binary_SVM_SMO_BIASED-SupportSet.txt experiment-b-timit/TIMIT_3binary_SVM_SMO_BIASED-SupportSet.txt.SASSO.RESULTS.FW.1.0E-07.DETERMINISTIC.2016-01-29-14hrs-29mins-48secs.txt

### Testing the regularization path, Option  -E 3

Reads a SASSO result and computes the test error for each model in the regularization path.

	-T filename indicates in which dataset the testing is performed, i.e.,  the data used to measure classification errors
	-k 2 -g 0.025 kernel parameters as usual
	-X filename: filename to save the testing results, optional
	file1: file with the original SVM, i.e., original b, support vectors and weights
	file2: file with the SASSO result we want to evaluate

This procedure (testing) can be performed on a sasso result with and without syntonization of b
In the last case, the value of b for each model in the 	regularization path is the value of b of the original (non sparse) SVM

Example:

	./sasso-train -E 3 -k 2 -g 0.1 -T experiment-b-web/w8a.original.test.01.txt -X experiment-b-web/results-web-AFTER-SYNC-TEST.txt experiment-b-web/w8a_SVM_SMO_BIASED_100000-SupportSet.txt experiment-b-web/w8a_SVM_SMO_BIASED_100000-SupportSet.txt.SASSO.RESULTS.FW.1.0E-07.DETERMINISTIC.2016-01-29-12hrs-37mins-01secs.txt.---SYNC-B---.txt

	./sasso-train -E 3 -k 2 -g 0.025 -T experiment-b-timit/TIMIT.3.binary.test.01.txt -X experiment-b-timit/results_new_testing.txt experiment-b-timit/TIMIT_3binary_SVM_SMO_BIASED-SupportSet.txt experiment-b-timit/TIMIT_3binary_SVM_SMO_BIASED-SupportSet.txt.SASSO.RESULTS.FW.1.0E-07.DETERMINISTIC.2016-01-29-14hrs-29mins-48secs.txt


### Test using 10 validation/test split, Option -E 5


Test sasso SVM using 10 validation/test split, also it allows syntonize bias using the validation dataset before test the sasso SVM.
When syntonize bias, generates a file with the same name of sasso svm but with suffix "SUMMARY-B.SYNTONIZATION" with time and B tested.
Command:

    ./sasso-train -E 5 options optimal_SVM_file sasso_SVM_file

Options to test using 10 split test set, parameters:

    -V Vadalition dataset wildcard filename, only needed if syntonize bias, ie ijcnn1.original.test.01.txt.val. that will be complete from 1 to 10 using: (ijcnn1.original.test.01.txt.val.1, ijcnn1.original.test.01.txt.val.2, .., ijcnn1.original.test.01.txt.val.10)
    -T Test dataset wildcard filename, ie ijcnn1.original.test.01.txt.test. that will be complete from 1 to 10 using: (ijcnn1.original.test.01.txt.test.1, ijcnn1.original.test.01.txt.test.2, .., ijcnn1.original.test.01.txt.test.10)
    -b value : 1 to syntonize bias 0 otherwise. default value: 0
    -X filename: optional output filename to write results from test process.
    -K kernel used, 1: POLINOMIAL, 2: GAUSSIAN
    -g gaussian parameter, depends of dataset.

Example:

    ./sasso-train -E 5 -k 2 -g 1.0 -b 0 -V /Users/maliq/Documents/sasso/IJCNN/ijcnn1.original.test.01.txt.val. -T /Users/maliq/Documents/sasso/IJCNN/ijcnn1.original.test.01.txt.test. -X /Users/maliq/Documents/sasso/experiments/result/IJCNN_sasso_10_step.txt /Users/maliq/Documents/sasso/experiments/inputSasso/IJCNN_SVM_SMO_BIASED_100000-SupportSet.txt /Users/maliq/Documents/sasso/experiments/inputSasso/IJCNN_SVM_SMO_BIASED_100000-SupportSet.txt.SASSO.RESULTS.FW.1.0E-07.DETERMINISTIC.2016-01-29-12hrs-36mins-07secs.txt


## TODO
* Randomized version works but has not been optimized in terms of efficiency.
* Probably, it is a good idea to add the recursive computation of the gradients corresponding to active points.


