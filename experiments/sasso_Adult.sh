#!/usr/bin/env bash
#dataset_dir=/home/ricky/Cotter-Code
#calculate_error=/home/ricky/Cotter-Code/issvm/computeMissRate.py
dataset_dir=..


DATASET=Adult
TRAIN_DATA=Adult/a8a.original.train.01.txt
VAL_DATA=Adult/a8a.original.test.01.txt.val.
TEST_DATA=Adult/a8a.original.test.01.txt.test.
METHOD=sasso
SMO_SOLUTION=inputSasso/${DATASET}_SVM_SMO_BIASED_100000-SupportSet.txt
SASSO_SOLUTION=/Users/maliq/Documents/sasso/experiments/inputSasso/TIMIT_3binary_SVM_SMO_BIASED-SupportSet.txt.SASSO.RESULTS.FW.1.0E-05.DETERMINISTIC.2016-05-06-17hrs-54mins-20secs.txt
K=0.1
. base_sasso.sh "$@"



