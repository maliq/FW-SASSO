#!/bin/bash
#dataset_dir=/home/ricky/Cotter-Code
#calculate_error=/home/ricky/Cotter-Code/issvm/computeMissRate.py
dataset_dir=..
DATASET=TIMIT
TRAIN_DATA=TIMIT/TIMIT.3.binary.train.01.txt
TEST_DATA=TIMIT/TIMIT.3.binary.test.01.txt
VAL_DATA=TIMIT/TIMIT.3.binary.test.01.txt.val.
TEST_DATA=TIMIT/TIMIT.3.binary.test.01.txt.test.
INIT_DIR=$METHOD/init
MODEL_DIR=$METHOD/model
TEST_DIR=$METHOD/test
K=0.025
METHOD=sasso
SMO_SOLUTION=inputSasso/${DATASET}_3binary_SVM_SMO_BIASED_100000-SupportSet.txt
SASSO_SOLUTION=
. base_sasso.sh "$@"