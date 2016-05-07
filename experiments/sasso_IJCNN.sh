#!/bin/bash
dataset_dir=..

DATASET=IJCNN
ITERATIONS=100000
TRAIN_DATA=IJCNN/ijcnn1.original.train.01.txt
TEST_DATA=IJCNN/ijcnn1.original.test.01.txt
VAL_DATA=IJCNN/ijcnn1.original.test.01.txt.val.
TEST_DATA=IJCNN/ijcnn1.original.test.01.txt.test.
K=1.0
METHOD=sasso
SMO_SOLUTION=inputSasso/${DATASET}_SVM_SMO_BIASED_100000-SupportSet.txt
SASSO_SOLUTION=
. base_sasso.sh "$@"