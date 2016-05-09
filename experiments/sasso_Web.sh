#!/bin/bash
dataset_dir=..

DATASET=w8a
TRAIN_DATA=Web/w8a.original.train.01.txt
TEST_DATA=Web/w8a.original.test.01.txt
VAL_DATA=Web/w8a.original.test.01.txt.val.
TEST_DATA=Web/w8a.original.test.01.txt.test.
INIT_DIR=$METHOD/init
MODEL_DIR=$METHOD/model
TEST_DIR=$METHOD/test
K=0.1
METHOD=sasso
SMO_SOLUTION=inputSasso/${DATASET}_SVM_SMO_BIASED_100000-SupportSet.txt
SASSO_SOLUTION=
. base_sasso.sh "$@"