#!/usr/bin/env bash
declare -a DS_LIST=(Adult) #IJCNN MNIST TIMIT Web)
for DS in "${DS_LIST[@]}"
do
    ln -s ../sasso/experiments/sasso_${DS}.sh sasso_${DS}.sh
done
ln -s ../sasso/experiments/base_sasso.sh base_sasso.sh