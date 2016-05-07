#!/usr/bin/env bash

usage()
{
cat << EOF
usage: $0 options

This script run the test1 or test2 over a machine.

OPTIONS:
   -o value     operation: train or test options
   -B           syntonice B, default disable
   -T value     tolerance, default: 0.00001
   -S value     SC, dedault:0
   -R           enable ramdonized version, default: disabled
   -F value     FM, default: 100000
   -N value     sasso path point, default: 10
   -D value     FD, default: 5000
   -A           enable tunning SV's, default: disabled
   -v           Verbose
EOF
}

OPTIND=1
OP=
B=0
SC=0
R=0
FM=100000
NS=10
FD=5000
MA=0
TOL=0.00001
while getopts "hbRAT:S:F:N:D:o:s:v" OPTION
do
     case $OPTION in
         h)
             usage
             ;;
         b)
             B=1
             echo "set B: ${B}"
             ;;
         T)
             TOLs=(${OPTARG})
             echo "set T: ${TOLs}"
             ;;
         S)
             SC=$OPTARG
             echo "set SC: ${SC}"
             ;;
         E)
             epsilons=($OPTARG)
             echo "set e: $OPTARG"
             ;;
         R)
             R=1
             echo "set n: ${norms}"
             ;;
         F)
             SC=$OPTARG
             echo "set FM: ${FM}"
             ;;
         N)
             NS=${OPTARG}
             echo "set NS: ${NS}"
             ;;
         D)
             FD=${OPTARG}
             echo "set FD: ${FD}"
             ;;
         A)
             MA=1
             echo "set MA: ${MA}"
             ;;
         o)
             OP=${OPTARG}
             echo "set op: ${OPTARG}"
             ;;
         s)
             SASSO_SOLUTION=${OPTARG}
             echo "set SASSO_SOLUTION: ${SASSO_SOLUTION}"
             ;;
         v)
             VERBOSE=1
             ;;
         ?)
             usage
#             exit
             ;;
     esac
done

if [[ -z $OP ]]
then
     usage
fi

if [ "$OP" == "train" ]; then
    echo "sasso-train -E 1 -k 2 -g ${K} -SC ${SC} -RS ${R} -FM ${FM} -NS ${NS} -FD ${FD} -e ${TOL} -T $dataset_dir/$TRAIN_DATA ${SMO_SOLUTION}"
    sasso-train -E 1 -k 2 -g ${K} -SC ${SC} -RS ${R} -FM ${FM} -NS ${NS} -FD ${FD} -e ${TOL} -T $dataset_dir/$TRAIN_DATA ${SMO_SOLUTION}
fi
if [ "$OP" == "test" ]; then
    echo "sasso-train -E 5 -k 2 -g ${K} -b ${B} -V ${VAL_DATA} -T ${TEST_DATA} -X sasso_result/${DATASET}_sasso_B-${B}_test.txt ${SMO_SOLUTION} ${SASSO_SOLUTION}"
    sasso-train -E 5 -k 2 -g ${K} -b ${B} -V ${dataset_dir}/${VAL_DATA} -T ${dataset_dir}/${TEST_DATA} -X sasso_result/${DATASET}_sasso_MA-${MA}_B-${B}_test.txt ${SMO_SOLUTION} ${SASSO_SOLUTION}
fi