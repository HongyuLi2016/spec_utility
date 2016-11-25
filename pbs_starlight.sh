#!/bin/bash
#This script is used for preparing pbs script,output file in pbsscript folder
#rm -rf pbsscript
mkdir -p $1/pbsscript 
mkdir -p $1/junk
path=`pwd`
for fname in `ls $1/grid/grid*.in`
do
tem=$(echo $fname | cut -f3 -d/)
printf '#!/bin/bash
#PBS -N %s
#PBS -o %s/junk
#PBS -e %s/junk
#PBS -l nodes=1:ppn=1
cd %s
starlight < %s' $1 $path/$1 $path/$1 $path/$1 grid/$tem >  $1/pbsscript/$tem.pbs
done
