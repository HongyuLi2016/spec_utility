#!/bin/bash
#This script is used for preparing pbs script,output file in pbsscript folder
#rm -rf pbsscript
path=`pwd`
printf '#!/bin/bash
#PBS -N p%s
#PBS -o %s
#PBS -e %s
#PBS -l nodes=1:ppn=1
cd %s
/share/data/D/lhy/manga/SPS/mpl5/make_spec.py %s 
run_ppxf.py %s
ssp_maps.py %s -f dump
ssp_rebin_maps.py %s -f dump
' $1 $path/$1 $path/$1 $path $1 $1 $1 $1 >  $1/ppxf.pbs
