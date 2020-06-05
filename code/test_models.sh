#!/bin/bash


outputpath=$1

# docker
dstdir=/root/preprocessed

# local
# dstdir=./data

PATH_TO_MODELS=model/Unet_norot_3m_preprocessed

# test
for FOLDR in ${PATH_TO_MODELS}/* ; do
    python ./src/predict.py --load ${FOLDR}/best.pth --data ${dstdir}
done

python ./src/submit.py --exp ${PATH_TO_MODELS} --csv ${dstdir}/test.csv --batch-size 2 --n-parts 1 --part 0 --to-save ${dstdir}/test_masks --tta 4 --submit-path $outputpath --thresh 0.5 --watershed
