#!/bin/bash

# docker
dstdir=/root/preprocessed

# local
# dstdir=/home/brodt/kaggle/spacenet-6/code/data/preprocessed_norot
# dstdir=/home/brodt/kaggle/spacenet-6/code/data/preprocessed_norot_3m

# dstdir=/home/brodt/kaggle/spacenet-6/code/data/preprocessed
# dstdir=/home/brodt/kaggle/spacenet-6/code/data/preprocessed_3m

mkdir -p $dstdir

# settings="\
# --rotationfilelocal $dstdir/SAR_orientations.txt \
# --maskdir $dstdir/masks \
# --sarprocdir $dstdir/sartrain \
# --opticalprocdir $dstdir/optical \
# --traincsv $dstdir/train.csv \
# --validcsv $dstdir/valid.csv \
# --opticaltraincsv $dstdir/opticaltrain.csv \
# --opticalvalidcsv $dstdir/opticalvalid.csv \
# --testcsv $dstdir/test.csv \
# --yamlpath $dstdir/sar.yaml \
# --opticalyamlpath $dstdir/optical.yaml \
# --modeldir $dstdir/weights \
# --testprocdir $dstdir/sartest \
# --testoutdir $dstdir/inference_continuous \
# --testbinarydir $dstdir/inference_binary \
# --testvectordir $dstdir/inference_vectors \
# --rotate \
# --transferoptical \
# --mintrainsize 20 \
# --mintestsize 80 \
# "


settings="\
--maskdir $dstdir/masks \
--sarprocdir $dstdir/sartrain \
--opticalprocdir $dstdir/optical \
--traincsv $dstdir/train.csv \
--validcsv $dstdir/valid.csv \
--opticaltraincsv $dstdir/opticaltrain.csv \
--opticalvalidcsv $dstdir/opticalvalid.csv \
--testcsv $dstdir/test.csv \
--yamlpath $dstdir/sar.yaml \
--opticalyamlpath $dstdir/optical.yaml \
--modeldir $dstdir/weights \
--testprocdir $dstdir/sartest \
--testoutdir $dstdir/inference_continuous \
--testbinarydir $dstdir/inference_binary \
--testvectordir $dstdir/inference_vectors \
--transferoptical \
--mintrainsize 20 \
--mintestsize 80 \
"
