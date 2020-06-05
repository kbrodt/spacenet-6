#!/bin/bash


GPU=0,1,2,3
N_GPUS=4

# docker
dstdir=/root/preprocessed

# local
# dstdir=./data/preprocessed_norot_3m

SMODEL=Unet
LR=0.0003
WORK_DIR=model/${SMODEL}_norot_3m_preprocessed

BS=4
ENCODER=efficientnet-b7

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=${N_GPUS} ./src/train.py --work-dir ${WORK_DIR} --batch-size ${BS} --lr ${LR} --encoder ${ENCODER} --epochs 300 --n-classes 3 --csv train.csv --data ${dstdir} --lovasz 999 --smodel ${SMODEL} --w3m --n-folds 5 --fold 0 --ft

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=${N_GPUS} ./src/train.py --work-dir ${WORK_DIR} --batch-size ${BS} --lr ${LR} --encoder ${ENCODER} --epochs 300 --n-classes 3 --csv train.csv --data ${dstdir} --lovasz 999 --smodel ${SMODEL} --w3m --n-folds 5 --fold 1 --ft

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=${N_GPUS} ./src/train.py --work-dir ${WORK_DIR} --batch-size ${BS} --lr ${LR} --encoder ${ENCODER} --epochs 300 --n-classes 3 --csv train.csv --data ${dstdir} --lovasz 999 --smodel ${SMODEL} --w3m --n-folds 5 --fold 2 --ft

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=${N_GPUS} ./src/train.py --work-dir ${WORK_DIR} --batch-size ${BS} --lr ${LR} --encoder ${ENCODER} --epochs 300 --n-classes 3 --csv train.csv --data ${dstdir} --lovasz 999 --smodel ${SMODEL} --w3m --n-folds 5 --fold 3 --ft

CUDA_VISIBLE_DEVICES=${GPU} python -m torch.distributed.launch --nproc_per_node=${N_GPUS} ./src/train.py --work-dir ${WORK_DIR} --batch-size ${BS} --lr ${LR} --encoder ${ENCODER} --epochs 300 --n-classes 3 --csv train.csv --data ${dstdir} --lovasz 999 --smodel ${SMODEL} --w3m --n-folds 5 --fold 4 --ft
