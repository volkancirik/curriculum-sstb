#!/usr/bin/env/sh

gpu=$1
prefix=$2
regime=$3
params=$4
addparams=$5

for dropout in 0 0.5
do
    echo "THEANO_FLAGS=device=$gpu python train.py --regime $regime --prefix $prefix --dropout $dropout $params $addparams"
    THEANO_FLAGS=device=$gpu python train.py --regime $regime --prefix $prefix --dropout $dropout $params $addparams
done
