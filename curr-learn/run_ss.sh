#!/usr/bin/env/sh

gpu=$1
prefix=$2
regime=$3
params=$4

for root in 'all' 'root'
do
    for dropout in 0 0.5
    do
	echo "THEANO_FLAGS=device=$gpu python train.py --dataset ss --root $root --regime $regime --prefix $prefix --dropout $dropout $params"
	THEANO_FLAGS=device=$gpu python train.py --dataset ss --root $root --regime $regime --prefix $prefix --dropout $dropout $params
    done
done
