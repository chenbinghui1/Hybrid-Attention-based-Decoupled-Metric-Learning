#!/usr/bin/env sh

GLOG_logtostderr=1 /home/pris2/DeML/build/tools/caffe train \
    --solver=solver_DeML3-3_512.prototxt \
   --weights=../pre-trained-model/3nets.caffemodel --gpu=0,1
