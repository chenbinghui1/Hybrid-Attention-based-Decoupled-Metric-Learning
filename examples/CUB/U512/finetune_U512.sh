#!/usr/bin/env sh

GLOG_logtostderr=1 /home/pris2/DeML/build/tools/caffe train \
    --solver=solver_U512.prototxt \
   --weights=../pre-trained-model/bvlc_googlenet.caffemodel --gpu=0
