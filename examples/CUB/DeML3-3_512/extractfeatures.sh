#!/usr/bin/env sh
EX_PATH=/home/pris2/DeML/examples/CUB/DeML3-3_512
MODEL_PATH=$EX_PATH"/run_DeML3-3_512"
FEATURE_PATH=$EX_PATH"/features"

for j in $(seq 10)
do
i=`expr \( $j + 20 \) \* 500`
k=`expr \( 0 + 0 \) \* 1`
modelname=$MODEL_PATH/model_iter_${i}.caffemodel
despath=$FEATURE_PATH/model_512_${i}
echo $i;
echo $modelname;
echo $despath;
/home/pris2/DeML/build/tools/extract_features \
     $modelname \
     "fea_extract_DeML3-3_512.prototxt" \
     "concat" \
     $despath \
     120 \
     GPU \
     0
done

