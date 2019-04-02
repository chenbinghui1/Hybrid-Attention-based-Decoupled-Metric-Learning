#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
template <typename Dtype>
void InnerProductLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  caffe_gpu_scale(1,Dtype(1),bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void InnerProductLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  caffe_gpu_scale(1,Dtype(1),top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff());
}
INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLossLayer);
}
