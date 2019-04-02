#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
template <typename Dtype>
void EntropyLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int N = bottom[0]->num();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  Dtype dot;
  caffe_gpu_add_scalar(bottom[0]->count(), Dtype(1e-5), bottom[0]->mutable_gpu_data());
  caffe_gpu_log(bottom[0]->count(), bottom_data, log_data_.mutable_gpu_data());
  caffe_gpu_dot(bottom[0]->count(), bottom_data, log_data_.gpu_data(), &dot);
  top_data[0] = dot * Dtype(-1) / Dtype(N);
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    
    const int N = bottom[0]->num();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    
    caffe_gpu_add_scalar(bottom[0]->count(), Dtype(1), log_data_.mutable_gpu_data());
    caffe_gpu_axpby(bottom[0]->count(), Dtype(-1)/Dtype(N)*top_diff[0], log_data_.gpu_data(), Dtype(0), bottom_diff);
}
INSTANTIATE_LAYER_GPU_FUNCS(EntropyLossLayer);
}
