//bottom[0] should be probabilities, i.e. ~[0,1]. This loss is conventional entropy loss. if like NIPS18 paper: Maximum Entropy Fine-Grained, the loss weight should be negative. e.g. -0.5.
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
template <typename Dtype>
void EntropyLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
     LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
      top[0]->Reshape(1,1,1,1); 
      log_data_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int N = bottom[0]->num();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  caffe_add_scalar(bottom[0]->count(), Dtype(1e-5), bottom[0]->mutable_cpu_data());
  caffe_log(bottom[0]->count(), bottom_data, log_data_.mutable_cpu_data());
  top_data[0] = caffe_cpu_dot(bottom[0]->count(), bottom_data, log_data_.cpu_data()) / Dtype(N) * Dtype(-1);
  
}

template <typename Dtype>
void EntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const int N = bottom[0]->num();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    
    caffe_add_scalar(bottom[0]->count(), Dtype(1), log_data_.mutable_cpu_data());
    caffe_cpu_axpby(bottom[0]->count(), Dtype(-1)/Dtype(N)*top_diff[0], log_data_.cpu_data(), Dtype(0), bottom_diff);
}
#ifdef CPU_ONLY
STUB_GPU(EntropyLossLayer);
#endif

INSTANTIATE_CLASS(EntropyLossLayer);
REGISTER_LAYER_CLASS(EntropyLoss);

}
