#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
template <typename Dtype>
void InnerProductLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
     LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void InnerProductLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
      top[0]->Reshape(1,1,1,1); 
}

template <typename Dtype>
void InnerProductLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->mutable_cpu_data()[0] = bottom[0]->cpu_data()[0];
}

template <typename Dtype>
void InnerProductLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  bottom[0]->mutable_cpu_diff()[0] = top[0]->cpu_diff()[0];
}
#ifdef CPU_ONLY
STUB_GPU(InnerProductLossLayer);
#endif

INSTANTIATE_CLASS(InnerProductLossLayer);
REGISTER_LAYER_CLASS(InnerProductLoss);

}
