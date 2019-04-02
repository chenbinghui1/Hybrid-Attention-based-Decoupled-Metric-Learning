//kl diverse loss [0]log[0]/[1]
#include <algorithm>
#include <vector>
#include "stdio.h"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/kl_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>
/**********************/
/**/
namespace caffe {

template <typename Dtype>
void KLLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
}
template <typename Dtype>
void KLLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
    top[0]->Reshape(loss_shape);
    temp_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void KLLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* P_data = bottom[0]->cpu_data();
    const Dtype* Q_data = bottom[1]->cpu_data();
    Dtype* temp = temp_.mutable_cpu_data();
    //P/Q
    caffe_div(bottom[0]->count(), P_data, Q_data, temp);
        //compute gradients w.r.t Q
        caffe_cpu_axpby(bottom[0]->count(), Dtype(-1), temp_.cpu_data(), Dtype(0), bottom[1]->mutable_cpu_diff());
    //log(P/Q)
    caffe_log(bottom[0]->count(), temp_.cpu_data(), temp);
        //compute gradients w.r.t P
        caffe_add_scalar(bottom[0]->count(), Dtype(1), temp);
        caffe_copy(bottom[0]->count(), temp, bottom[0]->mutable_cpu_diff());
    //compute loss
    caffe_add_scalar(bottom[0]->count(), Dtype(-1), temp);
    caffe_mul(bottom[0]->count(), P_data, temp_.cpu_data(), temp);
    top[0]->mutable_cpu_data()[0] = caffe_cpu_asum(bottom[0]->count(), temp_.cpu_data())/bottom[0]->num();
}

     

template <typename Dtype>
void KLLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
    //rescale bottom gradients
    caffe_cpu_scale(bottom[0]->count(), top[0]->cpu_diff()[0]/bottom[0]->num(), bottom[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
    caffe_cpu_scale(bottom[1]->count(), top[0]->cpu_diff()[0]/bottom[0]->num(), bottom[1]->cpu_diff(), bottom[1]->mutable_cpu_diff());
}




#ifdef CPU_ONLY
STUB_GPU(KLLossLayer);
#endif

INSTANTIATE_CLASS(KLLossLayer);
REGISTER_LAYER_CLASS(KLLoss);

}  // namespace caffe
