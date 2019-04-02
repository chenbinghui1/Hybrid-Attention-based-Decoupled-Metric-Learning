//需要设置solver里面iter_size=2 ，第一次前向反向传播为正常输入和正常梯度，并记录第一次正常输入，第二次前向为原输入。输入list需要在第一次和第二次的时候保持一样。
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cross_perturbation_input_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrossPerturbationInputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      CHECK_EQ(bottom.size(), 1);
      F_iter_size_ = 0;
}

template <typename Dtype>
void CrossPerturbationInputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      top[0]->ReshapeLike(*bottom[0]);
      temp_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void CrossPerturbationInputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //load cross weights
    //static int iter_size = 0;
    //forward propagation
    if(F_iter_size_ == 0)
    {
        caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
        caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), temp_.mutable_cpu_data());
        F_iter_size_ = 1;
    }
    else
    {

        caffe_copy(bottom[0]->count(), temp_.cpu_data(), top[0]->mutable_cpu_data());
        F_iter_size_ = 0;
    }
}

template <typename Dtype>
void CrossPerturbationInputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    caffe_copy(bottom[0]->count(), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(CrossPerturbationInputLayer);
#endif

INSTANTIATE_CLASS(CrossPerturbationInputLayer);
REGISTER_LAYER_CLASS(CrossPerturbationInput);

}  // namespace caffe
