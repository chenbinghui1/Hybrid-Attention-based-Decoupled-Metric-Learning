#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cross_perturbation_input_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrossPerturbationInputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //static int iter_size = 0;
    //forward propagation
    if(F_iter_size_ == 0)
    {
        caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
        caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), temp_.mutable_gpu_data());
        F_iter_size_ = 1;
    }
    else
    {//fi + ems[i]*gradient(fj)

        caffe_copy(bottom[0]->count(), temp_.gpu_data(), top[0]->mutable_gpu_data());
        F_iter_size_ = 0;
    }
}


template <typename Dtype>
void CrossPerturbationInputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

                caffe_copy(bottom[0]->count(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());

}

INSTANTIATE_LAYER_GPU_FUNCS(CrossPerturbationInputLayer);

}  // namespace caffe
