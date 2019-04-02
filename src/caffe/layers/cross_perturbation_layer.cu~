#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cross_perturbation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrossPerturbationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //load cross weights
    //static int iter_size = 0;
    vector<Dtype> ems;
    for(int i = 0; i < this->layer_param_.cross_perturbation_param().ems_size(); i++)
    {
        ems.push_back(this->layer_param_.cross_perturbation_param().ems(i));
    }
    //forward propagation
    if(F_iter_size_ == 0)
    {
        for(int i = 0; i < bottom.size(); i++)
                caffe_copy(bottom[i]->count(), bottom[i]->gpu_data(), top[i]->mutable_gpu_data());
        F_iter_size_ = 1;
    }
    else
    {//fi + ems[i]*gradient(fj)

        caffe_gpu_axpby(bottom[0]->count(), Dtype(1), bottom[0]->gpu_data(), Dtype(0), top[0]->mutable_gpu_data());
        caffe_gpu_axpby(bottom[1]->count(), Dtype(ems[0]), temp1_.gpu_data(), Dtype(1), top[0]->mutable_gpu_data());
        
        caffe_gpu_axpby(bottom[1]->count(), Dtype(1), bottom[1]->gpu_data(), Dtype(0), top[1]->mutable_gpu_data());
        caffe_gpu_axpby(bottom[0]->count(), Dtype(ems[1]), temp0_.gpu_data(), Dtype(1), top[1]->mutable_gpu_data());
        F_iter_size_ = 0;
    }
}


template <typename Dtype>
void CrossPerturbationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    
    //static int iter_size = 0;
    if(B_iter_size_ == 0)
    {
        for(int i = 0; i < bottom.size(); i++)
        {
                //propagate gradients
                caffe_copy(bottom[i]->count(), top[i]->gpu_diff(), bottom[i]->mutable_gpu_diff());
        }
        //store gradients
        caffe_copy(bottom[0]->count(), top[0]->gpu_diff(), temp0_.mutable_gpu_data());
        caffe_copy(bottom[1]->count(), top[1]->gpu_diff(), temp1_.mutable_gpu_data());
        B_iter_size_ = 1;
    }
    else
    {
        for(int i = 0; i < bottom.size(); i++)
        {
                //propagate gradients
                caffe_gpu_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_gpu_diff());
        }
        B_iter_size_ = 0;
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(CrossPerturbationLayer);

}  // namespace caffe
