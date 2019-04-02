//需要设置solver里面iter_size=2 ，第一次前向反向传播为正常输入和正常梯度，并记录反传回来的梯度，第二次前向为原输入+梯度干扰,输入list需要在第一次和第二次的时候保持一样。且根据ICLR18cross gradients 相应的loss也要做修改，此时只修改了BIER loss.
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/cross_perturbation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrossPerturbationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      CHECK_EQ(bottom.size(), 2);
      CHECK_EQ(bottom[0]->count(), bottom[1]->count());
      F_iter_size_ = 0;
      B_iter_size_ = 0;
}

template <typename Dtype>
void CrossPerturbationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      top[0]->ReshapeLike(*bottom[0]);
      top[1]->ReshapeLike(*bottom[1]);
      temp0_.ReshapeLike(*bottom[0]);
      temp1_.ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void CrossPerturbationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
                caffe_copy(bottom[i]->count(), bottom[i]->cpu_data(), top[i]->mutable_cpu_data());
        F_iter_size_ = 1;
    }
    else
    {//fi + ems[i]*gradient(fj)

        caffe_cpu_axpby(bottom[0]->count(), Dtype(1), bottom[0]->cpu_data(), Dtype(0), top[0]->mutable_cpu_data());
        caffe_cpu_axpby(bottom[1]->count(), Dtype(ems[0]), temp1_.cpu_data(), Dtype(1), top[0]->mutable_cpu_data());
        
        caffe_cpu_axpby(bottom[1]->count(), Dtype(1), bottom[1]->cpu_data(), Dtype(0), top[1]->mutable_cpu_data());
        caffe_cpu_axpby(bottom[0]->count(), Dtype(ems[1]), temp0_.cpu_data(), Dtype(1), top[1]->mutable_cpu_data());
        F_iter_size_ = 0;
    }
}

template <typename Dtype>
void CrossPerturbationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    //static int iter_size = 0;
    if(B_iter_size_ == 0)
    {
        for(int i = 0; i < bottom.size(); i++)
        {
                //propagate gradients
                caffe_copy(bottom[i]->count(), top[i]->cpu_diff(), bottom[i]->mutable_cpu_diff());
        }
        //store gradients
        caffe_copy(bottom[0]->count(), top[0]->cpu_diff(), temp0_.mutable_cpu_data());
        caffe_copy(bottom[1]->count(), top[1]->cpu_diff(), temp1_.mutable_cpu_data());
        B_iter_size_ = 1;
    }
    else
    {
        for(int i = 0; i < bottom.size(); i++)
        {
                //propagate gradients
                caffe_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_cpu_diff());
        }
        B_iter_size_ = 0;
    }
}

#ifdef CPU_ONLY
STUB_GPU(CrossPerturbationLayer);
#endif

INSTANTIATE_CLASS(CrossPerturbationLayer);
REGISTER_LAYER_CLASS(CrossPerturbation);

}  // namespace caffe
