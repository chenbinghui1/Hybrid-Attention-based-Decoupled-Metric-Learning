//unsupervised hashing loss
#include <algorithm>
#include <vector>
#include "stdio.h"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/uth_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>
/**********************/
/**/
namespace caffe {

template <typename Dtype>
void UTHLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[0]->num()%3, 0);
  ptemp_.Reshape(1,1,1,bottom[0]->channels());
  ntemp_.Reshape(1,1,1,bottom[0]->channels());
}
template <typename Dtype>
void UTHLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* data = bottom[0]->cpu_data();
    
    int num = bottom[0]->num();//
    int channels = bottom[0]->channels();

    Dtype margin = this->layer_param_.uth_loss_param().margin();
    
    Dtype loss = 0; 
    //compute loss
    for(int i = 0; i < num/3; i++)
    {
        caffe_copy(channels, data + i * channels, ptemp_.mutable_cpu_data());
        caffe_cpu_axpby(channels, Dtype(-1), data + (i + num/3) * channels, Dtype(1), ptemp_.mutable_cpu_data());
        
        caffe_copy(channels, data + i * channels, ntemp_.mutable_cpu_data());
        caffe_cpu_axpby(channels, Dtype(-1), data + (i + num*2/3) * channels, Dtype(1), ntemp_.mutable_cpu_data());
        
        Dtype loss1 = std::max(Dtype(0), margin + caffe_cpu_dot(channels,ptemp_.cpu_data(),ptemp_.cpu_data()) - caffe_cpu_dot(channels, ntemp_.cpu_data(), ntemp_.cpu_data()));
        loss = loss + loss1;
        //compute gradients
        if(loss1>Dtype(0))
        {
                caffe_cpu_axpby(channels, Dtype(2), ptemp_.cpu_data(), Dtype(1), bottom[0]->mutable_cpu_diff() + i * channels);//for anchor point
                caffe_cpu_axpby(channels, Dtype(-2), ntemp_.cpu_data(), Dtype(1), bottom[0]->mutable_cpu_diff() + i * channels);
                caffe_cpu_axpby(channels, Dtype(-2), ptemp_.cpu_data(), Dtype(1), bottom[0]->mutable_cpu_diff() + (i + num/3) * channels);//for positive point
                caffe_cpu_axpby(channels, Dtype(2), ntemp_.cpu_data(), Dtype(1), bottom[0]->mutable_cpu_diff() + (i + 2*num/3) * channels);//for negative point
        }
    }
    top[0]->mutable_cpu_data()[0] = loss * Dtype(3) / num;
}

     

template <typename Dtype>
void UTHLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
    caffe_cpu_scale(bottom[0]->count(), top[0]->cpu_diff()[0] * Dtype(3.0/bottom[0]->num()), bottom[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
}




#ifdef CPU_ONLY
STUB_GPU(UTHLossLayer);
#endif

INSTANTIATE_CLASS(UTHLossLayer);
REGISTER_LAYER_CLASS(UTHLoss);

}  // namespace caffe
