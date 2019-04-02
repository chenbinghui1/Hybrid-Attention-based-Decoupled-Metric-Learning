//unsupervised transformation stability loss
#include <algorithm>
#include <vector>
#include "stdio.h"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/transformation_stability_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>
/**********************/
/**/
namespace caffe {

template <typename Dtype>
void TransformationStabilityLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  for(int i=0; i<bottom.size(); i++){
        CHECK_EQ(bottom[i]->height(), 1);
        CHECK_EQ(bottom[i]->width(), 1);
        for(int j=i+1; j< bottom.size(); j++){
                 CHECK_EQ(bottom[i]->channels(), bottom[j]->channels());
                 CHECK_EQ(bottom[i]->num(), bottom[j]->num());
        }
  }
  temp_.Reshape(1,1,1,bottom[0]->channels());
}
template <typename Dtype>
void TransformationStabilityLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    int num = bottom[0]->num();//
    int channels = bottom[0]->channels();
    int M = bottom.size();
    Dtype* temp = temp_.mutable_cpu_data();
    
    Dtype loss = 0; 
    //compute loss
    for(int i = 0; i < num; i++)
    {
        for(int j = 0; j < M; j++)
        {
                for(int k = j+1; k < M; k++)
                {
                        caffe_copy(channels, bottom[j]->cpu_data() + i * channels, temp);
                        caffe_cpu_axpby(channels, Dtype(-1), bottom[k]->cpu_data() + i * channels, Dtype(1), temp);
                        loss = loss + caffe_cpu_dot(channels, temp, temp);
                        //compute gradients
                        caffe_cpu_axpby(channels, Dtype(1), temp, Dtype(1), bottom[j]->mutable_cpu_diff() + i * channels);
                        caffe_cpu_axpby(channels, Dtype(-1), temp, Dtype(1), bottom[k]->mutable_cpu_diff() + i * channels);
                }
        }
    }
    top[0]->mutable_cpu_data()[0] = loss * Dtype(0.5) / num;
}

     

template <typename Dtype>
void TransformationStabilityLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
    int M = bottom.size();
    for(int m = 0; m < M; m++)
    {
        caffe_cpu_scale(bottom[m]->count(), top[0]->cpu_diff()[0] /Dtype(bottom[m]->num()), bottom[m]->cpu_diff(), bottom[m]->mutable_cpu_diff());
    }
}




#ifdef CPU_ONLY
STUB_GPU(TransformationStabilityLossLayer);
#endif

INSTANTIATE_CLASS(TransformationStabilityLossLayer);
REGISTER_LAYER_CLASS(TransformationStabilityLoss);

}  // namespace caffe
