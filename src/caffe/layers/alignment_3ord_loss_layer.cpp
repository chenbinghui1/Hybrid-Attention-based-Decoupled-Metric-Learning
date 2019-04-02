//alignment 3ord loss layer, 整体实现kronecker layer + eulidean loss, 单独实现 可能超过blob最大size, Loss = sum_i ||Mean_s^i * Cov_s^i - Mean_t^i * Cov_t^i||, * ==>张量积.
#include <algorithm>
#include <vector>
#include "stdio.h"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/alignment_3ord_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>
/**********************/
/**/
namespace caffe {

template <typename Dtype>
void Alignment3ordLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[1]->channels(), bottom[3]->channels());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(bottom[1]->num(), bottom[3]->num());
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
}
template <typename Dtype>
void Alignment3ordLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
    top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void Alignment3ordLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* source_ave = bottom[0]->cpu_data();//ave vectors of each class for source data
    const Dtype* source_cov = bottom[1]->cpu_data();//cov vectors of each class for source data
    const Dtype* target_ave = bottom[2]->cpu_data();//ave vectors of each class for target data
    const Dtype* target_cov = bottom[3]->cpu_data();//cov vectors of each class for target data
    Dtype loss = 0;
    const int num = bottom[0]->num();//class number
    const int cov_channels = bottom[1]->channels();
    const int ave_channels = bottom[0]->channels();
    //compute loss, 将loss展开计算
    for(int i = 0; i < num; i++)
    {
        Dtype sa_sa = caffe_cpu_dot(ave_channels, source_ave + i * ave_channels, source_ave + i * ave_channels);
        Dtype sc_sc = caffe_cpu_dot(cov_channels, source_cov + i * cov_channels, source_cov + i * cov_channels);
        Dtype ta_ta = caffe_cpu_dot(ave_channels, target_ave + i * ave_channels, target_ave + i * ave_channels);
        Dtype tc_tc = caffe_cpu_dot(cov_channels, target_cov + i * cov_channels, target_cov + i * cov_channels);
        Dtype sa_ta = caffe_cpu_dot(ave_channels, source_ave + i * ave_channels, target_ave + i * ave_channels);
        Dtype sc_tc = caffe_cpu_dot(cov_channels, source_cov + i * cov_channels, target_cov + i * cov_channels);
        loss = loss + sa_sa * sc_sc + ta_ta * tc_tc - Dtype(2) * sa_ta * sc_tc;
        //compute gradients
        caffe_cpu_axpby(ave_channels, Dtype(2)*sc_sc, source_ave + i * ave_channels, Dtype(0), bottom[0]->mutable_cpu_diff() + i * ave_channels);
        caffe_cpu_axpby(ave_channels, Dtype(-2)*sc_tc, target_ave + i * ave_channels, Dtype(1), bottom[0]->mutable_cpu_diff() + i * ave_channels);
        
        caffe_cpu_axpby(cov_channels, Dtype(2)*sa_sa, source_cov + i * cov_channels, Dtype(0), bottom[1]->mutable_cpu_diff() + i * cov_channels);
        caffe_cpu_axpby(cov_channels, Dtype(-2)*sa_ta, target_cov + i * cov_channels, Dtype(1), bottom[1]->mutable_cpu_diff() + i * cov_channels);
        
        caffe_cpu_axpby(ave_channels, Dtype(2)*tc_tc, target_ave + i * ave_channels, Dtype(0), bottom[2]->mutable_cpu_diff() + i * ave_channels);
        caffe_cpu_axpby(ave_channels, Dtype(-2)*sc_tc, source_ave + i * ave_channels, Dtype(1), bottom[2]->mutable_cpu_diff() + i * ave_channels);
        
        caffe_cpu_axpby(cov_channels, Dtype(2)*ta_ta, target_cov + i * cov_channels, Dtype(0), bottom[3]->mutable_cpu_diff() + i * cov_channels);
        caffe_cpu_axpby(cov_channels, Dtype(-2)*sa_ta, source_cov + i * cov_channels, Dtype(1), bottom[3]->mutable_cpu_diff() + i * cov_channels);
    }
    top[0]->mutable_cpu_data()[0] = loss  * Dtype(0.5) / num;
}

     

template <typename Dtype>
void Alignment3ordLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
    //rescale bottom gradients
    caffe_cpu_scale(bottom[0]->count(), top[0]->cpu_diff()[0]/(Dtype(2)*bottom[0]->num()), bottom[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
    caffe_cpu_scale(bottom[1]->count(), top[0]->cpu_diff()[0]/(Dtype(2)*bottom[0]->num()), bottom[1]->cpu_diff(), bottom[1]->mutable_cpu_diff());
    caffe_cpu_scale(bottom[2]->count(), top[0]->cpu_diff()[0]/(Dtype(2)*bottom[0]->num()), bottom[2]->cpu_diff(), bottom[2]->mutable_cpu_diff());
    caffe_cpu_scale(bottom[3]->count(), top[0]->cpu_diff()[0]/(Dtype(2)*bottom[0]->num()), bottom[3]->cpu_diff(), bottom[3]->mutable_cpu_diff());
}




#ifdef CPU_ONLY
STUB_GPU(Alignment3ordLossLayer);
#endif

INSTANTIATE_CLASS(Alignment3ordLossLayer);
REGISTER_LAYER_CLASS(Alignment3ordLoss);

}  // namespace caffe
