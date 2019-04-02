//alignment loss layer for mean feature, loss_weight/C*sum(weight_c*||U_c^s-U_c^t||^2) + loss_weight*alpha*sum(||weight_c-1||^2),   loss_weight is not alignment_loss_parameter's loss_weight
#include <algorithm>
#include <vector>
#include "stdio.h"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/alignment_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>
/**********************/
/**/
namespace caffe {

template <typename Dtype>
void AlignmentLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
      this->blobs_.resize(1);
    // Initialize the weights
    vector<int> weight_shape(2);
    weight_shape[0]=this->layer_param_.alignment_loss_param().num();
    weight_shape[1]=1;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.alignment_loss_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);//should be initialized with constant 1;
  temp_.Reshape(1,1,1,bottom[0]->channels());
}
template <typename Dtype>
void AlignmentLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
    top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void AlignmentLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* s_data = bottom[0]->cpu_data();//source data
    const Dtype* t_data = bottom[1]->cpu_data();//target data
    const Dtype* s_label = bottom[2]->cpu_data();
    const Dtype* t_label = bottom[3]->cpu_data();
    Dtype* weight = this->blobs_[0]->mutable_cpu_data();
    
    int s_num = bottom[0]->num();//source number
    int t_num = bottom[1]->num();//target number
    int channels = bottom[0]->channels();
    Dtype alpha = this->layer_param_.alignment_loss_param().alpha();
    Dtype loss_weight = this->layer_param_.alignment_loss_param().loss_weight();//copy loss_weight value
    
    Dtype loss = 0; 
    //compute loss
    vector<int> flag;//store label
    vector<Dtype> weight_temp;//store weight gradients
    caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
    caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
    for(int i = 0; i < s_num; i++)
    {
        vector<int>::iterator result = find(flag.begin(), flag.end(), static_cast<int>(s_label[i]));
        if(result != flag.end())
        {
                continue;
        }
        else
        {
                flag.push_back(static_cast<int>(s_label[i]));// new class
                caffe_set(channels, Dtype(0), temp_.mutable_cpu_data());
                vector<int> s_index, t_index;
                //find all data belonging to this new class
                for(int j = 0; j < s_num; j++)
                {
                        if(static_cast<int>(s_label[j])==static_cast<int>(s_label[i]))
                                s_index.push_back(j);
                }
                for(int j = 0; j < t_num; j++)
                {
                        if(static_cast<int>(t_label[j])==static_cast<int>(s_label[i]))
                                t_index.push_back(j);
                }
                //compute domain mean w.r.t. class
                for(int j = 0; j < s_index.size(); j++)
                {
                        caffe_cpu_axpby(channels, Dtype(1.0/s_index.size()), s_data + s_index[j] * channels, Dtype(1), temp_.mutable_cpu_data());
                }
                for(int j = 0; j < t_index.size(); j++)
                {
                        caffe_cpu_axpby(channels, Dtype(-1.0/t_index.size()), t_data + t_index[j] * channels, Dtype(1), temp_.mutable_cpu_data());
                }
                Dtype loss_temp = caffe_cpu_dot(channels, temp_.cpu_data(), temp_.cpu_data());
                loss = loss + weight[static_cast<int>(s_label[i])] * loss_temp;
                //compute bottom gradients
                for(int j = 0; j < s_index.size(); j++)
                {
                        caffe_cpu_axpby(channels, Dtype(2.0*weight[static_cast<int>(s_label[i])]/s_index.size()), temp_.cpu_data(), Dtype(1), bottom[0]->mutable_cpu_diff() + s_index[j] * channels);
                }
                for(int j = 0; j < t_index.size(); j++)
                {
                        caffe_cpu_axpby(channels, Dtype(-2.0*weight[static_cast<int>(s_label[i])]/t_index.size()), temp_.cpu_data(), Dtype(1), bottom[1]->mutable_cpu_diff() + t_index[j] * channels);
                }
                //store weight gradients
                weight_temp.push_back(loss_weight * loss_temp);
        }
    }
    top[0]->mutable_cpu_data()[0] = loss/flag.size();
    //compute weighted param constraint loss
    for(int i = 0; i < flag.size(); i++)
    {
        top[0]->mutable_cpu_data()[0] = top[0]->mutable_cpu_data()[0] + alpha * (weight[flag[i]]-Dtype(1)) * (weight[flag[i]]-Dtype(1));
    }
    //rescale bottom gradients
    caffe_cpu_scale(bottom[0]->count(), Dtype(1.0/flag.size()), bottom[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
    caffe_cpu_scale(bottom[1]->count(), Dtype(1.0/flag.size()), bottom[1]->cpu_diff(), bottom[1]->mutable_cpu_diff());
    //compute final weight gradients
    for(int i = 0; i < flag.size(); i++)
    {
         this->blobs_[0]->mutable_cpu_diff()[flag[i]] = this->blobs_[0]->mutable_cpu_diff()[flag[i]] + weight_temp[i]/flag.size() + Dtype(2)*alpha*loss_weight*(weight[flag[i]]-Dtype(1));
    }
}

     

template <typename Dtype>
void AlignmentLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
    //rescale bottom gradients
    caffe_cpu_scale(bottom[0]->count(), top[0]->cpu_diff()[0], bottom[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
    caffe_cpu_scale(bottom[1]->count(), top[0]->cpu_diff()[0], bottom[1]->cpu_diff(), bottom[1]->mutable_cpu_diff());
}




#ifdef CPU_ONLY
STUB_GPU(AlignmentLossLayer);
#endif

INSTANTIATE_CLASS(AlignmentLossLayer);
REGISTER_LAYER_CLASS(AlignmentLoss);

}  // namespace caffe
