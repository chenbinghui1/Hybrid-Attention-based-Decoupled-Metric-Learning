// GCN CVPR488
/*
#include <vector>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <iostream>  // NOLINT(readability/streams)

#include "caffe/filler.hpp"
#include "caffe/layers/GCN_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GCNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  word_num_ = this->layer_param_.gcn_param().word_num();
  word_dim_ = this->layer_param_.gcn_param().word_dim();
}

template <typename Dtype>
void GCNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "Input size does not match.";
  vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape);
  embedding_.Reshape(1, 1, word_num_, word_dim_);
  word_temp_.Reshape(1,1,1,word_dim_);
  A_.Reshape(1,1,bottom[0]->num(),bottom[0]->num());
  D_.Reshape(1,1,bottom[0]->num(),bottom[0]->num());
  A_temp_.Reshape(1,1,bottom[0]->num(),bottom[0]->num());
  
  const string& file = this->layer_param_.gcn_param().file();
  std::ifstream infile(file.c_str());
  //initialize word embedding
  for(int i = 0; i < word_num_; i++)
  {
        for(int j = 0; j < word_dim_; j++)
        {
                infile >> embedding_.mutable_cpu_data()[i*word_dim_+j];
               // LOG(INFO)<<embedding_.mutable_cpu_data()[i*word_dim_+j];
        }
  }
  infile.close();
}

template <typename Dtype>
void GCNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* A = A_.mutable_cpu_data();
    Dtype* D = D_.mutable_cpu_data();
    Dtype* word_temp = word_temp_.mutable_cpu_data();
    Dtype* embedding = embedding_.mutable_cpu_data();
    Dtype t = this->layer_param_.gcn_param().t();
    
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    //compute adjacency matrix A
    for(int i = 0; i < num; i++)
    {
        for(int j = i; j < num; j++)
        {
                if(i==j)
                {
                        A[i*num+j]=Dtype(1);
                }
                else
                {
                        caffe_copy(word_dim_, embedding + static_cast<int>(label[i]) * word_dim_, word_temp);
                        caffe_cpu_axpby(word_dim_, Dtype(-1), embedding + static_cast<int>(label[j]) * word_dim_, Dtype(1), word_temp);
                        A[i*num+j] = std::exp(Dtype(-1)*caffe_cpu_dot(word_dim_, word_temp, word_temp)/t);
                        A[j*num+i] = A[i*num+j];
                }
        }
    }
    //compute matrix D
    caffe_set(D_.count(), Dtype(0), D);
    for(int i = 0; i < num; i++)
    {
        D[i*num+i] = Dtype(1)/std::sqrt(caffe_cpu_asum(num, A + i * num));
    }
    //compute Y=DADX
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, num, num, Dtype(1), D, A, Dtype(0), A_temp_.mutable_cpu_data());//DA
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, num, num, Dtype(1), A_temp_.mutable_cpu_data(), D, Dtype(0), A);//DAD
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, channels, num, Dtype(1), A, data, Dtype(0), top_data);//DADX
}

template <typename Dtype>
void GCNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    //compute bottom gradients
    if (propagate_down[0]) {
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bottom[0]->num(), bottom[0]->channels(), bottom[0]->num(), Dtype(1), A_.cpu_data(), top[0]->cpu_diff(), Dtype(0), bottom[0]->mutable_cpu_diff());
    }
}

#ifdef CPU_ONLY
STUB_GPU(GCNLayer);
#endif

INSTANTIATE_CLASS(GCNLayer);
REGISTER_LAYER_CLASS(GCN);

} */

//读取 similarity between classes, i.e. A
#include <vector>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <iostream>  // NOLINT(readability/streams)

#include "caffe/filler.hpp"
#include "caffe/layers/GCN_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GCNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  word_num_ = this->layer_param_.gcn_param().word_num();
  word_dim_ = this->layer_param_.gcn_param().word_dim();
}

template <typename Dtype>
void GCNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "Input size does not match.";
  vector<int> top_shape = bottom[0]->shape();
  top[0]->Reshape(top_shape);
  embedding_.Reshape(1, 1, word_num_, word_dim_);
  A_.Reshape(1,1,bottom[0]->num(),bottom[0]->num());
  D_.Reshape(1,1,bottom[0]->num(),bottom[0]->num());
  A_temp_.Reshape(1,1,bottom[0]->num(),bottom[0]->num());
  
  const string& file = this->layer_param_.gcn_param().file();
  std::ifstream infile(file.c_str());
  //initialize word embedding
  for(int i = 0; i < word_num_ * word_dim_; i++)
  {
                infile >> embedding_.mutable_cpu_data()[i];
               // LOG(INFO)<<embedding_.mutable_cpu_data()[i*word_dim_+j];
  }
  infile.close();
}

template <typename Dtype>
void GCNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* A = A_.mutable_cpu_data();
    Dtype* D = D_.mutable_cpu_data();
    Dtype* embedding = embedding_.mutable_cpu_data();
    
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    //compute adjacency matrix A
    for(int i = 0; i < num; i++)
    {
        for(int j = i; j < num; j++)
        {
                if(i==j)
                {
                        A[i*num+j]=Dtype(1);
                }
                else
                {
                        A[i*num+j] = embedding[static_cast<int>(label[i])*word_num_ + static_cast<int>(label[j])];
                        A[j*num+i] = A[i*num+j];
                }
        }
    }
    //compute matrix D
    caffe_set(D_.count(), Dtype(0), D);
    for(int i = 0; i < num; i++)
    {
        D[i*num+i] = Dtype(1)/std::sqrt(caffe_cpu_asum(num, A + i * num));
    }
    //compute Y=DADX
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, num, num, Dtype(1), D, A, Dtype(0), A_temp_.mutable_cpu_data());//DA
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, num, num, Dtype(1), A_temp_.mutable_cpu_data(), D, Dtype(0), A);//DAD
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, num, channels, num, Dtype(1), A, data, Dtype(0), top_data);//DADX
}

template <typename Dtype>
void GCNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    //compute bottom gradients
    if (propagate_down[0]) {
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bottom[0]->num(), bottom[0]->channels(), bottom[0]->num(), Dtype(1), A_.cpu_data(), top[0]->cpu_diff(), Dtype(0), bottom[0]->mutable_cpu_diff());
    }
}

#ifdef CPU_ONLY
STUB_GPU(GCNLayer);
#endif

INSTANTIATE_CLASS(GCNLayer);
REGISTER_LAYER_CLASS(GCN);

} 
