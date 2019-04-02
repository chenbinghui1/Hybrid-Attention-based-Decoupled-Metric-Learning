#include <vector> //类中心是softmax层的参数 w  配合inner_produc_loss层用  超参数在loss层指定loss_weight 可能会出现nan现象直接训练（可以先用lossweight0 初始化等lmcenterloss较小时候在用正常的lossweight）

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_for_lmcenter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
template <typename Dtype>
void InnerProductForLMCenterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_for_lmcenter_param().num_output();
  bias_term_ = this->layer_param_.inner_product_for_lmcenter_param().bias_term();
  transpose_ = this->layer_param_.inner_product_for_lmcenter_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_for_lmcenter_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_for_lmcenter_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_for_lmcenter_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductForLMCenterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_for_lmcenter_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  top[1]->Reshape(1,1,1,1);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
  w_norm_.Reshape(M_,1,1,1);
  x_norm_.Reshape(M_,1,1,1);
  diff_temp_.Reshape(M_,K_,1,1);
  memory_.Reshape(K_,K_,1,1);
  memory1_.Reshape(M_,K_,1,1);
  sqr_bottom_.Reshape(M_,K_,1,1);
}

template <typename Dtype>
void InnerProductForLMCenterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
  
  
  const Dtype* label = bottom[1]->cpu_data();
  Dtype loss1 = 0;
  for(int i = 0; i < M_; i++) {
  //compute |x|
     x_norm_.mutable_cpu_data()[i] = caffe_cpu_dot<Dtype>(K_, bottom_data + i * K_, bottom_data + i * K_);
     x_norm_.mutable_cpu_data()[i] = static_cast<Dtype>(std::sqrt(x_norm_.mutable_cpu_data()[i]));   
     
  //compute |w|
     w_norm_.mutable_cpu_data()[i] = caffe_cpu_dot<Dtype>(K_, weight + static_cast<int>(label[i]) * K_, weight + static_cast<int>(label[i]) * K_);
     w_norm_.mutable_cpu_data()[i] = static_cast<Dtype>(std::sqrt(w_norm_.mutable_cpu_data()[i]));
     
  //compute loss1 for top[1]
     caffe_cpu_axpby<Dtype>(K_, x_norm_.mutable_cpu_data()[i] / w_norm_.mutable_cpu_data()[i], weight + static_cast<int>(label[i]) * K_, Dtype(0), diff_temp_.mutable_cpu_data() + i * K_);
     caffe_sub(K_, bottom_data + i * K_, diff_temp_.mutable_cpu_data() + i * K_, diff_temp_.mutable_cpu_data() + i * K_);
     loss1 += caffe_cpu_dot(K_, diff_temp_.cpu_data() + i * K_, diff_temp_.cpu_data() + i * K_);
  }
  top[1]->mutable_cpu_data()[0] = loss1/M_/Dtype(2);
}

template <typename Dtype>
void InnerProductForLMCenterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    
    const Dtype* w_norm = w_norm_.cpu_data();
    const Dtype* x_norm = x_norm_.cpu_data();
    
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
    //come from top[1]->cpu_diff()
    Dtype* memory = memory_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
     for(int i = 0; i < M_; i++){
       caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,K_,K_,1,Dtype(1)/pow(w_norm[i],2), weight + static_cast<int>(label[i]) * K_, weight + static_cast<int>(label[i]) * K_, Dtype(0), memory);
       caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_, K_, Dtype(1), diff_temp_.cpu_data() + i * K_, memory, Dtype(0), memory1_.mutable_cpu_data() + i * K_);
       caffe_sub<Dtype>(K_, memory1_.mutable_cpu_data() + i * K_, diff_temp_.cpu_data() + i * K_, memory1_.mutable_cpu_diff() + i * K_);
       caffe_cpu_axpby<Dtype>(K_, top[1]->cpu_diff()[0]*x_norm[i]/(M_*w_norm[i]), memory1_.mutable_cpu_diff() + i * K_, Dtype(1), this->blobs_[0]->mutable_cpu_diff() + static_cast<int>(label[i]) * K_);
     }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
    // come from top[1]->cpu_diff()
     Dtype* memory = memory_.mutable_cpu_data();
     const Dtype* weight = this->blobs_[0]->cpu_data();
     const Dtype* label = bottom[1]->cpu_data();
     for(int i = 0; i < M_; i++){
       caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans,K_,K_,1,Dtype(1)/(x_norm[i]*w_norm[i]), weight + static_cast<int>(label[i]) * K_, bottom[0]->cpu_data() + i * K_, Dtype(0), memory);
       caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_, K_, Dtype(1), diff_temp_.cpu_data() + i * K_, memory, Dtype(0), memory1_.mutable_cpu_data() + i * K_);
       caffe_sub<Dtype>(K_, diff_temp_.cpu_data() + i * K_, memory1_.mutable_cpu_data() + i * K_, memory1_.mutable_cpu_diff() + i * K_);
       caffe_cpu_axpby<Dtype>(K_, top[1]->cpu_diff()[0]/M_, memory1_.mutable_cpu_diff() + i * K_, Dtype(1), bottom[0]->mutable_cpu_diff() + i * K_);
     }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InnerProductForLMCenterLayer);
#endif

INSTANTIATE_CLASS(InnerProductForLMCenterLayer);
REGISTER_LAYER_CLASS(InnerProductForLMCenter);
}
