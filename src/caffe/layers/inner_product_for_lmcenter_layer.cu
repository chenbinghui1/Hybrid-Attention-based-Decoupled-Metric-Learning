#include <vector> //类中心是softmax层的参数 w  配合inner_produc_loss层用  超参数在loss层指定loss_weight

#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_for_lmcenter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{


template <typename Dtype>
__global__ void compute_loss(int nthreads, const int K, const Dtype* x_norm,
	      const Dtype* w_norm ,const Dtype* weight, const Dtype* bottom_data, const Dtype* label, Dtype* diff_temp) {
	 
  CUDA_KERNEL_LOOP(index, nthreads) {
    int i = index / K;
    int j = index % K;
    const int label_value = static_cast<int>(label[i]);
    diff_temp[i*K+j] = bottom_data[i*K+j] - weight[label_value*K+j] * x_norm[i] / w_norm[i];
  }
}


template <typename Dtype>
void InnerProductForLMCenterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //  Forward_cpu(bottom,top);
 //   return;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
  // compute ||x|| num=M_
  caffe_gpu_powx(bottom[0]->count(), bottom_data, Dtype(2), sqr_bottom_.mutable_gpu_data());
  for (int i = 0; i < M_; i++) {
    Dtype a;
    caffe_gpu_asum<Dtype>(K_, sqr_bottom_.gpu_data() + i * K_, &a);
    caffe_gpu_set<Dtype>(1, std::sqrt(a), x_norm_.mutable_gpu_data() + i);
  }
  // compute ||w|| num=M_
  for (int i = 0; i < M_; i++) {
    Dtype a;
    caffe_gpu_dot<Dtype>(K_, weight + static_cast<int>(bottom[1]->cpu_data()[i]) * K_, weight + static_cast<int>(bottom[1]->cpu_data()[i]) * K_, &a);
    caffe_gpu_set<Dtype>(1,std::sqrt(a),w_norm_.mutable_gpu_data() + i);
  }
  ////compute loss1 for top[1]
  int nthreads = M_*K_;
  compute_loss<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, x_norm_.gpu_data(), w_norm_.gpu_data(), weight, bottom_data, bottom[1]->gpu_data(), diff_temp_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(M_ * K_, diff_temp_.gpu_data(), diff_temp_.gpu_data(), &dot);
  top[1]->mutable_cpu_data()[0] = Dtype(dot/M_/Dtype(2));
}


template <typename Dtype>
__global__ void scale(int nthreads, Dtype* memory, const Dtype* w_norm, const Dtype* w_norm1) {   
  CUDA_KERNEL_LOOP(index, nthreads) {
    memory[index] = memory[index]/(w_norm[0] * w_norm1[0]);
  }
}
template <typename Dtype>
__global__ void scale1(int nthreads, int M, Dtype* memory1, const Dtype* w_norm, const Dtype* x_norm, const Dtype* top1_diff) {   
  CUDA_KERNEL_LOOP(index, nthreads) {
    memory1[index] = memory1[index] * x_norm[0] * top1_diff[0]/ w_norm[0] / M;
  }
}

template <typename Dtype>
void InnerProductForLMCenterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
//Backward_cpu(top,propagate_down,bottom);
//return;
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
    Dtype* memory = memory_.mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int nthreads;
    for (int i = 0; i < M_; i++) {
    caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,K_,K_,1,Dtype(1), weight + static_cast<int>(label[i]) * K_, weight + static_cast<int>(label[i]) * K_, Dtype(0), memory);
    nthreads = K_*K_;
    scale<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, memory, w_norm_.gpu_data() + i, w_norm_.gpu_data() + i);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_, K_, Dtype(1), diff_temp_.gpu_data() + i * K_, memory, Dtype(0), memory1_.mutable_gpu_data() + i * K_);
    caffe_gpu_sub<Dtype>(K_, memory1_.gpu_data() + i * K_, diff_temp_.gpu_data() + i * K_, memory1_.mutable_gpu_data() + i * K_);
    nthreads = K_;
    scale1<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, memory1_.mutable_gpu_data() + i * K_, w_norm_.gpu_data() + i, x_norm_.gpu_data() + i, top[1]->gpu_diff());
    caffe_gpu_axpby<Dtype>(K_, Dtype(1), memory1_.mutable_gpu_data() + i * K_, Dtype(1), this->blobs_[0]->mutable_gpu_diff() + static_cast<int>(label[i]) * K_);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
    Dtype* memory = memory_.mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int nthreads;
    for (int i = 0; i < M_; i++) {
    caffe_gpu_gemm<Dtype>(CblasTrans,CblasNoTrans,K_,K_,1,Dtype(1), weight + static_cast<int>(label[i]) * K_, bottom[0]->gpu_data() + i * K_, Dtype(0), memory);
    nthreads = K_*K_;
    scale<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, memory, w_norm_.gpu_data() + i, x_norm_.gpu_data() + i);
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, K_, K_, Dtype(1), diff_temp_.gpu_data() + i * K_, memory_.gpu_data(), Dtype(0), memory1_.mutable_gpu_data() + i * K_);
    caffe_gpu_sub<Dtype>(K_, diff_temp_.gpu_data() + i * K_, memory1_.gpu_data() + i * K_, memory1_.mutable_gpu_data() + i * K_);
    caffe_gpu_axpby<Dtype>(K_, top[1]->cpu_diff()[0]/M_, memory1_.gpu_data() + i * K_, Dtype(1), bottom[0]->mutable_gpu_diff() + i * K_);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductForLMCenterLayer);
}
