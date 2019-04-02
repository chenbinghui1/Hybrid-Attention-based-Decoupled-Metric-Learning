#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "thrust/functional.h"
#include "thrust/sort.h"

namespace caffe {

  template <typename Dtype>
  __global__ void SoftmaxWithFocalLossForwardGPU(const int nthreads,
    Dtype* prob_data, const Dtype* label, Dtype* scale, Dtype* oriloss,
    const int dim, const int spatial_dim,
    const bool has_ignore_label_, const int ignore_label_,
    Dtype* counts, float alpha, float gamma) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int new_index = n * spatial_dim + s;
      const int label_value = static_cast<int>(label[new_index]);
      if ((has_ignore_label_ && label_value == ignore_label_)) {
        scale[new_index] = 0;
        oriloss[new_index] = 0;
        counts[new_index] = 0;
      }
      else {
        const Dtype prob_data_label = max(prob_data[n * dim + label_value * spatial_dim + s], Dtype(FLT_MIN));
        scale[new_index] = alpha * powf(1 - prob_data_label, gamma);
        oriloss[new_index] = -log(prob_data_label);
        counts[new_index] = 1;
      }
    }
  }

  template <typename Dtype>
  void SoftmaxWithFocalLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
    Dtype* prob_data = prob_.mutable_gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is not used for anything until it is overwritten
    // on the backward pass, we use it here to avoid having to allocate new GPU
    // memory to accumulate intermediate results in the kernel.
    Dtype* loss_data = bottom[0]->mutable_gpu_diff();
    Dtype* count_data = bottom[1]->mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxWithFocalLossForwardGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS >> >(nthreads, prob_data, label, scaler_.mutable_gpu_data(), scaler_.mutable_gpu_diff(),
          dim, inner_num_, has_ignore_label_, ignore_label_, count_data, alpha_, gamma_);
    caffe_gpu_mul(nthreads, scaler_.gpu_data(), scaler_.gpu_diff(), loss_data);
    Dtype loss;
    caffe_gpu_asum(nthreads, loss_data, &loss);
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID && has_ignore_label_) {
      caffe_gpu_asum(nthreads, count_data, &valid_count);
    }else{
        valid_count = nthreads;
    }
    normalizer_ = get_normalizer(normalization_, valid_count);
    top[0]->mutable_cpu_data()[0] = loss / normalizer_;
    if (top.size() == 2) {
      top[1]->ShareData(prob_);
    }
  }

  template <typename Dtype>
  __global__ void SoftmaxWithFocalLossBackwardFirstItemGPU(const int nthreads, const Dtype* scale,
                                         const Dtype* label, const int dim,
                                         const int spatial_dim, Dtype* firstItem) {
    const int channels = dim / spatial_dim;
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int new_index = n * spatial_dim + s;
      const int label_value = static_cast<int>(label[new_index]);
      firstItem[n * dim + label_value * spatial_dim + s] -= 1;
      for(int c = 0; c < channels; ++c){
          firstItem[n * dim + c * spatial_dim + s] *= scale[new_index];
      }
    }
  }

  template <typename Dtype>
  __global__ void SoftmaxWithFocalLossBackwardSecondItemGPU(const int nthreads,
                                         const Dtype* prob_data, const Dtype* oriloss,
                                         const Dtype* label, const int dim,
                                         const int spatial_dim, float alpha, float gamma, 
                                         Dtype* secondItem) {
    const int channels = dim / spatial_dim;
    CUDA_KERNEL_LOOP(index, nthreads) {
      const int n = index / spatial_dim;
      const int s = index % spatial_dim;
      const int new_index = n * spatial_dim + s;
      const int label_value = static_cast<int>(label[new_index]);
      const Dtype prob_data_label = max(prob_data[n * dim + label_value * spatial_dim + s], Dtype(FLT_MIN));
      for(int c = 0; c < channels; ++c){
        if(label_value == (c * spatial_dim + s)){
            secondItem[n * dim + c * spatial_dim + s] = -1 * alpha * gamma * powf(1 - prob_data_label, gamma) * prob_data_label;
        }else{
            Dtype prob_data_c = max(prob_data[n * dim + c * spatial_dim + s], Dtype(FLT_MIN)); 
            secondItem[n * dim + c * spatial_dim + s] = alpha * gamma * powf(1 - prob_data_c, gamma - 1) * prob_data_label * prob_data_c;
        }
        secondItem[n * dim + c * spatial_dim + s] *= oriloss[new_index];
      }
    }
  }

  template <typename Dtype>
  __global__ void SoftmaxWithFocalLossIgnoreDiffGPU(const int nthreads,
                const int ignore_label, const Dtype* label, const int dim, const int spatial_dim, 
                Dtype* diff) {
        const int channels = dim / spatial_dim;
        CUDA_KERNEL_LOOP(index, nthreads){
            const int n = index / spatial_dim;
            const int s = index % spatial_dim;
            const int new_index = n * spatial_dim + s;
            const int label_value = static_cast<int>(label[new_index]);
            if(label_value == ignore_label){
                for (int c = 0; c < channels; ++c) {
                    diff[n * dim + c * spatial_dim + s] = 0;
                }
            }
        }

  }

  template <typename Dtype>
  void SoftmaxWithFocalLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                 const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
      LOG(FATAL) << this->type()
        << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      const Dtype* prob_data = prob_.gpu_data();
      const Dtype* label = bottom[1]->gpu_data();
      // First Item Calculation
      caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
      const int dim = prob_.count() / outer_num_;
      const int nthreads = outer_num_ * inner_num_;
      // Since this memory is never used for anything else,
      // we use to to avoid allocating new GPU memory.
      SoftmaxWithFocalLossBackwardFirstItemGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS >> >(nthreads, scaler_.gpu_data(), label,
          dim, inner_num_, bottom_diff);
      // Second Item Calculation
      SoftmaxWithFocalLossBackwardSecondItemGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
          CAFFE_CUDA_NUM_THREADS >> >(nthreads, prob_data, scaler_.gpu_diff(), label,
          dim, inner_num_, alpha_, gamma_, scaler_.mutable_gpu_data());
      caffe_gpu_add(bottom[0]->count(), scaler_.gpu_data(), bottom[0]->gpu_diff(), bottom_diff);
      if(has_ignore_label_){
            SoftmaxWithFocalLossIgnoreDiffGPU<Dtype> << <CAFFE_GET_BLOCKS(nthreads),
                CAFFE_CUDA_NUM_THREADS >> >(nthreads, ignore_label_, label, dim, inner_num_, bottom_diff);
      }
      const Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
      caffe_gpu_scal(prob_.count(), loss_weight, bottom_diff);
    }
  }

  INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithFocalLossLayer);

}  // namespace caffe