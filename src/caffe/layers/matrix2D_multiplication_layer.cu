#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/matrix2D_multiplication_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Matrix2DMultiplicationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      bool transpose = this->layer_param_.matrix2d_multiplication_param().transpose();
    caffe_gpu_gemm(CblasNoTrans, transpose ? CblasTrans : CblasNoTrans, bottom[0]->num(), transpose ? bottom[1]->num() : bottom[1]->channels(), bottom[0]->channels(), (Dtype)1., bottom[0]->gpu_data(), bottom[1]->gpu_data(), (Dtype)0., top[0]->mutable_gpu_data());
}
template <typename Dtype>
void Matrix2DMultiplicationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  bool transpose = this->layer_param_.matrix2d_multiplication_param().transpose();
  //for bottom[0]
  if(propagate_down[0]){
  caffe_gpu_gemm(CblasNoTrans, transpose ? CblasNoTrans : CblasTrans, bottom[0]->num(), bottom[0]->channels(), top[0]->channels(), (Dtype)1., top[0]->gpu_diff(), bottom[1]->gpu_data(), (Dtype)0., bottom[0]->mutable_gpu_diff());
  }
  //for bottom[1]
  if(propagate_down[1]){
  if(transpose)
      caffe_gpu_gemm(CblasTrans, CblasNoTrans, bottom[1]->num(), bottom[1]->channels(), top[0]->num(), (Dtype)1., top[0]->gpu_diff(), bottom[0]->gpu_data(), (Dtype)0., bottom[0]->mutable_gpu_diff());
  else
      caffe_gpu_gemm(CblasTrans, CblasNoTrans, bottom[1]->num(), bottom[1]->channels(), bottom[0]->num(), (Dtype)1., bottom[0]->gpu_data(), top[0]->gpu_diff(), (Dtype)0., bottom[1]->mutable_gpu_diff());
      }
}

INSTANTIATE_LAYER_GPU_FUNCS(Matrix2DMultiplicationLayer);

}  // namespace caffe
