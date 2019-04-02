// for 2d matrix multiplication, e.g. m x n matirx A times n x p matrix B
#include <algorithm>
#include <vector>

#include "caffe/layers/matrix2D_multiplication_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Matrix2DMultiplicationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->height() * bottom[0]->width(),1) << "bottom[0] Height and Width dimension must be 1";
    CHECK_EQ(bottom[1]->height() * bottom[1]->width(),1) << "bottom[1] Height and Width dimension must be 1";
    const bool transpose = this->layer_param_.matrix2d_multiplication_param().transpose();
    if(!transpose)
        CHECK_EQ(bottom[0]->channels(), bottom[1]->num()) << "The input two matrixs' dimension don't match";
    else
        CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) << "The input two matrixs' dimension don't match";
}

template <typename Dtype>
void Matrix2DMultiplicationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    bool transpose = this->layer_param_.matrix2d_multiplication_param().transpose();
    if(!transpose)
        top[0]->Reshape(bottom[0]->num(), bottom[1]->channels(), 1, 1);
    else
        top[0]->Reshape(bottom[0]->num(), bottom[1]->num(), 1, 1);
   }

template <typename Dtype>
void Matrix2DMultiplicationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
    bool transpose = this->layer_param_.matrix2d_multiplication_param().transpose();
    caffe_cpu_gemm(CblasNoTrans, transpose ? CblasTrans : CblasNoTrans, bottom[0]->num(), transpose ? bottom[1]->num() : bottom[1]->channels(), bottom[0]->channels(), (Dtype)1., bottom[0]->cpu_data(), bottom[1]->cpu_data(), (Dtype)0., top[0]->mutable_cpu_data());
}

template <typename Dtype>
void Matrix2DMultiplicationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bool transpose = this->layer_param_.matrix2d_multiplication_param().transpose();
  //for bottom[0]
  if(propagate_down[0]){
  caffe_cpu_gemm(CblasNoTrans, transpose ? CblasNoTrans : CblasTrans, bottom[0]->num(), bottom[0]->channels(), top[0]->channels(), (Dtype)1., top[0]->cpu_diff(), bottom[1]->cpu_data(), (Dtype)0., bottom[0]->mutable_cpu_diff());
  }
  //for bottom[1]
  if(propagate_down[1]){
  if(transpose)
      caffe_cpu_gemm(CblasTrans, CblasNoTrans, bottom[1]->num(), bottom[1]->channels(), top[0]->num(), (Dtype)1., top[0]->cpu_diff(), bottom[0]->cpu_data(), (Dtype)0., bottom[0]->mutable_cpu_diff());
  else
      caffe_cpu_gemm(CblasTrans, CblasNoTrans, bottom[1]->num(), bottom[1]->channels(), bottom[0]->num(), (Dtype)1., bottom[0]->cpu_data(), top[0]->cpu_diff(), (Dtype)0., bottom[1]->mutable_cpu_diff());
      }
}

#ifdef CPU_ONLY
STUB_GPU(Matrix2DMultiplicationLayer);
#endif

INSTANTIATE_CLASS(Matrix2DMultiplicationLayer);
REGISTER_LAYER_CLASS(Matrix2DMultiplication);

}  // namespace caffe
