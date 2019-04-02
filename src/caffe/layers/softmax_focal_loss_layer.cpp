#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithFocalLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  /*****************************************************
  ** initialize softmax layer
  ******************************************************/
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.clear_loss_weight();
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  alpha_ = this->layer_param().softmax_focal_loss_param().alpha();
  gamma_ = this->layer_param().softmax_focal_loss_param().gamma();
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void SoftmaxWithFocalLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  scaler_.ReshapeLike(*bottom[0]);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype SoftmaxWithFocalLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, Dtype valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = valid_count;
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxWithFocalLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  Dtype count = 0;
  Dtype loss = 0;
  Dtype* oriloss = scaler_.mutable_cpu_diff();
  Dtype* scale = scaler_.mutable_cpu_data();
  Dtype* loss_data = bottom[1]->mutable_cpu_diff();
  /*****************************************************
  ** record the scale and original loss repectively
  ******************************************************/
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        scale[i * inner_num_ + j] = 0;
        oriloss[i * inner_num_ + j] = 0;
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      Dtype prob_data_label = std::max(prob_data[i * dim + label_value * inner_num_ + j], Dtype(FLT_MIN)); 
      scale[i * inner_num_ + j] = alpha_ * powf(1 - prob_data_label, gamma_);
      oriloss[i * inner_num_ + j] = -log(prob_data_label);
      count += 1;
    }
  }
  /*****************************************************************
  ** compute alpha * (1 - p_y_i) ^ gamma * (original loss)
  ******************************************************************/
  caffe_mul(outer_num_ * inner_num_, scale, oriloss, loss_data);
  /****************************************************************
  ** compute total loss
  *****************************************************************/
  loss = caffe_cpu_asum(outer_num_ * inner_num_, loss_data);
  normalizer_ = get_normalizer(normalization_, count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithFocalLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    /***********************************************************
    ** First item: d(oriloss) * scale
    ************************************************************/
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    int dim = prob_.count() / outer_num_;
    const Dtype* scale = scaler_.cpu_data();
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
        for(int c = 0; c < bottom[0]->shape(softmax_axis_); ++c){
          bottom_diff[i * dim + c * inner_num_ + j] *= scale[i * inner_num_ + j];
        }
      }
    }
    /**************************************************************
    ** Second item: oriloss * d(scale)
    ** 1. d(scale) = d(scale)/d(p_yi) * d(p_yi)/d(xi)
    ** 2. d(scale)/d(p_yi) = -1 * aplha * gamma * (1 - p_yi) ^ (gamma - 1)
    ** 3. 1) d(p_yi)/d(x_i) = p_yi * (1 - p_yi) if i == yi
    ** 3. 2) d(p_yi)/d(x_i) = -p_yi * p_i       if i != yi
    ***************************************************************/
    Dtype* secondItem = scaler_.mutable_cpu_data();
    const Dtype* oriloss = scaler_.cpu_diff();
    for(int i = 0; i < outer_num_; ++i){
      for(int j = 0; j < inner_num_; ++j){
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        const Dtype prob_data_label = std::max(prob_data[i * dim + label_value * inner_num_ + j], Dtype(FLT_MIN));
        for(int c = 0; c < bottom[0]->shape(softmax_axis_); ++c){
          if(label_value == (c * inner_num_ + j)){
            secondItem[i * dim + c * inner_num_ + j] = -1 * alpha_ * gamma_ * powf(1 - prob_data_label, gamma_) * prob_data_label;
          }else{
            Dtype prob_data_c = std::max(prob_data[i * dim + c * inner_num_ + j], Dtype(FLT_MIN)); 
            secondItem[i * dim + c * inner_num_ + j] = alpha_ * gamma_ * powf(1 - prob_data_c, gamma_ - 1) * prob_data_label * prob_data_c;
          }
          secondItem[i * dim + c * inner_num_ + j] *= oriloss[i * inner_num_ + j];
        }
      }
    }
    /**************************************************************
    ** Calculate d(FL) = scale * d(oriloss) + oriloss * d(scale)
    ***************************************************************/
    caffe_add(bottom[0]->count(), scaler_.cpu_data(), bottom[0]->cpu_diff(), bottom_diff);
    /**************************************************************
    ** Manipulate ignore_label
    ***************************************************************/
    for(int i = 0; i < outer_num_; ++i){
      for(int j = 0; j < inner_num_; ++j){
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithFocalLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithFocalLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithFocalLoss);

}  // namespace caffe
