#ifndef CAFFE_BOOSTING_DPH_LOSS_LAYER_HPP_
#define CAFFE_BOOSTING_DPH_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe{
template <typename Dtype>
class BoostingDPHLossLayer : public LossLayer<Dtype> {
 public:
  explicit BoostingDPHLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "BoostingDPHLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /// @copydoc ContrastiveLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  vector<vector<Dtype> > center_temp_;//vector<Blob<Dtype>*>不能用(3)来初始化,因为Blob没有构造函数
  Blob<Dtype> one_;
};
}

#endif
