#ifndef CAFFE_HARDMAN_LOSS_LAYER_HPP_
#define CAFFE_HARDMAN_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe{
template <typename Dtype>
class HardmanLossLayer : public LossLayer<Dtype> {
 public:
  explicit HardmanLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "HardmanLoss"; }
  
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }

 protected:
  /// @copydoc ContrastiveLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
      
      Blob<Dtype> temp_;
      Blob<Dtype> diff_;
};
}

#endif
