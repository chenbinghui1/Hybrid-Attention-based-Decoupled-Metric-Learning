#ifndef CAFFE_ALIGNMENT_3ORD_LOSS_LAYER_HPP_
#define CAFFE_ALIGNMENT_3ORD_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe{
template <typename Dtype>
class Alignment3ordLossLayer : public LossLayer<Dtype> {
 public:
  explicit Alignment3ordLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      
  virtual inline int ExactNumBottomBlobs() const { return 4; }//bottom[0] [1], [2] [3] are source and target data,respectively. [0] [2] are mean vectors, [1] [3] are cov vectors. They must satisfy source classes equal target classes.
  virtual inline const char* type() const { return "Alignment3ordLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc ContrastiveLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
   //   const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
 // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
   //   const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};
}

#endif
