#ifndef CAFFE_ALIGNMENT_2ORD_LOSS_LAYER_HPP_
#define CAFFE_ALIGNMENT_2ORD_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe{
template <typename Dtype>
class Alignment2ordLossLayer : public LossLayer<Dtype> {
 public:
  explicit Alignment2ordLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      
  virtual inline int ExactNumBottomBlobs() const { return 4; }//bottom[0] [1] are source and target data,respectively. And the data from the same class must be adjacent. Target data should be N*3. [2]  [3] are label. They must satisfy source classes equal target classes.
  virtual inline const char* type() const { return "Alignment2ordLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  /// @copydoc ContrastiveLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  Blob<Dtype> temp_source_;//IX
  Blob<Dtype> temp_target_;//Ix
  Blob<Dtype> temp_;//XIX_s-XIX_t
  Blob<Dtype> one_source_;//I
  Blob<Dtype> one_target_;//I
};
}

#endif
