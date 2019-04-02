#ifndef CAFFE_UTH_LOSS_LAYER_HPP_
#define CAFFE_UTH_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"


namespace caffe{
template <typename Dtype>
class UTHLossLayer : public LossLayer<Dtype> {
 public:
  explicit UTHLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      
  virtual inline int ExactNumBottomBlobs() const { return 2; }//bottom[1] no use, bottom[0] index: anchor 0---(num/3-1), positive num/3-----(num*2/3-1), negative num*2/3------(num-1)
  virtual inline const char* type() const { return "UTHLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  /// @copydoc ContrastiveLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
      
      Blob<Dtype> ptemp_;
      Blob<Dtype> ntemp_;
};
}

#endif
