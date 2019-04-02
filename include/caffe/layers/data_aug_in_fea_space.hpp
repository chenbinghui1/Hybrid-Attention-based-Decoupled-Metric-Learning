#ifndef CAFFE_DATA_AUG_IN_FEA_SPACE_LAYER_HPP_
#define CAFFE_DATA_AUG_IN_FEA_SPACE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DataAuginFeaSpaceLayer : public Layer<Dtype> {
 public:
  explicit DataAuginFeaSpaceLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DataAuginFeaSpace"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }//bottom 0 is data ,bottom 1 is label.
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    //  const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    //  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
   Blob<Dtype> w_;
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
