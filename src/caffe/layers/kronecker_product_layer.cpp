#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/kronecker_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KroneckerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) << "bottom input must have the same dimision";
}

template <typename Dtype>
void KroneckerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels() * bottom[1]->channels(),1,1);
  //top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->channels(), bottom[1]->channels());
}

template <typename Dtype>
void KroneckerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void KroneckerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(KroneckerProductLayer);
#endif

INSTANTIATE_CLASS(KroneckerProductLayer);
REGISTER_LAYER_CLASS(KroneckerProduct);

}  // namespace caffe
