#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}


template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  
  Dtype loss = 0.0;
  for(int i = 0; i < bottom[0]->num(); i++){
        const int label_value = static_cast<int>(label[i]);
        DCHECK_GE(label_value, 0);
        loss -= log(std::max(bottom_data[i * bottom[0]->channels() + label_value],
                           Dtype(FLT_MIN)));
  }
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    for(int i = 0; i < bottom[0]->num(); i++){
        const int label_value = static_cast<int>(label[i]);
        DCHECK_GE(label_value, 0);
        bottom_diff[i * bottom[0]->channels() + label_value]-= Dtype(1)/std::max(bottom[0]->cpu_data()[i * bottom[0]->channels() + label_value],Dtype(FLT_MIN));
    }
    caffe_scal(bottom[0]->count(), Dtype(1.0)/Dtype(bottom[0]->num()) * top[0]->cpu_diff()[0], bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(CrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(CrossEntropyLossLayer);
REGISTER_LAYER_CLASS(CrossEntropyLoss);

}  // namespace caffe
