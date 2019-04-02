#include <vector>
#include <algorithm>

#include "caffe/filler.hpp"
#include "caffe/layers/gradients_reverse_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GradientsReverseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      progress_ = 0;
}

template <typename Dtype>
void GradientsReverseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GradientsReverseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    progress_++;
    progress_ = std::min(progress_, static_cast<long int>(100000));
    caffe_copy(bottom[0]->count(),bottom[0]->cpu_data(),top[0]->mutable_cpu_data());
}

template <typename Dtype>
void GradientsReverseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    Dtype lamda = Dtype(2)/(Dtype(1) + std::exp(Dtype(-1) * this->layer_param_.gradient_reverse_param().gama() * (Dtype(progress_)/Dtype(this->layer_param_.gradient_reverse_param().iter_size())/Dtype(this->layer_param_.gradient_reverse_param().max_iter())))) - Dtype(1);
    caffe_cpu_scale(bottom[0]->count(),Dtype(-1)*lamda,top[0]->cpu_diff(),bottom[0]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(GradientsReverseLayer);
#endif

INSTANTIATE_CLASS(GradientsReverseLayer);
REGISTER_LAYER_CLASS(GradientsReverse);

}  // namespace caffe
