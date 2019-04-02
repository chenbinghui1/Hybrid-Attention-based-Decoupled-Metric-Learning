// bottom0 和 bottom1两个batch之间比
#include <algorithm>
#include <vector>

#include "caffe/layers/batch_contrastive_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BatchContrastiveLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_.Reshape(1, bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}

template <typename Dtype>
void BatchContrastiveLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int channels = bottom[0]->channels();
  Dtype margin = this->layer_param_.batch_contrastive_loss_param().margin();
  const Dtype* datai = bottom[0]->cpu_data();
  const Dtype* dataj = bottom[1]->cpu_data();
  const Dtype* labeli = bottom[2]->cpu_data();
  const Dtype* labelj = bottom[3]->cpu_data();
  Dtype loss(0.0);
  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
  
  int w_pos=0, w_neg=0;
    for(int i=0; i<bottom[0]->num(); i++)
        for(int j=0; j<bottom[1]->num(); j++)
            static_cast<int>(labeli[i])==static_cast<int>(labelj[j]) ? w_pos++ : w_neg++;
  w_pos=w_pos/2;
  
  for (int i = 0; i < bottom[0]->num(); ++i) {
        for(int j = 0; j < bottom[1]->num(); ++j)
        {
                if(static_cast<int>(labeli[i])==static_cast<int>(labelj[j]))
                {
                        caffe_sub(bottom[0]->channels(), datai + i * channels, dataj + j * channels, diff_.mutable_cpu_data());
                        loss = loss + Dtype(0.5) * caffe_cpu_dot(channels, diff_.cpu_data(), diff_.cpu_data()) / Dtype(w_pos);
                        //gradients
                        caffe_cpu_axpby(channels, Dtype(1)/w_pos, diff_.cpu_data(), Dtype(1), bottom[0]->mutable_cpu_diff() + i * channels);
                        caffe_cpu_axpby(channels, Dtype(-1)/w_pos, diff_.cpu_data(), Dtype(1), bottom[1]->mutable_cpu_diff() + j * channels);
                }
                else
                {
                        caffe_sub(bottom[0]->channels(), datai + i * channels, dataj + j * channels, diff_.mutable_cpu_data());
                        loss = loss + std::max(Dtype(0), margin - caffe_cpu_dot(channels, diff_.cpu_data(), diff_.cpu_data())) / Dtype(2*w_neg);
                        //gradients
                        if(margin>caffe_cpu_dot(channels, diff_.cpu_data(), diff_.cpu_data()))
                        {
                                caffe_cpu_axpby(channels, Dtype(-1)/w_neg, diff_.cpu_data(), Dtype(1), bottom[0]->mutable_cpu_diff() + i * channels);
                                caffe_cpu_axpby(channels, Dtype(1)/w_neg, diff_.cpu_data(), Dtype(1), bottom[1]->mutable_cpu_diff() + j * channels);
                        }
                }
        }
  }
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void BatchContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    caffe_cpu_scale(bottom[0]->count(), top[0]->cpu_diff()[0], bottom[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
    caffe_cpu_scale(bottom[1]->count(), top[0]->cpu_diff()[0], bottom[1]->cpu_diff(), bottom[1]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(BatchContrastiveLossLayer);
#endif

INSTANTIATE_CLASS(BatchContrastiveLossLayer);
REGISTER_LAYER_CLASS(BatchContrastiveLoss);

}  // namespace caffe
