//compute ave vector for each class, the input data must be adjacent and the total number of classes must be fixed.
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/class_ave_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ClassAveLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ClassAveLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->height(),1);
  CHECK_EQ(bottom[0]->width(),1);
  top[0]->Reshape(bottom[0]->num()/this->layer_param_.class_ave_param().per_class_num(), bottom[0]->channels(), 1, 1);
}

template <typename Dtype>
void ClassAveLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int per_class_num = this->layer_param_.class_ave_param().per_class_num();
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();

    //compute class ave vector
    for(int i = 0; i < num/per_class_num; i++)
    {
        for(int j = 0; j < per_class_num; j++)
        {
                caffe_cpu_axpby(channels, Dtype(1)/Dtype(per_class_num), bottom_data + (i * per_class_num +j) * channels, Dtype(1), top_data + i * channels);
        }
    }
}

template <typename Dtype>
void ClassAveLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
        const int per_class_num = this->layer_param_.class_ave_param().per_class_num();
        const int num = bottom[0]->num();
        const int channels = bottom[0]->channels();
        
        for(int i = 0; i < num/per_class_num; i++)
        {
                for(int j = 0; j < per_class_num; j++)
                {
                        caffe_cpu_axpby(channels, Dtype(1)/Dtype(per_class_num), top[0]->cpu_diff() + i * channels, Dtype(0), bottom[0]->mutable_cpu_diff() + (i * per_class_num + j) * channels);
                }
        }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ClassAveLayer);
#endif

INSTANTIATE_CLASS(ClassAveLayer);
REGISTER_LAYER_CLASS(ClassAve);

}  // namespace caffe
