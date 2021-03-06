//compute cov matrix (vectorize) for each class, the input data must be adjacent and the total number of classes must be fixed.
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/class_cov_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ClassCovLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ClassCovLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->height(),1);
  CHECK_EQ(bottom[0]->width(),1);
  top[0]->Reshape(bottom[0]->num()/this->layer_param_.class_cov_param().per_class_num(), bottom[0]->channels()*bottom[0]->channels(), 1, 1);
  temp_.Reshape(1,1,this->layer_param_.class_cov_param().per_class_num(),bottom[0]->channels());
    one_.Reshape(1,1,this->layer_param_.class_cov_param().per_class_num(),this->layer_param_.class_cov_param().per_class_num());
    //initialize ones metrix
    for(int i=0; i<this->layer_param_.class_cov_param().per_class_num(); i++)
        for(int j=0; j<this->layer_param_.class_cov_param().per_class_num(); j++)
        {
                if(i==j)
                        one_.mutable_cpu_data()[i*this->layer_param_.class_cov_param().per_class_num()+j] = Dtype((this->layer_param_.class_cov_param().per_class_num()-1))/this->layer_param_.class_cov_param().per_class_num()/this->layer_param_.class_cov_param().per_class_num();
                else
                        one_.mutable_cpu_data()[i*this->layer_param_.class_cov_param().per_class_num()+j] = -1.0/this->layer_param_.class_cov_param().per_class_num()/this->layer_param_.class_cov_param().per_class_num();
        }    
}

template <typename Dtype>
void ClassCovLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int per_class_num = this->layer_param_.class_cov_param().per_class_num();
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();

    //compute class ave vector
    for(int i = 0; i < num/per_class_num; i++)
    {
                caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, per_class_num, channels, per_class_num, Dtype(1), one_.cpu_data(), bottom_data + i * per_class_num * channels, Dtype(0), temp_.mutable_cpu_data());
                caffe_cpu_gemm(CblasTrans, CblasNoTrans, channels, channels, per_class_num, Dtype(1), bottom_data + i * per_class_num * channels, temp_.cpu_data(), Dtype(0), top_data + i * channels * channels);
    }
}

template <typename Dtype>
void ClassCovLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const int per_class_num = this->layer_param_.class_cov_param().per_class_num();
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    for(int i = 0; i < num/per_class_num; i++)
    {
                caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, per_class_num, channels, per_class_num, Dtype(1), one_.cpu_data(), bottom_data + i * per_class_num * channels, Dtype(0), temp_.mutable_cpu_data());
                caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, per_class_num, channels, channels, Dtype(2), temp_.cpu_data(), top[0]->cpu_diff() + i * channels * channels, Dtype(0), bottom[0]->mutable_cpu_diff() + i * per_class_num * channels);
    }
    
}

#ifdef CPU_ONLY
STUB_GPU(ClassCovLayer);
#endif

INSTANTIATE_CLASS(ClassCovLayer);
REGISTER_LAYER_CLASS(ClassCov);

}  // namespace caffe
