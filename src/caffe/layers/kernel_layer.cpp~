#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/kernel_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KernelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.kernel_param().num_output();


  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
      this->blobs_.resize(1);
    // Initialize the weights
    vector<int> weight_shape(2);
      weight_shape[0] = num_output;
      weight_shape[1] = bottom[0]->channels();
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.kernel_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void KernelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.kernel_param().num_output();
  top[0]->Reshape(bottom[0]->num(),num_output,1,1);
  temp_.Reshape(1,bottom[0]->channels(),1,1);
}

template <typename Dtype>
void KernelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* temp = temp_.mutable_cpu_data();
  
  const int num_output = this->layer_param_.kernel_param().num_output();
  const Dtype ksi = this->layer_param_.kernel_param().ksi();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  for(int i = 0; i < num; i++)
  {
        for(int j = 0; j < num_output; j++)
        {
                caffe_copy(channels, bottom_data + i * channels, temp);
                caffe_cpu_axpby(channels, Dtype(-1.0), weight + j * channels, Dtype(1.0), temp);
                top_data[i * num_output + j] = std::max(std::exp(Dtype(-0.5)*caffe_cpu_dot(channels, temp, temp)/(ksi*ksi)), Dtype(1e-10));
                //LOG(INFO)<<top_data[i*num_output+j];
        }
  }
}

template <typename Dtype>
void KernelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_data = top[0]->cpu_data();
    Dtype* temp = temp_.mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    
    const int num_output = this->layer_param_.kernel_param().num_output();
    const Dtype ksi = this->layer_param_.kernel_param().ksi();
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    caffe_set(bottom[0]->count(), Dtype(0.0), bottom[0]->mutable_cpu_diff());
    // Gradient with respect to bottom data and weights
    for(int i = 0; i < num; i++)
    {
                for(int j = 0; j < num_output; j++)
                {
                        caffe_copy(channels, bottom_data + i * channels, temp);
                        caffe_cpu_axpby(channels, Dtype(-1.0), weight + j * channels, Dtype(1.0), temp);
                        //bottom data diff
                        caffe_cpu_axpby(channels, Dtype(-1.0*top_diff[i*num_output+j] * top_data[i*num_output+j]/(ksi*ksi)), temp, Dtype(1.0), bottom_diff + i * channels);
                        //weights diff
                        caffe_cpu_axpby(channels, Dtype(top_diff[i*num_output+j] * top_data[i*num_output+j]/(ksi*ksi)), temp, Dtype(1.0), this->blobs_[0]->mutable_cpu_diff() + j * channels);
                }
    }

}

#ifdef CPU_ONLY
STUB_GPU(KernelLayer);
#endif

INSTANTIATE_CLASS(KernelLayer);
REGISTER_LAYER_CLASS(Kernel);

}  // namespace caffe
