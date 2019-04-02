#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void CrossEntropyForwardGPU(const int nthreads,
          const Dtype* bottom_data, const Dtype* label, Dtype* loss_data, int channels) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    
      loss_data[index * channels + label_value] = -log(max(bottom_data[index * channels + label_value],
                      Dtype(FLT_MIN)));
    
  }
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    //return Forward_cpu(bottom, top);
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();

  
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0), loss_data);
  const int nthreads = bottom[0]->num();
  // NOLINT_NEXT_LINE(whitespace/operators)
  CrossEntropyForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data, label, loss_data, bottom[0]->channels());
  Dtype loss = 0.0;
  caffe_gpu_asum(bottom[0]->count(), loss_data, &loss);

  top[0]->mutable_cpu_data()[0] = loss / Dtype(nthreads);

}

template <typename Dtype>
__global__ void CrossEntropyBackwardGPU(const int nthreads, const Dtype* top_diff,
          const Dtype* label, Dtype* bottom_diff, const Dtype* bottom_data, const int channels) {
          
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);

      bottom_diff[index * channels + label_value] -= Dtype(1)/max(bottom_data[index * channels + label_value],Dtype(FLT_MIN));

  }
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    //return Backward_cpu(top, propagate_down, bottom);
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* label = bottom[1]->gpu_data();
    const int nthreads = bottom[0]->num();
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
    CrossEntropyBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_diff, label, bottom_diff, bottom_data, bottom[0]->channels());
    caffe_gpu_scal(bottom[0]->count(), Dtype(1.0)/Dtype(bottom[0]->num()) * top[0]->cpu_diff()[0] , bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CrossEntropyLossLayer);

}  // namespace caffe
