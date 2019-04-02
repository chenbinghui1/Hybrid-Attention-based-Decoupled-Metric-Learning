#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include "caffe/layers/erase_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layer.hpp"
#include <cmath>

namespace caffe {

template<typename Dtype>
void EraseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
}

template<typename Dtype>
void EraseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    vector<int> top_shape = bottom[0]->shape();
    for(int i = 0; i < top.size(); i++){
        top[i]->Reshape(top_shape);
    }
    map_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
}
//降序排列
bool comp(std::pair<float, int> i, std::pair<float, int> j){ return (i.first>j.first); }
template<typename Dtype>
void EraseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
        
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* map= map_.mutable_cpu_data();
        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int width = bottom[0]->width();
        int height = bottom[0]->height();
        caffe_set(map_.count(), Dtype(0), map);
        
        
        for(int i = 0; i < num; i++){
                //compute mean map for  i-th data
                for(int j = 0; j < channels; j++){
                        caffe_cpu_axpby(width*height, Dtype(1)/channels, bottom_data + i * channels * width * height + j * width * height, Dtype(1), map + i * width * height);
                }
                //compute top K activation location
                std::vector<std::pair<Dtype, int> > location;
                for(int j = 0; j < width * height; j ++){
                        location.push_back(std::make_pair(map[i*width*height+j], j));
                }
                std::partial_sort(location.begin(), location.begin() + top.size(), location.end(), comp);
                
                for(int j = 0; j < top.size(); j++){
                        caffe_copy(channels*width*height, bottom_data + i * channels * width * height, top[j]->mutable_cpu_data() + i * channels * width * height);
                        //erase data
                        for(int k = j; k <= j; k++){
                                for(int z = 0; z < channels; z++){
                                        top[j]->mutable_cpu_data()[i * channels * width * height + z * width * height + location[k].second] = Dtype(0);
                                }
                        }
                }
        }
}



template<typename Dtype>
void EraseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* map = map_.cpu_data();
        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int width = bottom[0]->width();
        int height = bottom[0]->height();
        caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
        
        for(int i = 0; i < num; i++){
                //compute top K activation location
                std::vector<std::pair<Dtype, int> > location;
                for(int j = 0; j < width * height; j ++){
                        location.push_back(std::make_pair(map[i * width * height + j], j));
                }
                std::partial_sort(location.begin(), location.begin() + top.size(), location.end(), comp);
                //erase diff
                for(int j = 0; j < top.size(); j++){
                        for(int k = j; k <= j; k++){
                                for(int z = 0; z < channels; z++){
                                        top[j]->mutable_cpu_diff()[i * channels * width * height + z * width * height + location[k].second] = Dtype(0);
                                }
                        }
                }
        }
        //trans all top_diff to bottom_diff
        for(int j = 0; j < top.size(); j++)
                caffe_cpu_axpby(top[j]->count(), Dtype(1), top[j]->cpu_diff(), Dtype(1), bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(EraseLayer);
#endif

INSTANTIATE_CLASS(EraseLayer);
REGISTER_LAYER_CLASS(Erase);

}  // namespace caffe
