function evaluate_recall()
clear all;
label_num=5924;%cub 5924  cars 8131  ebay 60502 market 13115/19732 flower 4696
p=importdata('~/DeML/examples/CUB/U512/val.txt');%cub ../../caffe/examples/model/cub/val.txt%cars  ../../caffe/examples/model/cars/car_test.txt
%ebay ../../caffe/examples/model/ebay/ebay_test.txt
%flower ../../caffe/examples/flower/flower_test.txt
label=p.data;
clear p
curve=[];
for ind =500:500:15000
try
    features_all=[];
    for i=1:1
    fea1=sprintf('~/DeML/examples/CUB/U512/features/model_512_%d.fea',ind)
    feafile=fopen(fea1,'rb');
    [dims]=fread(feafile,1,'int');
    [num]=fread(feafile,1,'int');
    feature=fread(feafile,dims*num,'float');
    fclose(feafile);
     features=reshape(feature,dims,num);
     clear feature
     
     features_all=[features_all;features(:,1:label_num)];%5924,5864==val,train
    end
     features=features_all;
     clear features_all;
       
catch
    error('filename is non existent.\n');
end

kk = [1 2 4 8];
             features=features./repmat(sqrt(sum(features.^2)),size(features,1),1);

features=single(features');
dims = size(features);

 m = size(features, 1);
t = single(ones(m, 1));
x = single(zeros(m, 1));
for i = 1:m
    n = norm(features(i,:));
    x(i) = n * n;
end
D2 =-1*features * features'; % for cosine
class_ids=label;

 num = dims(1);
%  D2 = (sqrt(abs(D2)));
 D2(1:num+1:num*num) = inf;
% 

knn_class_inds=single(zeros(num,kk(end)));
tic
for i = 1 : num
    this_row = D2(i,:);
    [~, inds] = sort(this_row, 'ascend');
    knn_inds = inds(1:kk(end));
    
    knn_class_inds(i,:) = class_ids(knn_inds);
end
toc
for K = kk
    curve=[curve;compute_recall_at_K(K, class_ids, num, knn_class_inds(:,1:K))];
end

end 
disp('done');

% compute recall@K
function recall = compute_recall_at_K(K, class_ids, num, knn_class_inds)
num_correct = 0;
for i = 1 : num
    this_gt_class_idx = class_ids(i);
    
    if sum(ismember(knn_class_inds(i,:), this_gt_class_idx)) > 0
        num_correct = num_correct + 1;
    end
end

recall = num_correct / num;
fprintf('K: %d, Recall: %.3f\n', K, recall);

