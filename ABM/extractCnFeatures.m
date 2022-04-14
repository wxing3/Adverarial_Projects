setup ; % \using Vlfeat libraries

net = load('imagenet-vgg-verydeep-16.mat') ;
n_features=4096;
cnnFeatures_kitti17=zeros(size(gt,1),4096);
for i=1:size(gt,1)
    k=gt(i,1);
imgName=sprintf('c:/users/sima/documents/adversarialBipartitMatching/dataSet/2DMOT2015/train/KITTI-17/img1/%06d.jpg',k);    
im=imcrop(imread(imgName), gt(i,3:6));

im_ = single(im) ; % note: 255 range

im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus,im_,net.normalization.averageImage) ;
% run the CNN
res = vl_simplenn(net, im_) ;
   im_cnnFeatures=res(36).x;
cnnFeatures_kitti17(i,:)=reshape(im_cnnFeatures,[],1);
end
save cnnFeatures_kitti17 cnnFeatures_kitti17