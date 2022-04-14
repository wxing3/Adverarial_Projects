setup ; % \using Vlfeat libraries

hogFeatures_kitti13=zeros(size(gt,1),81);
for i=1:size(gt,1)
    k=gt(i,1);
imgName=sprintf('c:/users/sima/documents/adversarialBipartitMatching/dataSet/2DMOT2015/train/KITTI-13/img1/%06d.jpg',k);    
im=imcrop(imread(imgName), gt(i,3:6));

  
hogFeatures_kitti13(i,:)= HOG(im);

end
save hogFeatures_kitti13 hogFeatures_kitti13