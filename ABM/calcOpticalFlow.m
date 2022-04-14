n_features=256;
opticalHist_Stadtmitte=zeros(size(gt,1),n_features);
ds='TUD-Stadtmitte';
for i=1:size(gt,1)
    opticFlow = opticalFlowHS;
    k=gt(i,1);
imgName=sprintf('c:/users/sima/documents/adversarialBipartitMatching/dataSet/2DMOT2015/train/%s/img1/%06d.jpg',ds,k);    
im1=imcrop(imread(imgName), gt(i,3:6));
I1 = im2double(rgb2gray(im1));
 flow_1 = estimateFlow(opticFlow,I1); 
 p1= ofHistogram(flow_1.Vx,flow_1.Vy,256);
opticalHist_Stadtmitte(i,:)=p1;
end
save opticalHist_Stadtmitte opticalHist_Stadtmitte