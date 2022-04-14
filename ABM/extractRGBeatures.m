
n_row=3;
n_col=7;
n_blocks=n_row*n_col;
n_features=8^3;
n_images=size(gt,1);
rgbHist_img_Stadtmitte=cell(n_images,1);
for i=1:n_images
    k=gt(i,1);
imgName=sprintf('c:/users/sima/documents/adversarialBipartitMatching/dataSet/2DMOT2015/train/TUD-Stadtmitte/img1/%06d.jpg',k);    
im=imcrop(imread(imgName), gt(i,3:6));
im=imresize(im,[15,35]);
w=15;
h=35;
wb=ceil(w/n_col);
hb=ceil(h/n_row);
m=0;
n=0;
rgbHist_blocks=zeros(n_row,n_col,n_features);
for r=1:hb:h-hb
    m=m+1;
    for c=1:wb:w-wb
      n=n+1;
annot=[c, r, min(wb-1,w-c), min(hb-1,h-r)];   

        imc=imcrop(im, annot);
[freq, freq_emph, freq_ly] = image_hist_RGB_3d(imc,8);
rgbHist_blocks(m,n,:)=reshape(freq,[],1);  % 512 *1
    end
    n=0;
end
rgbHist_img_Stadtmitte{i}=reshape(rgbHist_blocks,n_blocks,n_features);
end
      save rgbHist_img_Stadtmitte      rgbHist_img_Stadtmitte