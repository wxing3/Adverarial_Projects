
n_row=3;
n_col=7;
n_blocks=n_row*n_col;
n_features=59;
n_images=size(gt,1);
lbpHist_img_Sunnyday=cell(n_images,1);
nFiltSize=8;
nFiltRadius=1;
filtR=generateRadialFilterLBP(nFiltSize, nFiltRadius);

for i=1:n_images
    k=gt(i,1);
    imgName=sprintf('c:/users/sima/documents/adversarialBipartitMatching/dataSet/2DMOT2015/train/ETH-Sunnyday/img1/%06d.jpg',k);
    im=imcrop(imread(imgName), gt(i,3:6));
    im=imresize(im,[15,35]);
    w=15;
    h=35;
    wb=ceil(w/n_col);
    hb=ceil(h/n_row);
    m=0;
    n=0;
    lbpHist_blocks=zeros(n_row,n_col,n_features);
    for r=1:hb:h-hb
        m=m+1;
        for c=1:wb:w-wb
            n=n+1;
            annot=[c, r, min(wb-1,w-c), min(hb-1,h-r)];
            
            imc=imcrop(im, annot);
            gray_imc=rgb2gray(imc);
            lbpFeatures = extractLBPFeatures(gray_imc);
           % numNeighbors = 8;
            %numBins = numNeighbors*(numNeighbors-1)+3;
            %lbpCellHists = reshape(lbpFeatures,numBins,[]);
            %lbpCellHists = bsxfun(@rdivide,lbpCellHists,sum(lbpCellHists));
            lbpFeatures = reshape(lbpCellHists,1,[]);
            lbpHist_blocks(m,n,:)=lbpFeatures;
        end
      n=0;
    end
        lbpHist_img_Sunnyday{i}=reshape(lbpHist_blocks,n_blocks,n_features);
    end

    save lbpHist_img_Sunnyday      lbpHist_img_Sunnyday