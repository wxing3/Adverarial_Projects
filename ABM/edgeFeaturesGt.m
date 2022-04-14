function features_edges=edgeFeaturesGt(gt)
%gt=import(gtFileAddress);
nf=size(unique(gt(:,1)),1);
maxNodes=max(hist(gt(:,1),unique(gt(:,1)))); % find the max number of onjects in frames
u=-1;
n_nodes=2* maxNodes; %  every node can be in the frame or no, therefore 2*Max_nodes
n_features=2; %IOU and Eucleadian distance
features_edges=cell(nf-1,1);
matching=cell(nf-1,2);
for i=1:nf-1
    featuresMat=zeros(n_nodes,n_nodes,n_features);
    s1=find(gt(:,1)==i);    % return the indicies of the frame i
    s2=find(gt(:,1)==i+1);% return the indicies of the frame i+1
    f1=gt(s1,:); %extract the first frame
    f2=gt(s2,:) ;%extract the second frame
  matching{i,1}=f1;
  matching{i,2}=f2;
  featuresMat(:,:,1)=0;
   featuresMat(:,:,2)=256;
    for m=1:size(s1,2)
         for n=1:size(s2,2)
           obj1= f1(m,2);
           obj2=f2(n,2);
           annot=[f1(m,3:6) ,f2(n,3:6)];
             featuresMat(obj1,obj2,1)=boxiou(annot);
              featuresMat(obj1,obj2,2)=calcEuclidianDist(f1(m,3:6), f2(n,3:6));
              matches
         end
    end
     features_edges{i}=featuresMat;
end
 save features_edges features_edges
 save matching matching