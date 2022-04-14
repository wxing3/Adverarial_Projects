function data=dataExtraction(gtFileAddress)
gt=import(gtFileAddress);
nf=size(unique(gt(:,1)),1);
maxNodes=max(hist(gt(:,1),unique(gt(:,1)))); % find the max number of onjects in frames
u=-1;
n_nodes=maxNodes; %  every node can be in the frame or no, therefore 2*Max_nodes
n_features=2; %IOU and Eucleadian distance
data=cell(nf-1,1);
randomizedGt=zeros(nf,maxNodes);
for i=1:nf-1
    featuresMat=zeros(n_nodes,n_nodes,n_features);
    matches=zeros(n_nodes,n_nodes);
        ee=zeros(n_nodes,n_nodes); %enter/exit to/from the frame

    s1=find(gt(:,1)==i);    % return the indicies of the frame i
    s2=find(gt(:,1)==i+1);% return the indicies of the frame i+1
    f1=gt(s1,:); %extract the first frame
    f2=gt(s2,:) ;%extract the second frame
   % randomizedGt(i,1:maxNodes)=randperm(maxNodes);
    for k=1:maxNodes
        if (ismember(k,f1(:,2)))
            f1Obj=find(f1(:,2)==k); %finds the index of node k in frame 1
        else
            f1Obj=0;
        end
        if (ismember(k,f2(:,2)))
            f2Obj=find(f2(:,2)==k);%finds the index of node k in frame 2
        else
            f2Obj=0;
        end
        if (f1Obj & f2Obj) % if there is a match between 2 nodes
            annot=[f1(f1Obj,3:6) ,f2(f2Obj,3:6)];
            featuresMat(k,k,1)=boxiou(annot);
            featuresMat(k,k,2)=calcEuclidianDist(f1(f1Obj,3:6), f2(f2Obj,3:6));
            matches(k,k)=1;
        else
            if (f1Obj) % if node k disppears in the next frame
                ee(k,k)=-1;
            else % if node k just appears in the second frame
                if(f2Obj)
                    ee(k,k)=-1;
                end
            end
        end
    end
    data{i,1}=featuresMat;
    data{i,2}=matches;
    data{i,3}=features_edges{i};
    data{i,4}=ee;
    
    
end
save data data
end

