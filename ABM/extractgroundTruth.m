function groundTruth=extractgroundTruth(randomizedFrames)
global n_nodes;
s=size(randomizedFrames,2);
groundTruth=cell(s,1);
for i=1:s
    g=zeros(n_nodes,n_nodes);
    temp=randomizedFrames{i};
    for m=1:17
        ind1=temp(m,1);
        ind2=temp(m,7);
        g(ind1,ind2)=1;
    end
    groundTruth{i}=g;
end
save groundTruth groundTruth
    
    