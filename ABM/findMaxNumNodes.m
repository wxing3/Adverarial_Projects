tedad=zeros(1000,1);
frameNodes=zeros(1000,10);
for i=1:1000
    k=1;
for j=1:7670
    if (gt(j,1) == i)
        tedad(i,1)=tedad(i,1)+1;
        frameNodes(i,k)=gt(j,2);
        k=k+1;
        
    end
end
end