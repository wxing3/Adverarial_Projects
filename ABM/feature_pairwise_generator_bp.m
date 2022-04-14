


function feature_pairwise=feature_pairwise_generator_bp(featuresA,featuresB,general)
%gt=1 the general pairwise features assuming or labeld 1
%we assume feature_pairwise is additive and is non-zero only between the
%samples which have different nodes.

global weight_size_pairwise;
cValue=1;

n_nodesA=size(featuresA,2); 
n_nodesB=size(featuresB,2); 

% n_nodes=size(features,1);
feature_pairwise=zeros(n_nodesA,n_nodesB,weight_size_pairwise);
for i=1:n_nodesA
    for j=1:n_nodesB
        
                    feature_pairwise(i,j,:)=1./(abs(featuresA(:,i)- featuresB(:,j))+cValue);
    end
end
% to do: payiin mosalas ra niz hesaab kon

end



