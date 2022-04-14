function jointFeatures=jointFeatureExtraction(features_edges,permutation)
global n_nodes;
global n_features;
jointFeatures=zeros(n_nodes,n_nodes,n_features);
for i=1:n_nodes
   k= permutation(i);
   jointFeatures(i,k,1)=features_edges(i,k,1);
   jointFeatures(i,k,2)=features_edges(i,k,2);
end
