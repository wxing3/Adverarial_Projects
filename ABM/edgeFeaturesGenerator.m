function features_edges=edgeFeaturesGenerator(frames)
% Generates all the pairwise features
global n_nodes;
global n_features;
n_frames=size(frames,2);
features_edges=cell(n_frames,1);
ds='ETH-Bahnhof';
for f=1:n_frames
    frame=frames{f};
    features=zeros(n_nodes,n_nodes,n_features);
    for i=1:17
        m= frame(i,7);
        annot=[frame(i,2:5),frame(i,8:11)];
        features(i,m,1)=boxiou(annot);
        features(i,m,2)=calcEuclidianDist(frame(i,2:5),frame(i,8:11));                                                        
         features(i,m,3)=calcOpticalFlow(f,i,frame(i,2:5),f+1,m,frame(i,8:11),ds)
    end

features_edges{f}=features;
end
save features_edges features_edges;