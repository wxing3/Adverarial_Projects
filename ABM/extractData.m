
function[frames,frames_randomized,features_edges,groundTruth,n_nodes,n_frames]=extractData(dataAddress)
%dataset ETH-Bahnhof

gt=importdata(dataAddress);    % 7670*10 -- cell-1:frame number, cell-2: objectID, cell-3:cell-6 :annotation
global n_frames;
global n_nodes;
[frames,n_nodes,n_frames]=extractFrames(gt); 
frames_randomized=randomize(frames);
features_edges=edgeFeaturesGenerator(frames_randomized);
groundTruth=extractgroundTruth(frames_randomized);
save data.mat frames frames_randomized features_edges groundTruth;