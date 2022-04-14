function adDist = getAdDist01Binary(varargin)
% Input:
% cost nLabel-by-nLabel: cost matrix
%
% theta nFeatureDimensionAll-by-1: Langrage parameters we trained in last step
%
% features nFeature*nLabel-by-nSample: Features for each possible label
% which can be got from the following code:
%       [unqSample,unqLabels,~,~] = sampleFreq(training.x,training.y); % unqSampel is a
%       features = featureEval(unqSample,unqLabels,featureFuncs);
%
% statistcs nFeature-by-nSample: Feature from the impirical data, which can
% be got using the following code
%   
%   nFeature = size(features,1)/nLabel;
%   statistics = zeros(nFeature,size(unqSample,2));
%   sortedLabelValues  = sort(unique(unqLabels);
%   for i = 1:size(unqSample,2)
%               idx = find(sortedLabelValues == unqLabels(i));
%               statistics(:,i) = features(nFeature*(idx-1)+1:nFeature*idx,i);
%   end
%
% Output:
% adDist - nLabel-by-nSample: Each colume is the adversary condition
% distribution P_check(y|x) for one sample x.
%
% Created by Wei Xing @ UIC , 2014-11-25
% Updated by Wei Xing @ UIC , 2014-11-25

cost = varargin{1};
theta = varargin{2};
features = varargin{3};
statistics = varargin{4};

nUnqSample = size(features,2);
nFeature = length(theta);
nLabel = size(features,1)/nFeature;

sampleFeatures = zeros(nFeature,nLabel,nUnqSample);
for i = 1:nUnqSample
    for l = 1:nLabel
        sampleFeatures(:,l,i) = features(nFeature*(l-1)+1:nFeature*l,i)';
    end
end

adDist = zeros(nLabel,nUnqSample); % Matrix of P(y|x), each colume for each sample x

for i = 1:nUnqSample
    adDist(:,i) = constrained_minimax(cost, theta, sampleFeatures(:,:,i), statistics(:,i));
end