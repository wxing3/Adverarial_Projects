function features = featureEval(samples,labels,featureFuncs)
% Evaluate the statistics using given functions
% Inputs:
%   samples - nDimension*nSample matrix: samples
%
%   labels - nSample*1 vector: lables
%
%   featureFuncs - 1*1: function handles which provide features
%`
% Outputs:
%   features - nFeature*nSample: features of the given data. Each element in
%   features are nSample * nClasses**dimension(projected feature space)
%   matrix.
%
% Created by Wei Xing @ UIC , 2014-07-09
% Updated by Wei Xing @ UIC , 2014-11-25

nSample = size(samples,2);
sortedLabelValues  = sort(unique(labels)); % Default is ascending order.
nClasses           = length(sortedLabelValues);

nFeature = length(featureFuncCombined(featureFuncs,samples(:,1),labels(1)));

features = zeros(nFeature*nClasses,nSample); % The result for each x and y is a row vector because samples are presented as rows.
for j=1:nClasses
    for k = 1:nSample
        features((j-1)*nFeature+1:j*nFeature,k) = featureFuncCombined(featureFuncs,samples(:,k),sortedLabelValues(j));
    end
end

end