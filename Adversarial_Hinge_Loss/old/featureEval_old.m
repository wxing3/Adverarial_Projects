function features = featureEval_old(samples,labels,featureFunc)
% Evaluate the statistics using given functions
% Inputs:
%   samples - nSample*nDimension matrix: samples
%
%   labels - nSample*1 vector: lables
%
%   featureFuncs - cell nStat*1: function handles which provide statistic
%   constrains.
%`
% Outputs:
%   features - cell nStat*1: statstics of the given data. Each element in
%   features are nSample * nClasses**dimension(projected feature space)
%   matrix.
%
% Created by Wei Xing @ UIC , 2014-07-09
% Updated by Wei Xing @ UIC , 2014-11-19

[nSample,~] = size(samples);
sortedLabelValues  = sort(unique(labels)); % Default is ascending order.
nClasses           = length(sortedLabelValues);

nStat = length(featureFunc);
features = cell(nStat,1);
for i=1:nStat
    nFeatureDimension = length(featureFunc{i}(samples(1,:),labels(1))); 
    features{i} = zeros(nSample,nFeatureDimension*nClasses); % The result for each x and y is a row vector because samples are presented as rows.
    for j=1:nClasses
        for k = 1:nSample
            features{i}(k,(j-1)*nFeatureDimension+1:j*nFeatureDimension) = featureFunc{i}(samples(k,:),sortedLabelValues(j));
        end
    end
    
end

end