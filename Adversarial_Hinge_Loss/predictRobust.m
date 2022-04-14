function labels = predictRobust(samples,sortedLabelValues,featureFunc,paras,typeLoss)
% Predict the results using robust classifier
%
% Inputs:
%   sample - nSample*nDimension: unlabeled testing data
%
%   sortedLabelValues - nClasses * 1: availabe lables in ascending order.
%
%   featureFuncs - cell nStat*1: function handles which provide valid
%   features in statistics constrain.
%
%   paras cell nStat*1: trained Lagrange parameters and other given parameters.
%
%   typeLoss - string: loss function type
%
% Outputs:
%   labels :
%
% Available types are: 'hinge'
%
% nSample is the number of samples and nDimension is the dimension of feature
% space, nStat is number of statistic funcitons.
%
% Created by Wei Xing @ UIC , 2014-07-03
% Updated by Wei Xing @ UIC , 2014-07-10

[nSample,~] = size(samples);
labels = zeros(nSample,1);
nStat = length(featureFunc);

switch typeLoss
    case {'hinge'}
        
        slope = zeros(nSample,1);
        
        features = featureEval(samples,sortedLabelValues,featureFunc);
        for k = 1:nStat
            nFeatureDimension = length(paras{k});
            for i = 1:nSample
                slope(i) = slope(i) + paras{k}*(features{k}(i,nFeatureDimension+1:2*nFeatureDimension)-features{k}(i,1:nFeatureDimension))';
            end
        end
        
        for i = 1:nSample
            if slope(i) > 0
                labels(i) = sortedLabelValues(2);
            else 
                labels(i) = sortedLabelValues(1);
            end
        end
end

end

