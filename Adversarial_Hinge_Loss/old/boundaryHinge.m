function b = boundaryHinge(theta,otherParas)
% This function get the boundaries if a point in space of theta is in some
% boundaries. A boundary is ralated to a sample and is the hyper plane that
% make the "slope" be 2 or -2.
%
% Inputs:
%   theta - sum(nFeatureDimension)*1: It is the Lagrange parameters of each
%   statistic constrains. As the constrains can be vectors, theta
%   concatenate all the vectors.
%   freq - nUnqSample * 1: frequency of unique samples in training data.
%   features - cell nStat*1: statstics of the given data. Each element in
%   features are nSample * nClasses**dimension(projected feature space)
%   matrix.
% Outputs:
%   g - 1*sum(nFeatureDimension): gradients of Lagrange parameters of
%   each statistic constrains. As the constrains can be vectors, g{i}
%   concatenate all the vectors.
%
% Created by Wei Xing @ UIC , 2014-10-30
% Updated by Wei Xing @ UIC , 2014-11-03

epsilon = 1e-8;

freq = otherParas{1};
features = otherParas{2};
statistics = otherParas{3};
nUnqSample = length(freq);
nStat = length(statistics);
nFeatureDimension = zeros(nStat,1); % Record the feature dimensions of each feautre funciton


slope = zeros(nUnqSample,1); % slope determines griandients
lastIdx = 0; % Record the ending position of last feature function's feature vector in theta.
for k = 1:nStat
    nFeatureDimension(k) = length(statistics{k});
    for i = 1:nUnqSample
        slope(i) = slope(i) + theta(lastIdx+1:lastIdx+nFeatureDimension(k))*(features{k}(i,nFeatureDimension(k)+1:2*nFeatureDimension(k))-features{k}(i,1:nFeatureDimension(k)))';
    end
    lastIdx = lastIdx + nFeatureDimension(k);
end

g = zeros(1,length(theta)); % This is a row vector because samples are presented as rows.
lastIdx = 0;
for k = 1:nStat
    for i = 1:nUnqSample
        g(lastIdx+1:lastIdx+nFeatureDimension(k)) = g(lastIdx+1:lastIdx+nFeatureDimension(k)) + freq(i) * chooseOptFeature(slope(i),features{k}(i,:),nFeatureDimension(k));
    end
    g(lastIdx+1:lastIdx+nFeatureDimension(k)) = g(lastIdx+1:lastIdx+nFeatureDimension(k)) - statistics{k};
    lastIdx = lastIdx + nFeatureDimension(k);
end

end

function optFeature = chooseOptFeature(slope, features, dimension)
% Get the optimal feature according to slope.
if slope > 2 + epsilon
    optFeature = features(dimension+1:2*dimension);
elseif slope < -2 - epsilon
    optFeature = features(1:dimension);
elseif slope > -2 + epsilon && slope < 2 - epsilon
    optFeature = (features(1:dimension)+features(dimension+1:2*dimension))/2;
elseif slope >= 2 - epsilon && slope <= 2 + epsilon
    optFeature = 3 * features(dimension+1:2*dimension) / 4 + features(1:dimension) / 4;
elseif slope >= -2 - epsilon && slope <= -2 + epsilon
    optFeature = features(dimension+1:2*dimension) / 4 + 3 * features(1:dimension) / 4;
end
end
