function o = objectFuncHinge(theta,otherParas)
% This is the object function of  for statistic constrains
% in binary hinge loss robust classifation model which omit all the parts
% that have nothing to do theta(the Lagrange parameter of statistic
% constrains).
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
%   o : value of object function
%
% Created by Wei Xing @ UIC , 2014-07-10
% Updated by Wei Xing @ UIC , 2014-07-10

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
        slope(i) = slope(i) + theta(lastIdx+1:lastIdx+nFeatureDimension(k)) * (features{k}(i,nFeatureDimension(k)+1:2*nFeatureDimension(k))-features{k}(i,1:nFeatureDimension(k)))';
    end
    lastIdx = lastIdx + nFeatureDimension(k);
end

o = 0;

for i = 1:nUnqSample
    pos = 0; neg = 0; lastIdx = 0;
    for k = 1:nStat
        pos = pos + theta(lastIdx+1:lastIdx+nFeatureDimension(k))*(features{k}(i,nFeatureDimension(k)+1:2*nFeatureDimension(k)))';
        neg = neg + theta(lastIdx+1:lastIdx+nFeatureDimension(k))*(features{k}(i,1:nFeatureDimension(k)))';
        lastIdx = lastIdx + nFeatureDimension(k);
    end
    o = o + freq(i) * chooseOptFeature(slope(i),pos,neg);
end

lastIdx = 0;
for k = 1:nStat
    o = o - theta(lastIdx+1:lastIdx+nFeatureDimension(k))*(statistics{k})';
    lastIdx = lastIdx + nFeatureDimension(k);
end

end

function optFeature = chooseOptFeature(slope, pos, neg)
% Get the optimal feature according to slope.
if slope > 2
    optFeature = pos;
elseif slope < -2
    optFeature = neg;
else
    optFeature = (pos+neg)/2+1; % "+1" here stands for 2*min(P(y=1|x),1-P(y=1|x))
end
end

