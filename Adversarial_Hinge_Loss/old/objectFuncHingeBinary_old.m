function o = objectFuncHingeBinary_old(theta,otherParas)
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
%   featuresDifference - cell nStat*1: statstics of the given data. Each
%   element in features are nSample * nClasses**dimension(projected feature
%   space) matrix. featureDifferece is the difference using the feature
%   of the label 1 minus the one of the label -1 for each sample. 
%   featureSum cell nStat * 1: featureSum is the difference using the
%   feature of the label 1 plus the one of the label -1 for each sample. 
%   statistics - cell nStat * 1: The expectation of features of the training
%   data.
%
% Outputs:
%   o : value of object function
%
% Created by Wei Xing @ UIC , 2014-07-10
% Updated by Wei Xing @ UIC , 2014-11-08

freq = otherParas{1};
featureDifference = otherParas{2};
featureSum = otherParas{3};
statistics = otherParas{4};

nUnqSample = length(freq);
nStat = length(statistics);
nFeatureDimension = zeros(nStat,1); % Record the feature dimensions of each feautre funciton

slope = zeros(nUnqSample,1); % slope determines griandients
lastIdx = 0; % Record the ending position of last feature function's feature vector in theta.
for k = 1:nStat
    nFeatureDimension(k) = length(statistics{k});
    for i = 1:nUnqSample
        slope(i) = slope(i) + theta(lastIdx+1:lastIdx+nFeatureDimension(k)) * (featureDifference{k}(i,:))';
    end
    lastIdx = lastIdx + nFeatureDimension(k);
end

o = 0;

for i = 1:nUnqSample
    pos = 0; neg = 0; lastIdx = 0;
    for k = 1:nStat
        pos = pos + theta(lastIdx+1:lastIdx+nFeatureDimension(k))*((featureSum{k}(i,:) + featureDifference{k}(i,:)) / 2)';
        neg = neg + theta(lastIdx+1:lastIdx+nFeatureDimension(k))*((featureSum{k}(i,:) - featureDifference{k}(i,:)) / 2)';
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

