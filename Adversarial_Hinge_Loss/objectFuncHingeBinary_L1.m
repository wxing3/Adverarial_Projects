function o = objectFuncHingeBinary_L1(theta,otherParas)
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


regPara = 1e-4; % L1 relulization parameter
nUnqSample = length(freq);

slope = zeros(nUnqSample,1); % slope determines griandients

for i = 1:nUnqSample
    slope(i) = slope(i) + theta' * (featureDifference(:,i));
end

o = 0;

for i = 1:nUnqSample
    pos = 0; neg = 0; 
        pos = pos + theta'*((featureSum(:,i) + featureDifference(:,i)) / 2);
        neg = neg + theta'*((featureSum(:,i) - featureDifference(:,i)) / 2);
    o = o + freq(i) * chooseOptFeature(slope(i),pos,neg);
end

o = o - theta' * statistics * freq + regPara * norm(theta,1);

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

