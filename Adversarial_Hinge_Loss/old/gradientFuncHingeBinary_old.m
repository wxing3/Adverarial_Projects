function [g,b] = gradientFuncHingeBinary_old(theta,tolerance,otherParas)
% This is gradient function of Lagrange parameters for statistic constrains
% in binary hinge loss robust classifation model.
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
%   tolerance - 1*1: Error tolerance used to decide whether the point is
%   on a hyperplane. If tolerance is negtive, than return precise graident
%   but only use this tolerance to judge whether the point is on a
%   hyperlane.
%
% Outputs:
%   g - 1*sum(nFeatureDimension): gradients of Lagrange parameters of
%   each statistic constrains. As the constrains can be vectors, g{i}
%   concatenate all the vectors.
%   b - cell(nBoundaries,1): Information about whether a point is on the boundarys,
%   which includes:
%       b{1} nBoundaries * 1: The index of sample which define the boundary with a sign. If
%       the sign is + then the plane is has "xxx = 2" form, other wise it
%       has "xxx = -2" form.
%       b{2}: sum(nFeatureDimension) * nBoundaries: A matrix in which each
%       colume is the coefficient of a hyperplane that is definde by
%       slop=2 * signof{b{1}}.
%
% Created by Wei Xing @ UIC , 2014-07-04
% Updated by Wei Xing @ UIC , 2014-11-19

isPrecise = false;

if tolerance < 0
    tolerance = -tolerance;
    isPrecise = true;
end

freq = otherParas{1};
featureDifference = otherParas{2};
featureSum = otherParas{3};
statistics = otherParas{4};
nUnqSample = length(freq);
nStat = length(statistics);
nFeatureDimension = zeros(nStat,1); % Record the feature dimensions of each feautre funciton
for k = 1:nStat
    nFeatureDimension(k) = length(statistics{k});
end

slope = zeros(nUnqSample,1); % slope determines griandients
lastIdx = 0; % Record the ending position of last feature function's feature vector in theta.
for k = 1:nStat
    nFeatureDimension(k) = length(statistics{k});
    for i = 1:nUnqSample
        slope(i) = slope(i) + theta(lastIdx+1:lastIdx+nFeatureDimension(k))*(featureDifference{k}(i,:))';
        if isnan(slope(i))
            fprintf('ERROR: Slope value is invalid.\n');
            pause; return;
        end
    end
    lastIdx = lastIdx + nFeatureDimension(k);
end

b = cell(2,1);
b{1} = [];
b{2} = [];

g = zeros(1,length(theta)); % This is a row vector because samples are presented as rows.
lastIdx = 0;
for k = 1:nStat
    for i = 1:nUnqSample
        if isPrecise
            optFeature = chooseOptFeature(slope(i),featureDifference{k}(i,:),featureSum{k}(i,:),0);
        else
            optFeature = chooseOptFeature(slope(i),featureDifference{k}(i,:),featureSum{k}(i,:),tolerance);
        end
        g(lastIdx+1:lastIdx+nFeatureDimension(k)) = g(lastIdx+1:lastIdx+nFeatureDimension(k)) + freq(i) * optFeature;
        % Only check whether the point is on the boundary once
    end
    g(lastIdx+1:lastIdx+nFeatureDimension(k)) = g(lastIdx+1:lastIdx+nFeatureDimension(k)) - statistics{k};
    lastIdx = lastIdx + nFeatureDimension(k);
end

normVector = zeros(sum(nFeatureDimension),1); % A colume vector which is the norm vector of the hyperplane that the point is on.
for i = 1:nUnqSample
    if (slope(i) >= 2 - tolerance) && (slope(i) <= 2 + tolerance)
        b{1} = [b{1}; i];
        lastIdx = 0;
        for k = 1:nStat
            normVector(lastIdx+1:lastIdx+nFeatureDimension(k)) = featureDifference{k}(i,:)';
            lastIdx = lastIdx + nFeatureDimension(k);
        end
        b{2} = [b{2} normVector];
    elseif (slope(i) >= -2 - tolerance) && (slope(i) <= -2 + tolerance)
        b{1} = [b{1}; -i];
        lastIdx = 0;
        for k = 1:nStat
            normVector(lastIdx+1:lastIdx+nFeatureDimension(k)) = featureDifference{k}(i,:)';
            lastIdx = lastIdx + nFeatureDimension(k);
        end
        b{2} = [b{2} normVector];
    end
end

end

function optFeature = chooseOptFeature(slope, featureDifference, featureSum, tolerance)

% Get the optimal feature according to slope.
if slope > 2 + tolerance
    optFeature = (featureSum + featureDifference) / 2;
elseif slope < -2 - tolerance
    optFeature = (featureSum - featureDifference) / 2;
elseif slope > -2 + tolerance && slope < 2 - tolerance
    optFeature = featureSum / 2;
elseif slope >= 2 - tolerance && slope <= 2 + tolerance
    optFeature = featureSum / 2 + featureDifference / 4;
elseif slope >= -2 - tolerance && slope <= -2 + tolerance
    optFeature = featureSum / 2 - featureDifference / 4;
else
    fprintf('ERROR: Slope value cannot be decided.\n');
    pause; return;
end

end