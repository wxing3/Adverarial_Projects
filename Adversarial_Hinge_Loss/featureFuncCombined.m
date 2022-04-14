function feature = featureFuncCombined(featureFuncs,x,y,nClasses)
% This statistics is y
%
% Inputs:
%   featureFuncs cell nFunc * 1: feature functions which all provide
%   vector features
%   x nFeature*1: feature
%   y 1: label
% Outputs:
%   statistics - cell nStat*1: statstics of the given data
%
%
% Created by Wei Xing @ UIC , 2014-11-25
% Updated by Wei Xing @ UIC , 2014-11-25

nFunc = length(featureFuncs);
features = cell(nFunc,1);
nFeature = zeros(nFunc,1);

for k = 1:nFunc
    features{k} = featureFuncs{k}(x,y,nClasses);
    nFeature(k) = length(features{k});
end

feature = zeros(sum(nFeature),1);

lastIdx = 0;
for k = 1:nFunc
   feature(lastIdx+1 : lastIdx + nFeature(k)) = features{k}; 
   lastIdx = lastIdx + nFeature(k);
end

end

