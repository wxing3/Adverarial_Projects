function feature = featureFuncLabel(~,y,nClasses)
% This statistics is x*y
%
% Inputs:
%   x nDimension*1: A data sample
%   y : Label of x which is mapping to a integer in {1,...,nClasses}
%   nClasses: Number of possible classes
% Outputs:
%   feature - cell nFeatureDimension*1: A feature vector.
%
% Created by Wei Xing @ UIC , 2014-09-22
% Updated by Wei Xing @ UIC , 2014-01-09

feature  = zeros(nClasses,1);
feature(y) = 1;

end

