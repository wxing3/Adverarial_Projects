function feature = featureFuncQuad(x,y)
% This statistics is x*y
%
% Inputs:
%   x nDimension*1: A data sample
%   y : Label of x which is mapping to a integer in {1,...,nClasses}
%   nClasses: Number of possible classes
% Outputs:
%   feature - cell nFeatureDimension*1: A feature vector.
%
% Updated by Wei Xing @ UIC , 2014-07-09
% Updated by Wei Xing @ UIC , 2014-01-09


nDimension = length(x);
x = x*x';
feature  = zeros((nDimension+1) * nDimension / 2,1);
for i=1:length(feature)
    feature(i) = x(floor(i/nDimension),mod(i,nDimension));
end

end