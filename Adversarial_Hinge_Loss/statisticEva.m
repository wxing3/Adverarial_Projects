function statistics = statisticEva(training,statisticFuncs)
% Evaluate the statistics using given functions
% Inputs:
%   training - structure: training data
%   statisticFuncs - cell nStat*1: function handles which provide statistic
%   constrains.
% Outputs:
%   statistics - cell nStat*1: statstics of the given data
%
% Training data should both be a structure with
%   y nSample*1 vector: lables
%   x nDimension*nSample matrix: samples
%
% Created by Wei Xing @ UIC , 2014-11-25
% Updated by Wei Xing @ UIC , 2014-11-25

[nSample,~] = size(training.x);

statistics = zeros(length(featureFuncCombined(statisticFuncs,training.x(:,1),training.y(1))),1); % This is a row vector because samples are presented as rows.
for j=1:nSample
    statistics = statistics + featureFuncsCombined(statisticFuncs,training.x(:,j),training.y(j));
end
statistics = statistics / nSample;

end

