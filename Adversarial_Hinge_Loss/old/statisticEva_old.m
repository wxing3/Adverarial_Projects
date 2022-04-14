function statistics = statisticEva_old(training,statisticFunc)
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
%   x nSample*nDimension matrix: samples
%
% Created by Wei Xing @ UIC , 2014-07-06
% Updated by Wei Xing @ UIC , 2014-11-25

nStat = length(statisticFunc);
[nSample,~] = size(training.x,1);

statistics = cell(nStat,1);
for i=1:nStat
    statistics{i} = zeros(1,length(statisticFunc{i}(training.x(1,:),training.y(1)))); % This is a row vector because samples are presented as rows.
    for j=1:nSample
        statisticFunc{i}(training.x(j,:),training.y(j))
        statistics{i} = statistics{i} + statisticFunc{i}(training.x(j,:),training.y(j));
    end
    statistics{i} = statistics{i} / nSample;
end

end

