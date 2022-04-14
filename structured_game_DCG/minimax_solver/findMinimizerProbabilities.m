function [p, v, errorFlag] = findMinimizerProbabilities(scoreMatrix, calledTimes, verbose)
    if nargin < 2
        calledTimes = 1;
    end
    [p, v, errorFlag] = findMaximizerProbabilities(-scoreMatrix', calledTimes, verbose);
    v = -v; % revserse minimizer's value to get the same symbol with maximizer
end

