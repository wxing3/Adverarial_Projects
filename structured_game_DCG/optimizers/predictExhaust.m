function [bestAction, bestValue] = predictExhaust(lagrangeMultipliers, extraPara, isMinimizer, verbose)
% Assuming rows are maximizer actions and columes are minimizer actions, and minimizer is the adversary
if nargin < 4
    verbose = false;
    if nargin <3
        isMinimizer = false;
    end
end

global findBestMaximizerResponseAction findBestMinimizerResponseAction;
global buildFullMaximizerActions buildFullMinimizerActions;

[nSubSample, nCase] = size(lagrangeMultipliers);

if nCase == 1
    lagrangeMultipliers = [zeros(nSubSample,1) lagrangeMultipliers]; % For binary case, only documents with label 1 has valid features
    nCase = 2;
end


if isMinimizer % Finding best minimizer strategy
    minimizerActions = buildFullMinimizerActions(nSubSample,nCase);
    nActions = size(minimizerActions,2);
    bestValue = realmax;
    for i = 1:nActions
        [~, bestResponseValue] = findBestMaximizerResponseAction(...
        1, minimizerActions(:,i), lagrangeMultipliers, extraPara);
        if bestResponseValue < bestValue
            bestValue = bestResponseValue;
            bestAction = minimizerActions(:,i);
        end
    end
else % Finding best maximizer strategy
    maximizerActions = buildFullMaximizerActions(nSubSample,nCase);
    nActions = size(maximizerActions,2);
    bestValue = -realmax;
    for i = 1:nActions
        [~, bestResponseValue] = findBestMinimizerResponseAction(...
        1, maximizerActions(:,i), lagrangeMultipliers, extraPara);
        if bestResponseValue > bestValue
            bestValue = bestResponseValue;
            bestAction = maximizerActions(:,i);
        end
    end
end

if verbose
    if isMinimizer
        fprintf('Best minimizer action leads to game value %f and it is:\n',bestValue);
        disp(bestAction');
    else
        fprintf('Best maximizer action leads to game value %f and it is:\n',bestValue);
        disp(bestAction');
    end
end