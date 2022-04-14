function [bestAction, bestValue] = predictFull(lagrangeMultipliers, extraPara, isMinimizer, verbose)
% Assuming rows are maximizer actions and columes are minimizer actions,
% and minimizer is the adversary
if nargin < 4
    verbose = false;
    if nargin <3
        isMinimizer = false;
    end
end

global buildFullMaximizerActions buildFullMinimizerActions;
global buildGameMatrix;

[nSubSample, nCase] = size(lagrangeMultipliers);

if nCase == 1
    lagrangeMultipliers = [zeros(nSubSample,1) lagrangeMultipliers]; % For binary case, only documents with label 1 has valid features
    nCase = 2;
end

minimizerActions = buildFullMinimizerActions(nSubSample,nCase);
maximizerActions = buildFullMaximizerActions(nSubSample,nCase);
gameMatrix = buildGameMatrix(minimizerActions, maximizerActions, lagrangeMultipliers, extraPara);
   
if isMinimizer % Finding best minimizer strategy
    [maxV, ~] = max(gameMatrix);
    [minV, minI] = min(maxV);
    bestAction = minimizerActions(:,minI);
    bestValue = minV;
else % Finding best maximizer strategy
    [minV, ~] = min(gameMatrix, [], 2);
    [maxV, maxI] = max(minV);
    bestAction = maximizerActions(:,maxI);
    bestValue = maxV;
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