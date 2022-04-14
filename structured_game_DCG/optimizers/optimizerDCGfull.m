function [scoreMatrix, ...
    minimizerActions, maximizerActions, ...
    minimizerProbabilities, maximizerProbabilities, ...
    minimizerValue, maximizerValue, flag] = optimizerDCGfull(lagrangeMultipliers, IDCG, valuePrecision, verbose)
% This version stops when there's no new action anymore.
%     if (~isvector(lagrangeMultipliers))
%         error('Input "lagrangeMultipliers" must be a vector.');
if nargin < 4
    verbose = false;
end

global buildGameMatrix;
global buildFullMaximizerActions buildFullMinimizerActions;

[nDoc, nRel] = size(lagrangeMultipliers);

if nRel == 1
    lagrangeMultipliers = [zeros(nDoc,1) lagrangeMultipliers]; % For binary case, only documents with label 1 has valid features
    nRel_real = 2;
else
    nRel_real = nRel;
end

maximizerActions = buildFullMaximizerActions(nDoc,nRel_real);
minimizerActions = buildFullMinimizerActions(nDoc,nRel_real);

scoreMatrix = buildGameMatrix(minimizerActions, maximizerActions, lagrangeMultipliers, IDCG);

calledTimes = 0;
flag = 0;

[maximizerProbabilities, maximizerValue, errorFlag1] = findMaximizerProbabilities(scoreMatrix, calledTimes, verbose);

[nonZeroMaximizerProbabilities, nonZeroProbMaximizerActions] = ...
    findNonZeroProbabilitiesAndActions(maximizerProbabilities, maximizerActions, valuePrecision);

nonZeroMaximizerProbabilities = nonZeroMaximizerProbabilities / sum(nonZeroMaximizerProbabilities);

maximizerActions = nonZeroProbMaximizerActions;
maximizerProbabilities = nonZeroMaximizerProbabilities;

[minimizerProbabilities, minimizerValue, errorFlag2] = findMinimizerProbabilities(scoreMatrix, calledTimes, verbose);

[nonZeroMinimizerProbabilities, nonZeroProbMinimizerActions] = ...
    findNonZeroProbabilitiesAndActions(minimizerProbabilities, minimizerActions, valuePrecision);

nonZeroMinimizerProbabilities = nonZeroMinimizerProbabilities / sum(nonZeroMinimizerProbabilities);

minimizerActions = nonZeroProbMinimizerActions;
minimizerProbabilities = nonZeroMinimizerProbabilities;
scoreMatrix = buildScoreMatrix(minimizerActions, maximizerActions, lagrangeMultipliers, IDCG);

if errorFlag1 || errorFlag2
    flag = 1;
    warning('The game value cannot be solved.');
end

end
