function [bestResponseActions, bestResponseValue] = findBestMaximizerResponseActionExhaust(...
        minimizerProbs, minimizerActions, lagrangeMultipliers, extraPara, valuePrecision)
    [nDoc,~] = size(lagrangeMultipliers);
    maximizerActions = perms(1:nDoc)';
    scoreMatrix = buildScoreMatrix(minimizerActions, maximizerActions, lagrangeMultipliers, extraPara);
    maximizerResponseValues = scoreMatrix * minimizerProbs;
    bestResponseValue = max(maximizerResponseValues);
    bestResponseActions = maximizerActions(:,abs(maximizerResponseValues-bestResponseValue)<=valuePrecision);
end