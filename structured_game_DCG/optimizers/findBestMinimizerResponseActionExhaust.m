function [bestResponseActions, bestResponseValue] = findBestMinimizerResponseActionExhaust(...
            maximizerProbs, maximizerActions, lagrangeMultipliers, extraPara, valuePrecision)
    [nDoc, nRel] = size(lagrangeMultipliers);
    nRel_real = nRel; % Assuming lagrangeMultipliers have been preprocessed
    minimizerActions = de2bi(0:nRel_real^nDoc-1,[],nRel_real)';
    scoreMatrix = buildScoreMatrix(minimizerActions, maximizerActions, lagrangeMultipliers, extraPara);
    minimizerResponseValues = maximizerProbs' *scoreMatrix;
    bestResponseValue = min(minimizerResponseValues);
    bestResponseActions = minimizerActions(:,abs(minimizerResponseValues-bestResponseValue)<=valuePrecision);   
end