function [bestResponseAction, bestResponseValue] = findBestMinimizerDCGResponseAction(...
        maximizerProbs, maximizerActions, lagrangeMultipliers, IDCG)
    if isrow(maximizerProbs) % make nonZeroProbs a column, one entry for each action
        maximizerProbs = maximizerProbs';
    end
    [nDoc, nRel] = size(lagrangeMultipliers);
    bestResponseAction = zeros(nDoc,1);
    
    posScore = (log(2) ./ log(maximizerActions + 1)) * maximizerProbs / IDCG;
    
    baseRel = 1:nRel; % row vector
    baseRel = 2 .^ (baseRel - 1) - 1;
    
    bestResponseValue = 0;
    for i = 1:nDoc
        [V,bestResponseAction(i)] = min(posScore(i) * baseRel - lagrangeMultipliers(i,:));
        bestResponseValue = bestResponseValue + V;
    end
    
    bestResponseAction = bestResponseAction - 1;
end

