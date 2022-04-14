function [bestResponseAction, bestResponseValue] = findBestMinimizerRecallResponseAction(...
        maximizerProbs, maximizerActions, lagrangeMultipliers, ~)
    if isrow(maximizerProbs) % make nonZeroProbs a column, one entry for each action
        maximizerProbs = maximizerProbs';
    end
    if isrow(lagrangeMultipliers) % let lagrangeMultipliers be a column
        lagrangeMultipliers = lagrangeMultipliers';
    end
    
    % maximizerActions is column wise, each column is an action
    bitMarginalProbs = maximizerActions * maximizerProbs; % bitMarginalProbs is a column
    m = length(lagrangeMultipliers);
    S = 1 ./ (1:m);
    F = bitMarginalProbs * S; % each column is P_i/s_j, i.e for one s value
    F = bsxfun(@minus, F, lagrangeMultipliers);
    
    [~, sortedIndices] = sort(F); % 'ascend'(default)
    topIndices = triu(sortedIndices);
    baseIndices = triu(m * ones(m, 1) * (0 : m-1));
    topIndices = topIndices + baseIndices;
    nonZeroIndicesInMatrix = nonzeros(topIndices);
    
    bestResponseActionMatrix = zeros(m);
    bestResponseActionMatrix(nonZeroIndicesInMatrix) = 1;
    bestResponseValues = sum(bestResponseActionMatrix .* F);
    
    bestResponseAction = zeros(m, 1);
    bestResponseValue = 0;
    
    [minBestResponseValue, minIndex] = min(bestResponseValues);
    if(minBestResponseValue < bestResponseValue)
        bestResponseValue = minBestResponseValue;
        bestResponseAction = bestResponseActionMatrix(:, minIndex);
    end
end

