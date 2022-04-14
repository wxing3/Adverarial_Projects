function [bestResponseAction, bestResponseValue] = findBestMinimizerPrecisionResponseAction(...
        maximizerProbs, maximizerActions, lagrangeMultipliers, k)
    if iscolumn(maximizerProbs) % make nonZeroProbs a row, one entry for each action
        maximizerProbs = maximizerProbs';
    end
    if isrow(lagrangeMultipliers) % let lagrangeMultipliers be a column
        lagrangeMultipliers = lagrangeMultipliers';
    end
    
    % maximizerActions is column wise, each column is an action
    % subtract lagrange multipliers from each action bit
    maximizerActions = bsxfun(@minus, maximizerActions, (k * lagrangeMultipliers));
    bitMarginalProbs = maximizerProbs * maximizerActions';
    [sortedBitMarginalProbs, indices] = sort(bitMarginalProbs); % 'ascend'(default)
    % as long as the value is still negative, adding it will make the sum smaller
    topIndices = indices(sortedBitMarginalProbs < 0);
    
    m = length(lagrangeMultipliers); % m bits, i.e. number of documents
    bestResponseAction = zeros(m, 1);
    bestResponseAction(topIndices) = 1;
    bestResponseValue = sum(bitMarginalProbs(topIndices)) / k;
end

