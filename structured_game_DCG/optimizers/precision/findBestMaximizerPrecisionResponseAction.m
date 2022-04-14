function [bestResponseAction, bestResponseValue] = findBestMaximizerPrecisionResponseAction(...
        minimizerProbs, minimizerActions, lagrangeMultipliers, k)
    if iscolumn(minimizerProbs) % make minimizerProbs a row, one entry for each action
        minimizerProbs = minimizerProbs';
    end
    if isrow(lagrangeMultipliers) % let lagrangeMultipliers be a column
        lagrangeMultipliers = lagrangeMultipliers';
    end
    
    % minimizerActions is column wise, each column is an action
    bitMarginalProbs = minimizerProbs * minimizerActions';
    [sortedBitMarginalProbs, sortedIndices] = sort(bitMarginalProbs, 'descend');
    topIndices = sortedIndices(1:k);
    topBitMarginalProbs = sortedBitMarginalProbs(1:k);
    
    m = length(lagrangeMultipliers); % m bits, i.e. number of documents
    bestResponseAction = zeros(m, 1);
    bestResponseAction(topIndices) = 1;
    bestResponseValue = sum(topBitMarginalProbs) / k; % expectation
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % modify the expected value based on lagrange multipliers
    lagrangeMultipliersSum = bitMarginalProbs * lagrangeMultipliers;
    bestResponseValue = bestResponseValue - lagrangeMultipliersSum;
end

