function [bestResponseAction, bestResponseValue] = findBestMaximizerRecallResponseAction(...
        minimizerProbs, minimizerActions, lagrangeMultipliers, k)
    if iscolumn(minimizerProbs) % make minimizerProbs a row, one entry for each action
        minimizerProbs = minimizerProbs';
    end
    if isrow(lagrangeMultipliers) % let lagrangeMultipliers be a column
        lagrangeMultipliers = lagrangeMultipliers';
    end
    
    m = length(lagrangeMultipliers);
    [P, ~] = buildMatrixP(minimizerProbs, minimizerActions, m);
    S = 1 ./ (1:m)';
    F = P * S;
    
    [sortedWeightedBitMarginalProbs, sortedIndices] = sort(F, 'descend');
    topIndices = sortedIndices(1:k);
    topBitMarginalProbs = sortedWeightedBitMarginalProbs(1:k);
    
    bestResponseAction = zeros(m, 1);
    bestResponseAction(topIndices) = 1;
    bestResponseValue = sum(topBitMarginalProbs); % expectation
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % modify the expected value based on lagrange multipliers
    lagrangeMultipliersSum = minimizerProbs * minimizerActions' * lagrangeMultipliers;
    bestResponseValue = bestResponseValue - lagrangeMultipliersSum;
end
