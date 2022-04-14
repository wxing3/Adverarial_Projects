function [h_star, expectation_star] = findBestMaximizerF1ResponseAction(...
        nonZeroProbs, nonZeroProbActions, lagrangeMultipliers, ~)
    if iscolumn(nonZeroProbs)
        nonZeroProbs = nonZeroProbs';
    end
    if isrow(lagrangeMultipliers)
        lagrangeMultipliers = lagrangeMultipliers';
    end
    
    m = length(lagrangeMultipliers);
    [P, p_0] = buildMatrixP(nonZeroProbs, nonZeroProbActions, m); % slow!
    [h_star, expectation_star] = gfm(P, p_0, zeros(m, 1), true); % no lagrangeMultiplier
    
    % modify the expected value based on lagrange multipliers
    lagrangeMultipliersSum = nonZeroProbs * nonZeroProbActions' * lagrangeMultipliers;
    expectation_star = expectation_star - lagrangeMultipliersSum;
end

