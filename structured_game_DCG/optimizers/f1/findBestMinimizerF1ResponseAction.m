function [h_star, expectation_star] = findBestMinimizerF1ResponseAction(...
        nonZeroProbs, nonZeroProbActions, lagrangeMultipliers, ~)
    [P, p_0] = buildMatrixP(nonZeroProbs, nonZeroProbActions, length(lagrangeMultipliers)); % slow!
    [h_star, expectation_star] = gfm(P, p_0, lagrangeMultipliers, false);
end
