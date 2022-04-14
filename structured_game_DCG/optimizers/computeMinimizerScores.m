function minimizerScoreColumn = computeMinimizerScores(...
        minimizerAction, maximizerActions, lagrangeMultipliers,extraPara)
    global buildGameMatrix;
    minimizerScoreColumn = buildGameMatrix(minimizerAction, maximizerActions, lagrangeMultipliers, extraPara);
    assert(iscolumn(minimizerScoreColumn), '"scoreMatrix" here must be a column.');
end