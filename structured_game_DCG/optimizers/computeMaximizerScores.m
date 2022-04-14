function maximizerScoreRow = computeMaximizerScores(...
        maximizerAction, minimizerActions, lagrangeMultipliers,IDCG)
    global buildGameMatrix;
    maximizerScoreRow = buildGameMatrix(minimizerActions, maximizerAction, lagrangeMultipliers, IDCG);
    assert(isrow(maximizerScoreRow), '"scoreMatrix" here must be a row.');
end