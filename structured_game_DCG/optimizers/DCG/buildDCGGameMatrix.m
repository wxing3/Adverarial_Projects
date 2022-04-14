function scoreMatrix = buildDCGGameMatrix(minimizerActions, maximizerActions, lagrangeMultipliers, IDCG)
    global computeScore;
    
    % number of columns is the count of actions.
    numOfMinimizerActions = size(minimizerActions, 2);
    numOfMaximizerActions = size(maximizerActions, 2);
    
    scoreMatrix = zeros(numOfMaximizerActions, numOfMinimizerActions);
    
    for minimizerIndex = 1:numOfMinimizerActions % column index in scoreMatrix
        minimizerAction = minimizerActions(:, minimizerIndex);
        
        % The same minimizer action have the same lagrangeMultiplierSum
        lagrangeMultiplierSum = 0;
        for i = 1:size(lagrangeMultipliers,1)
            lagrangeMultiplierSum = lagrangeMultiplierSum + lagrangeMultipliers(i,minimizerAction(i)+1);
        end
        
        for maximizerIndex = 1:numOfMaximizerActions % row index in scoreMatrix
            maximizerAction = maximizerActions(:, maximizerIndex);
          
            score = computeScore(minimizerAction, maximizerAction, IDCG);
            
            % 'lagrangeMultipliers' are subtracted from each minimizer column.
            scoreMatrix(maximizerIndex, minimizerIndex) = score - lagrangeMultiplierSum;
        end
    end
end