function [bestMinimizerActions, bestExpectationValue, maximizerProbs, maximizerActions, lagrangeMultipliers] = ...
        abstract_test_findBestMinimizerResponseAction(m, computeScoreFunction, k, randomLowerBound, randomUpperBound)
    if(nargin < 4)
        randomLowerBound = -1;
    end
    if(nargin < 5)
        randomUpperBound = 1;
    end
    
    clear gfm; % remove the persistent variable W
    
    rng('shuffle');
    lagrangeMultipliers = randomLowerBound + (randomUpperBound - randomLowerBound) .* rand(1, m);
    
    possibleMinimizerActions = [];
    for i = 0:2^m-1
        possibleMinimizerActions = [possibleMinimizerActions, dec2binvec(i, m)']; %#ok<AGROW>
    end
    
    if(nargin < 3 || k < 0) % no k
        maximizerActions = possibleMinimizerActions;
    else
        maximizerActions = possibleMinimizerActions(:, sum(possibleMinimizerActions) == k);
    end
    maximizerProbs = rand(1,size(maximizerActions, 2));
    maximizerProbs = maximizerProbs / sum(maximizerProbs);
    
    scoreExpectations = zeros(1, 2^m);
    for minimizerIndex = 1:size(possibleMinimizerActions, 2);
        possibleMinimizerAction = possibleMinimizerActions(:, minimizerIndex);
        lagrangeMultipliersSum = sum(lagrangeMultipliers(find(possibleMinimizerAction))); %#ok<FNDSB>
        
        scoreExpectation = 0;
        for maximizerIndex = 1:size(maximizerActions, 2);
            maximizerAction = maximizerActions(:, maximizerIndex);
            maximizerProb = maximizerProbs(maximizerIndex);
            
            % minimizerAction is trueTags, while maximizerAction is that we want to predict
            score = computeScoreFunction(possibleMinimizerAction, maximizerAction);
            weightedScore = maximizerProb * (score - lagrangeMultipliersSum);
            
            scoreExpectation = scoreExpectation + weightedScore;
        end
        scoreExpectations(minimizerIndex) = scoreExpectation;
    end
    
    bestExpectationValue = min(scoreExpectations);
    bestMinimizerActions = possibleMinimizerActions(:, (scoreExpectations == bestExpectationValue));
end
