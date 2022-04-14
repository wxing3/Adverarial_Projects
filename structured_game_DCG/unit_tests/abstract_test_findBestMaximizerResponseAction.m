function [bestMaximizerActions, bestExpectationValue, minimizerProbs, minimizerActions, lagrangeMultipliers] = ...
        abstract_test_findBestMaximizerResponseAction(m, computeScoreFunction, k, randomLowerBound, randomUpperBound)
    if(nargin < 4)
        randomLowerBound = -1;
    end
    if(nargin < 5)
        randomUpperBound = 1;
    end
    
    clear gfm; % remove the persistent variable W
    
    rng('shuffle');
    lagrangeMultipliers = randomLowerBound + (randomUpperBound - randomLowerBound) .* rand(1, m);
    
    minimizerActions = [];
    for i = 0:2^m-1
        minimizerActions = [minimizerActions, dec2binvec(i, m)']; %#ok<AGROW>
    end
    minimizerProbs = rand(1,2^m);
    minimizerProbs = minimizerProbs / sum(minimizerProbs);
    
    if(nargin < 3) % no k
        possibleMaximizerActions = minimizerActions;
    else
        possibleMaximizerActions = minimizerActions(:, sum(minimizerActions) == k);
    end
    
    scoreExpectations = zeros(1, size(possibleMaximizerActions, 2));
    for maximizerIndex = 1:size(possibleMaximizerActions, 2);
        possibleMaximizerAction = possibleMaximizerActions(:, maximizerIndex);
        
        scoreExpectation = 0;
        for minimizerIndex = 1:size(minimizerActions, 2);
            minimizerAction = minimizerActions(:, minimizerIndex);
            minimizerProb = minimizerProbs(minimizerIndex);
            
            % minimizerAction is trueTags, while maximizerAction is that we want to predict
            score = computeScoreFunction(minimizerAction, possibleMaximizerAction);
            lagrangeMultipliersSum = sum(lagrangeMultipliers(find(minimizerAction))); %#ok<FNDSB>
            weightedScore = minimizerProb * (score - lagrangeMultipliersSum);
            
            scoreExpectation = scoreExpectation + weightedScore;
        end
        scoreExpectations(maximizerIndex) = scoreExpectation;
    end
    
    bestExpectationValue = max(scoreExpectations);
    bestMaximizerActions = possibleMaximizerActions(:, (scoreExpectations == bestExpectationValue));
end
