function [bestResponseAction, bestResponseValue] = findBestMaximizerDCGResponseAction(...
        minimizerProbs, minimizerActions, lagrangeMultipliers, IDCG)
    if isrow(minimizerProbs) % make minimizerProbs a column, one entry for each action
        minimizerProbs = minimizerProbs';
    end
    
    [nDoc,nAction] = size(minimizerActions);
    
    relScores = (2.^minimizerActions - 1) * minimizerProbs;
    [~,I] = sort(relScores, 'descend');
    [~,bestResponseAction] = sort(I);
    bestResponseValue = relScores * log(2) ./ log(bestResponseAction+1);
    bestResponseValue = sum(bestResponseValue) / IDCG;
    
    minimizerActionValues = zeros(1,nAction);
    
    for i = 1:nAction
        for j = 1:nDoc
            minimizerActionValues(i) = minimizerActionValues(i) + lagrangeMultipliers(j,minimizerActions(j,i)+1); 
        end
    end
    
    bestResponseValue = bestResponseValue - minimizerActionValues * minimizerProbs;
end
