function [gradient] = computeGradientDCG(...
        trueFeatureValues, minimizerActions, minimizerProbabilities, ...
        featureMatrix, theta, regularization, valuePrecision)
    
    nonZeroProbabilityLogicalIndices = minimizerProbabilities > valuePrecision;
    nonZeroProbActions = minimizerActions(:, nonZeroProbabilityLogicalIndices);
    nonZeroProbabilities = minimizerProbabilities(nonZeroProbabilityLogicalIndices);
    nonZeroProbabilities = nonZeroProbabilities / sum(nonZeroProbabilities);

    expectedFeatureValues = zeros(k * nColumn, 1); % Sum the feature of the documents which share the same relevance level, and concatenate them to a single vector for each sample.

    for l = 1:length(nonZeroProbabilities)
        actionFeatureValues = zeros(k * nColumn, 1);
        for j = 1:k
            actionFeatureValues((j-1)*nColumn+1:j*nColumn) = featureMatrix' * (nonZeroProbActions(:,l)==(j-1)); % Assumming relevance level start from 0 to k-1
        end
        expectedFeatureValues = expectedFeatureValues + actionFeatureValues * nonZeroProbabilities(l);
    end
    
    % for minimize objective value; otherwise reverse the order
%     gradient = expectedFeatureValues - trueFeatureValues;
    gradient = expectedFeatureValues - trueFeatureValues;
    
    %% regularization 
    if(regularization.parameter ~= 0)
        if(strcmpi(regularization.method, 'L1'))
            gradient = gradient + regularization.parameter * sign(theta);
        elseif(strcmpi(regularization.method, 'L2'))
            gradient = gradient +  2 * regularization.parameter * theta;
        else
            error('Unknown regularization method ''%s''', regularization.method);
        end
    end
end

