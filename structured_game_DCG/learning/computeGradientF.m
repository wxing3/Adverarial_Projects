function [gradient] = computeGradientF(...
        trueFeatureValues, minimizerActions, minimizerProbabilities, ...
        featureMatrix, theta, regularization, valuePrecision)
    
    nonZeroProbabilityLogicalIndices = minimizerProbabilities > valuePrecision;
    nonZeroProbActions = minimizerActions(:, nonZeroProbabilityLogicalIndices);
    nonZeroProbabilities = minimizerProbabilities(nonZeroProbabilityLogicalIndices);
    
    expectedFeatureValueMatrix = featureMatrix' * nonZeroProbActions * diag(nonZeroProbabilities);
    expectedFeatureValues = sum(expectedFeatureValueMatrix, 2); % sum each row
    
    % for minimize objective value; otherwise reverse the order
    gradient = expectedFeatureValues - trueFeatureValues;
    
    %% regularization 
    if(regularization.parameter ~= 0)
        if(strcmpi(regularization.method, 'L1'))
            gradient = gradient + regularization.parameter * sign(theta);
        elseif(strcmpi(regularization.method, 'L2'))
            gradient = gradient + 2 * regularization.parameter * theta;
        else
            error('Unknown regularization method ''%s''', regularization.method);
        end
    end
end

