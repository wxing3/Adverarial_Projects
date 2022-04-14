function [score, gradient, rawScore, errorFlag] = objectiveSampleFunctionDCG(...
    i, theta, trueFeatureValues, featureMatrix, IDCG, regularization, valuePrecision,verbose)
    
    nFeature = size(featureMatrix{1},2);
    nRel = length(theta) / nFeature;
    lagrangeMultipliers = featureMatrix{i} * reshape(theta,[nFeature, nRel]);

    % Since our goal is to maximize the score, the minmizer got from the
    % game is the condictional distribution of the adversarial.
    [~, minimizerActions, ~, minimizerProbabilities, ~, ~, maximizerValue,errorFlag] =...
        optimizerDCG(lagrangeMultipliers, IDCG(i), valuePrecision,verbose);

    %% score
    rawScore =  - maximizerValue - theta' * trueFeatureValues;  % negative, because the objective is to maximize score, but the L-BFGS we use is to minimize the value.


    %% gradient
    nonZeroProbabilityLogicalIndices = minimizerProbabilities > valuePrecision;
    nonZeroProbActions = minimizerActions(:, nonZeroProbabilityLogicalIndices);
    nonZeroProbabilities = minimizerProbabilities(nonZeroProbabilityLogicalIndices);
    nonZeroProbabilities = nonZeroProbabilities / sum(nonZeroProbabilities);

    expectedFeatureValues = zeros(nRel * nFeature, 1); % Sum the feature of the documents which share the same relevance level, and concatenate them to a single vector for each sample.

    for l = 1:length(nonZeroProbabilities)
        actionFeatureValues = zeros(nRel * nFeature, 1);
        for j = 1:nRel
            actionFeatureValues((j-1)*nFeature+1:j*nFeature) = featureMatrix{i}' * (nonZeroProbActions(:,l)==(j-1)); % Assumming relevance level start from 0 to k-1
        end
        expectedFeatureValues = expectedFeatureValues + actionFeatureValues * nonZeroProbabilities(l);
    end
    gradient = expectedFeatureValues - trueFeatureValues;

    %% regularization
    regularizationValue = 0;
    if(regularization.parameter ~= 0)
        if(strcmpi(regularization.method, 'L1'))
            regularizationValue = regularization.parameter * norm(theta, 1);
            regularizationGradient = regularization.parameter * sign(theta);
        elseif(strcmpi(regularization.method, 'L2'))
            regularizationValue = regularization.parameter * (theta' * theta);
            regularizationGradient = 2 * regularization.parameter * theta;
        else
            error('Unknown regularization method ''%s''', regularization.method);
        end
        % since we want to minimize the value, plus regularization to penalize the large weights
        score = rawScore + regularizationValue;
        gradient = gradient + regularizationGradient;
    end

    %%
    if verbose
        fprintf('\t>>>> ''objectiveFunction''\trawScore=%.6f\tregularization[%s]=%.6f\tscore=%.6f\n', ...
            rawScore, regularization.method, regularizationValue, score);
    end