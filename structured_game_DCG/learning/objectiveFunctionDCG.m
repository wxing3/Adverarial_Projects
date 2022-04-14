function [score, gradient, errorFlag] = objectiveFunctionDCG(...
    theta, trueFeatureValues, featureMatrix, IDCG, regularization, valuePrecision,verbose)

global optimizer;

nSample = length(featureMatrix);
nFeature = size(featureMatrix{1},2);

nRel = length(theta) / nFeature; % For binary case, nRel is 1 here while there are actually 2 differnet relevance levels

lagrangeMultipliers = cell(nSample,1);  % Pre-calculate each theta' * feature of every document for each different case (different relevance level)

nDocs = zeros(nSample,1);
for i = 1:nSample
    [nDocs(i),~] = size(featureMatrix{i});
    lagrangeMultipliers{i} = featureMatrix{i} * reshape(theta,[nFeature, nRel])/ nDocs(i);
    %         if nRel == 1
    %             lagrangeMultipliers{i} = [zeros(sum(nDocs),1) lagrangeMultipliers{i}]; % For binary case, only documents with label 1 has valid features
    %         end
end



% lagrangeMultipliers = lagrangeMultipliers ./ abs(lagrangeMultipliers);
gradient = 0;
rawScore = 0;
invalidSample = 0;

for i = 1:nSample
    
    % Since our goal is to maximize the score, the minmizer got from the
    % game is the condictional distribution of the adversarial.
    

    
   [~, minimizerActions, ~, minimizerProbabilities, ~, ~, maximizerValue,errorFlag] =...
                optimizer(lagrangeMultipliers{i}, IDCG(i), valuePrecision, verbose);

            %% Debugging using full game matrix
%     gameSolver1 = @optimizerDCG;
%     gameSolver2 = @optimizerDCGfull;
%     [scoreMatrix, minimizerActions, maximizerActions, minimizerProbabilities, maximizerProbabilities, minimizerValue, maximizerValue,errorFlag] =...
%         gameSolver1(lagrangeMultipliers{i}, IDCG(i), valuePrecision, verbose);
%     [scoreMatrix_f, minimizerActions_f, maximizerActions_f, minimizerProbabilities_f, maximizerProbabilities_f, minimizerValue_f, maximizerValue_f,errorFlag_f] =...
%         gameSolver2(lagrangeMultipliers{i}, IDCG(i), valuePrecision, verbose);
%     if abs(maximizerValue-minimizerValue) > valuePrecision || abs(maximizerValue_f-minimizerValue_f) > valuePrecision
%         warning('Solving maximizer and minimizer results in different game value.');
%         fprintf('Maximizer game value:%0.8f, minimiazer game value:%0.8f for double oracle.\n',maximizerValue, minimizerValue);
%         fprintf('Maximizer game value:%0.8f, minimiazer game value:%0.8f for full game.\n',maximizerValue_f, minimizerValue_f);
%     end
%     if abs(maximizerValue-maximizerValue_f) > valuePrecision || abs(minimizerValue-minimizerValue_f) > valuePrecision
%         warning('Solving by double oracle and by full matrix results in different game value.');
%         fprintf('Maximizer game value:%0.8f, minimiazer game value:%0.8f for double oracle.\n',maximizerValue, minimizerValue);
%         fprintf('Maximizer game value:%0.8f, minimiazer game value:%0.8f for full game.\n',maximizerValue_f, minimizerValue_f);
%     end
%     [~,nMaxActions] = size(maximizerActions);
%     [~,nMinActions] = size(minimizerActions);
%     [~,nMaxActions_f] = size(maximizerActions_f);
%     [~,nMinActions_f] = size(minimizerActions_f);
%     if maximizerValue>valuePrecision || minimizerValue>valuePrecision || maximizerValue_f>valuePrecision || minimizerValue_f>valuePrecision
%         if nMaxActions~=nMaxActions_f
%             warning('Solving by double oracle and by full matrix results in different number of valid maximizer actions: %d and %d.',nMaxActions,nMaxActions_f);
%         end
%         if nMinActions~=nMinActions_f
%             warning('Solving by double oracle and by full matrix results in different number of valid minimizer actions: %d and %d.',nMinActions,nMinActions_f);
%         end
%     else
%         fprintf('Non-trivial game achieved.\n');
%     end

    %% ignore corrupted results
    if errorFlag ~=0
        invalidSample = invalidSample + 1;
        continue;
    end
    
    %% score
    rawScore = rawScore - maximizerValue;  % negative, because the objective is to maximize score, but the L-BFGS we use is to minimize the value.
    
    
    %% gradient
    nonZeroProbabilityLogicalIndices = minimizerProbabilities > valuePrecision;
    nonZeroProbActions = minimizerActions(:, nonZeroProbabilityLogicalIndices);
    nonZeroProbabilities = minimizerProbabilities(nonZeroProbabilityLogicalIndices);
    nonZeroProbabilities = nonZeroProbabilities / sum(nonZeroProbabilities);
    
    expectedFeatureValues = zeros(nRel * nFeature, 1); % Sum the feature of the documents which share the same relevance level, and concatenate them to a single vector for each sample.
    
    for l = 1:length(nonZeroProbabilities)
        actionFeatureValues = zeros(nRel * nFeature, 1);
        if nRel == 1
            actionFeatureValues = featureMatrix{i}' * (nonZeroProbActions(:,l) == 1 ) / nDocs(i); % For binary case, only documens with label 1 has non-zero features, for label 0 it is 0
        else
            for j = 1:nRel
                actionFeatureValues((j-1)*nFeature+1:j*nFeature) = featureMatrix{i}' * (nonZeroProbActions(:,l)==(j-1)) / nDocs(i); % Assumming relevance level start from 0 to k-1
            end
        end
        expectedFeatureValues = expectedFeatureValues + actionFeatureValues * nonZeroProbabilities(l);
    end
    gradient = gradient + expectedFeatureValues;
    
end

if invalidSample > 0
    errorFlag = 1;
end

if invalidSample >= nSample
    errorFlag = 2;
    score = NaN;
    gradient = NaN;
    error('All the minimax problem of the samples encounter some errors.');
end

gradient = gradient / (nSample - invalidSample) - trueFeatureValues; %Assuming each sample only appear once.
rawScore = rawScore / (nSample - invalidSample) - theta' * trueFeatureValues;
score = rawScore;


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
    score = score + regularizationValue;
    gradient = gradient + regularizationGradient;
end

%%
if verbose
    fprintf('\t>>>> ''objectiveFunction''\trawScore=%.6f\tregularization[%s]=%.6f\tscore=%.6f\n', ...
        rawScore, regularization.method, regularizationValue, score);
end
end

