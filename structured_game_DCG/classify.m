function [result] = classify(measure, testFileName, theta, useQuadFeatureValues, ...
    extraPara, biasFeatureValue, negativeClassTags, valuePrecision)
if nargin < 6
    biasFeatureValue = 0;
end
if nargin < 7
    negativeClassTags = false;
end
if nargin < 8
    valuePrecision = 10^-6;
end

isDebug = false;

initialize(measure);
global findBestMaximizerResponseAction findBestMinimizerResponseAction;
global computeScore;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[trueTags, featureMatrix] = loadDataset(['data/' testFileName], useQuadFeatureValues);
if(negativeClassTags)
    trueTags = -trueTags;
    warning('Using reversed class tags...');
end

if(biasFeatureValue ~= 0)
    featureMatrix = [featureMatrix, biasFeatureValue * ones(size(featureMatrix, 1), 1)];
end

verbose = false;

[nDocAll, numberOfTestFeatures] = size(featureMatrix);
numberOfLearnedWeights = length(theta);

if strcmpi(measure, 'precision') || strcmpi(measure, 'recall') || strcmpi(measure, 'f1')
    
    if(numberOfTestFeatures < numberOfLearnedWeights)
        warning('numberOfTestFeatures(%d) < numberOfLearnedWeights(%d)', ...
            numberOfTestFeatures, numberOfLearnedWeights);
        if(biasFeatureValue == 0)
            theta = theta(1:numberOfTestFeatures, 1);
        else % the last one is biasFeatureValue
            theta = [theta(1:(numberOfTestFeatures-1), 1); theta(end, 1)];
        end
    else
        if(numberOfTestFeatures > numberOfLearnedWeights)
            error('numberOfTestFeatures(%d) > numberOfLearnedWeights(%d)', ...
                numberOfTestFeatures, numberOfLearnedWeights);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    lagrangeMultipliers = featureMatrix * theta;
    k = floor(sum(trueTags > 0) * extraPara); % compute the k (for precision/recall@k only)
    k = min(k, length(trueTags)); % make sure max k is the total number of instances
    
    optimizer = @optimizerF;
    
    trueTags = trueTags > 0;
    
    nSample = 1;
    
elseif strcmpi(measure, 'DCG')
    
    if ischar(extraPara)
        nDocs = loadDataset(['data/' extraPara]);
    else
        nDocs = extraPara;
    end
    rowFeatureMatrix = featureMatrix;
    
    %         if rem(nRow,extraPara)~=0 || extraPara > nRow % extraPara is the number of documents in each query-document pair
    %             error('Number of documents is not compatible with the data.');
    %         end
    %         nSample = nRow/extraPara;
    
    if sum(nDocs) ~= nDocAll % extraPara is the number of documents in each query-document pair
        error('Number of documents is not compatible with the data.');
    end
    nSample = length(nDocs);
    
    nPos = 10; % Position nDCG ends.
    
    %% Modify Tags
    tagList = sort(unique(trueTags)); % Warning:
    nRel_real = length(tagList); % k is the number of all relevance levels, convert any format to relevance level start from 0 to k-1
    nRel = numberOfLearnedWeights / numberOfTestFeatures;
    
    if nRel < nRel_real
        if nRel > 1 || nRel_real > 2
            error('Unrecognizable labels appeared in test data');
        end
    elseif nRel > nRel_real
        nRel_real = nRel; % When some labels did not appear in test data
        warning('Not all labels appeared in testing data, true label mapping maybe wrong which leads to wrong NDCG results');
    end
    
    for j = 1:nRel_real
        idx = (trueTags==tagList(j));
        trueTags(idx) = ones(sum(idx),1) * (j-1);
    end
    
    %         trueTags = reshape(trueTags,[extraPara,nSample]);
    
    %% Mofiy features
    featureMatrix = cell(nSample,1); % Reshape the feature for each sample of query, document pair
    tagVector = cell(nSample,1);
    lagrangeMultipliers = cell(nSample,1);  % Pre-calculate each theta' * feature of every document for each different case (different relevance level)
    %         idealPos = zeros(nDoc,nSample);
    
    %% Compute IDCG for first n positions and overall NDCG
    IDCG = zeros(nPos+1,nSample);
    
    startIdx = 1;
    for i = 1:nSample
        featureMatrix{i} = rowFeatureMatrix(startIdx:startIdx+nDocs(i)-1,:); % Reshape the feature for each sample of query, document pair
        tagVector{i} = trueTags(startIdx:startIdx+nDocs(i)-1);
        startIdx = startIdx + nDocs(i);
        
        lagrangeMultipliers{i} = featureMatrix{i} * reshape(theta,[numberOfTestFeatures, nRel]) / nDocs(i);
        
        rel = tagVector{i};
        [~,I] = sort(rel, 'descend');
        %             [~,idealPos(:,i)] = sort(I);
        [~,idealPos] = sort(I);
        
        for j = 1:nPos
            largestIdx = (idealPos<=j);
            largestRel = rel(largestIdx);
            largestPos = idealPos(largestIdx);
            IDCG(j,i) = computeDCG(largestRel, largestPos);
        end
        IDCG(nPos+1,i) = computeDCG(rel,idealPos,1);
    end
    %%
    
    %trainingIDCG = IDCG(nPos+1,:)';
    
    trainingIDCG = ones(nSample,1); % If training use DCG rather than NDCG
    
    optimizer = @optimizerDCG;
    %optimizer = @optimizerDCGfull;
else
    fprintf('Unrecognized measure: %s.', measure);
    return;
end

%     minimizerActions = cell(i,1);
%     maximizerActions = cell(i,1);
%     minimizerProbabilities = cell(i,1);
%     maximizerProbabilities = cell(i,1);
maxMaximizerProb = cell(nSample,1);
maxMinimizerProb = cell(nSample,1);
maxProbMaximizerAction = cell(nSample,1);
maxProbMinimizerAction = cell(nSample,1);
bestMaximizerResponseAction = cell(nSample,1);
bestMinimizerResponseAction = cell(nSample,1);
quasiRobustMaximizerAction = cell(nSample,1);
quasiRobustMaximizerScore = zeros(nSample,1);
maxProbMaximizerScore = zeros(nSample,1);
maximizerScore = zeros(nSample,1);



if isDebug
    robustMaximizerAction = cell(nSample,1);
    robustMaximizerScore = zeros(nSample,1);
end

if strcmpi(measure, 'precision') || strcmpi(measure, 'recall') || strcmpi(measure, 'f1')
    maxProbMinimizerScore = zeros(nSample,1); % Minimizer cannot be used to compute score in DCG
    minimizerScore = zeros(nSample,1);
    
    maxProbMaximizerF1 = zeros(nSample,1);
    maxProbMinimizerF1 = zeros(nSample,1);
    maximizerF1 = zeros(nSample,1);
    minimizerF1 = zeros(nSample,1);
    
    maxProbMaximizerPrecision = zeros(nSample,1);
    maxProbMinimizerPrecision = zeros(nSample,1);
    maximizerPrecision = zeros(nSample,1);
    minimizerPrecision = zeros(nSample,1);
    
    maxProbMaximizerRecall = zeros(nSample,1);
    maxProbMinimizerRecall = zeros(nSample,1);
    maximizerRecall = zeros(nSample,1);
    minimizerRecall = zeros(nSample,1);
    
elseif strcmpi(measure, 'DCG')
    maxProbMaximizerNDCG_K = zeros(nPos,nSample);
    maximizerNDCG_K = zeros(nPos,nSample);
    
    quasiRobustMaximizerNDCG_K = zeros(nPos,nSample);
    if isDebug
        robustMaximizerNDCG_K = zeros(nPos,nSample);
    end
end

%%
result = {};

for i = 1:nSample
    
    if strcmpi(measure, 'precision') || strcmpi(measure, 'recall') || strcmpi(measure, 'f1')
        
        [~, minimizerActions, maximizerActions, ...
            minimizerProbabilities, maximizerProbabilities, ...
            ~, ~, flag] = optimizer(lagrangeMultipliers(:,:,i), k, valuePrecision);
        
        [maxMaximizerProb{i}, maxProbMaximizerActionIndex] = max(maximizerProbabilities);
        maxProbMaximizerAction{i} = maximizerActions(:, maxProbMaximizerActionIndex);
        
        [maxMinimizerProb{i}, maxProbMinimizerActionIndex] = max(minimizerProbabilities);
        maxProbMinimizerAction{i} = minimizerActions(:, maxProbMinimizerActionIndex);
        
        [nonZeroMinimizerProbabilities, nonZeroProbMinimizerActions] = ...
            findNonZeroProbabilitiesAndActions(minimizerProbabilities, minimizerActions, valuePrecision);
        bestMaximizerResponseAction{i} = findBestMaximizerResponseAction(...
            nonZeroMinimizerProbabilities, nonZeroProbMinimizerActions, lagrangeMultipliers(:,:,i), k);
        
        [nonZeroMaximizerProbabilities, nonZeroProbMaximizerActions] = ...
            findNonZeroProbabilitiesAndActions(maximizerProbabilities, maximizerActions, valuePrecision);
        bestMinimizerResponseAction{i} = findBestMinimizerResponseAction(...
            nonZeroMaximizerProbabilities, nonZeroProbMaximizerActions, lagrangeMultipliers(:,:,i), k);
        
        maxProbMaximizerScore(i) = computeScore(trueTags(:,i), maxProbMaximizerAction{i});
        maxProbMinimizerScore(i) = computeScore(trueTags(:,i), maxProbMinimizerAction{i});
        maximizerScore(i) = computeScore(trueTags(:,i), bestMaximizerResponseAction{i});
        minimizerScore(i) = computeScore(trueTags(:,i), bestMinimizerResponseAction{i});
        
        maxProbMaximizerF1(i) = computeF1(trueTags, maxProbMaximizerAction{i});
        maxProbMinimizerF1(i) = computeF1(trueTags(:,:,i), maxProbMinimizerAction{i});
        maximizerF1(i) = computeF1(trueTags(:,:,i), bestMaximizerResponseAction{i});
        minimizerF1(i) = computeF1(trueTags(:,:,i), bestMinimizerResponseAction{i});
        
        maxProbMaximizerPrecision(i) = computePrecision(trueTags(:,:,i), maxProbMaximizerAction{i});
        maxProbMinimizerPrecision(i) = computePrecision(trueTags(:,:,i), maxProbMinimizerAction{i});
        maximizerPrecision(i) = computePrecision(trueTags(:,:,i), bestMaximizerResponseAction{i});
        minimizerPrecision(i) = computePrecision(trueTags(:,:,i), bestMinimizerResponseAction{i});
        
        maxProbMaximizerRecall(i) = computeRecall(trueTags(:,:,i), maxProbMaximizerAction{i});
        maxProbMinimizerRecall(i) = computeRecall(trueTags(:,:,i), maxProbMinimizerAction{i});
        maximizerRecall(i) = computeRecall(trueTags(:,:,i), bestMaximizerResponseAction{i});
        minimizerRecall(i) = computeRecall(trueTags(:,:,i), bestMinimizerResponseAction{i});
        
    elseif strcmpi(measure, 'DCG')
        
        if nRel == 1
            lagrangeMultipliers{i} = [zeros(length(lagrangeMultipliers{i}),1) lagrangeMultipliers{i}]; % For binary case, only documents with label 1 has valid features
        end
        
        [~, minimizerActions, maximizerActions, ...
            minimizerProbabilities, maximizerProbabilities, ...
            ~, ~, flag] = optimizer(lagrangeMultipliers{i}, trainingIDCG(i), valuePrecision, verbose);
        
        [maxMaximizerProb{i}, maxProbMaximizerActionIndex] = max(maximizerProbabilities);
        maxProbMaximizerAction{i} = maximizerActions(:, maxProbMaximizerActionIndex);
        
        [maxMinimizerProb{i}, maxProbMinimizerActionIndex] = max(minimizerProbabilities);
        maxProbMinimizerAction{i} = minimizerActions(:, maxProbMinimizerActionIndex);
        
        [nonZeroMinimizerProbabilities, nonZeroProbMinimizerActions] = ...
            findNonZeroProbabilitiesAndActions(minimizerProbabilities, minimizerActions, valuePrecision);
        bestMaximizerResponseAction{i} = findBestMaximizerResponseAction(...
            nonZeroMinimizerProbabilities, nonZeroProbMinimizerActions, lagrangeMultipliers{i}, trainingIDCG(i));
        
        [nonZeroMaximizerProbabilities, nonZeroProbMaximizerActions] = ...
            findNonZeroProbabilitiesAndActions(maximizerProbabilities, maximizerActions, valuePrecision);
        bestMinimizerResponseAction{i} = findBestMinimizerResponseAction(...
            nonZeroMaximizerProbabilities, nonZeroProbMaximizerActions, lagrangeMultipliers{i}, trainingIDCG(i));
        
        [quasiRobustMaximizerAction{i},~] =  predict(lagrangeMultipliers{i}, trainingIDCG(i), valuePrecision, false, false);
        if isDebug
            [robustMaximizerAction{i},~] =  predictExhaust(lagrangeMultipliers{i}, trainingIDCG(i), false, false);
        end
        rel = tagVector{i};
        
        maxProbMaximizerScore(i) = computeScore(rel, maxProbMaximizerAction{i},IDCG(nPos+1,i));
        maximizerScore(i) = computeScore(rel, bestMaximizerResponseAction{i},IDCG(nPos+1,i));
        quasiRobustMaximizerScore(i) = computeScore(rel, quasiRobustMaximizerAction{i},IDCG(nPos+1,i));
        if isDebug
            robustMaximizerScore(i) = computeScore(rel, robustMaximizerAction{i},IDCG(nPos+1,i));
        end
        
        for j = 1:nPos
            largestIdx = (maxProbMaximizerAction{i}<=j);
            largestRel = rel(largestIdx);
            largestPos = maxProbMaximizerAction{i}(largestIdx);
            maxProbMaximizerNDCG_K(j,i) = computeDCG(largestRel, largestPos, IDCG(j,i));
            
            largestIdx = (bestMaximizerResponseAction{i}<=j);
            largestRel = rel(largestIdx);
            largestPos = bestMaximizerResponseAction{i}(largestIdx);
            maximizerNDCG_K(j,i) = computeDCG(largestRel, largestPos, IDCG(j,i));
            
            largestIdx = (quasiRobustMaximizerAction{i}<=j);
            largestRel = rel(largestIdx);
            largestPos = quasiRobustMaximizerAction{i}(largestIdx);
            quasiRobustMaximizerNDCG_K(j,i) =  computeDCG(largestRel, largestPos, IDCG(j,i));
            
            if isDebug
                largestIdx = (robustMaximizerAction{i}<=j);
                largestRel = rel(largestIdx);
                largestPos = robustMaximizerAction{i}(largestIdx);
                robustMaximizerNDCG_K(j,i) =  computeDCG(largestRel, largestPos, IDCG(j,i));
            end
        end
    end
end

%% record results
result.maxMaximizerProb = maxMaximizerProb;
result.maxMinimizerProb = maxMinimizerProb;
result.maxProbMaximizerAction = maxProbMaximizerAction;
result.maxProbMinimizerAction = maxProbMinimizerAction;
result.bestMaximizerResponseAction = bestMaximizerResponseAction;
result.bestMinimizerResponseAction = bestMinimizerResponseAction;
result.quasiRobustMaximizerAction = quasiRobustMaximizerAction;
if isDebug
    result.robustMaximizerAction = robustMaximizerAction;
end

result.maxProbMaximizerScore = maxProbMaximizerScore;
result.maximizerScore = maximizerScore;
result.quasiRobustMaximizerScore = quasiRobustMaximizerScore;
if isDebug
    result.robustMaximizerScore = robustMaximizerScore;
end

validScore = maxProbMaximizerScore(~isnan(maxProbMaximizerScore));
result.avg_maxProbMaximizerScore = mean(validScore);

validScore = maximizerScore(~isnan(maximizerScore));
result.avg_maximizerScore = mean(validScore);

validScore = quasiRobustMaximizerScore(~isnan(quasiRobustMaximizerScore));
result.avg_quasiRobustMaximizerScore = mean(validScore);

if isDebug
    validScore = robustMaximizerScore(~isnan(robustMaximizerScore));
    result.avg_robustMaximizerScore = mean(validScore);
end

if strcmpi(measure, 'precision') || strcmpi(measure, 'recall') || strcmpi(measure, 'f1')
    result.maxProbMinimizerScore = maxProbMinimizerScore;
    result.minimizerScore = minimizerScore;
    
    result.avg_maxProbMinimizerScore = mean(maxProbMinimizerScore);
    result.avg_minimizerScore = mean(minimizerScore);
    
    result.maxProbMaximizerF1 = maxProbMaximizerF1;
    result.maxProbMinimizerF1 = maxProbMinimizerF1;
    result.maximizerF1 = maximizerF1;
    result.minimizerF1 = minimizerF1;
    
    result.avg_maxProbMaximizerF1 = mean(maxProbMaximizerF1);
    result.avg_maxProbMinimizerF1 = mean(maxProbMinimizerF1);
    result.avg_maximizerF1 = mean(maximizerF1);
    result.avg_minimizerF1 = mean(minimizerF1);
    
    result.maxProbMaximizerPrecision = maxProbMaximizerPrecision;
    result.maxProbMinimizerPrecision = maxProbMinimizerPrecision;
    result.maximizerPrecision = maximizerPrecision;
    result.minimizerPrecision = minimizerPrecision;
    
    result.avg_maxProbMaximizerPrecision = mean(maxProbMaximizerPrecision);
    result.avg_maxProbMinimizerPrecision = mean(maxProbMinimizerPrecision);
    result.avg_maximizerPrecision = mean(maximizerPrecision);
    result.avg_minimizerPrecision = mean(minimizerPrecision);
    
    result.maxProbMaximizerRecall = maxProbMaximizerRecall;
    result.maxProbMinimizerRecall = maxProbMinimizerRecall;
    result.maximizerRecall = maximizerRecall;
    result.minimizerRecall = minimizerRecall;
    
    result.avg_maxProbMaximizerRecall = mean(maxProbMaximizerRecall);
    result.avg_maxProbMinimizerRecall = mean(maxProbMinimizerRecall);
    result.avg_maximizerRecall = mean(maximizerRecall);
    result.avg_minimizerRecall = mean(minimizerRecall);
    
elseif strcmpi(measure, 'DCG')
    result.maxProbMaximizerNDCG_K = maxProbMaximizerNDCG_K;
    result.maximizerNDCG_K = maximizerNDCG_K;
    result.quasiRobustMaximizerNDCG_K = quasiRobustMaximizerNDCG_K;
    if isDebug
        result.robustMaximizerNDCG_K = robustMaximizerNDCG_K;
    end
    
    avg_maxProbMaximizerNDCG_K = zeros(nPos,1);
    avg_maximizerNDCG_K  = zeros(nPos,1);
    avg_quasiRobustMaximizerNDCG_K  = zeros(nPos,1);
    if isDebug
        avg_robustMaximizerNDCG_K  = zeros(nPos,1);
    end
    
    for j = 1:nPos
        validScore = maxProbMaximizerNDCG_K(j,:);
        validScore = validScore(~isnan(validScore));
        avg_maxProbMaximizerNDCG_K(j) = mean(validScore);
        
        validScore = maximizerNDCG_K(j,:);
        validScore = validScore(~isnan(validScore));
        avg_maximizerNDCG_K(j) = mean(validScore);
        
        validScore = quasiRobustMaximizerNDCG_K(j,:);
        validScore = validScore(~isnan(validScore));
        avg_quasiRobustMaximizerNDCG_K(j) = mean(validScore);
        
        if isDebug
            validScore = robustMaximizerNDCG_K(j,:);
            validScore = validScore(~isnan(validScore));
            avg_robustMaximizerNDCG_K(j) = mean(validScore);
        end
    end
    
    result.avg_maxProbMaximizerNDCG_K = avg_maxProbMaximizerNDCG_K;
    result.avg_maximizerNDCG_K = avg_maximizerNDCG_K;
    result.avg_quasiRobustMaximizerNDCG_K = avg_quasiRobustMaximizerNDCG_K;
    if isDebug
        result.avg_robustMaximizerNDCG_K = avg_robustMaximizerNDCG_K;
    end
end
end
