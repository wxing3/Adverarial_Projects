function [score, theta, output] = learn(measure, learningFileName, learnInitialTheta, ...
        useQuadFeatureValues, extraPara, regularization, biasFeatureValue, negativeClassTags, valuePrecision)
    if nargin < 6
        regularization = struct('method', 'L2', 'parameter', 0);
    end
    if nargin < 7
        biasFeatureValue = 0;
    end
    if nargin < 8
        negativeClassTags = false;
    end
    if nargin < 9
        valuePrecision = 10^-6;
    end
    
    initialize(measure);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [trueTags, featureMatrix] = loadDataset(['data/' learningFileName], useQuadFeatureValues);
    
    
    if(negativeClassTags)
        trueTags = -trueTags;
        warning('Using reversed class tags...');
    end
    
    if(biasFeatureValue ~= 0)
        featureMatrix = [featureMatrix, biasFeatureValue * ones(size(featureMatrix, 1), 1)];
    end
    
    numberOfFeatures = size(featureMatrix, 2);
    
    if length(learnInitialTheta) > 1
        theta = learnInitialTheta;
    elseif learnInitialTheta
        theta = learnInitialWeightsFromLR(trueTags, featureMatrix);
        [nRow, nColume] = size(theta);
        theta = reshape(theta,[nRow*nColume,1]); % For multiclass results, concatenate the parameters of each class to one vector
    else
        theta = zeros(numberOfFeatures, 1) + 10^-8;
    end
    
    if strcmpi(measure, 'precision') || strcmpi(measure, 'recall') || strcmpi(measure, 'f1') 
        trueTags = trueTags > 0;
        k = floor(sum(trueTags) * extraPara); % compute the k (for precision/recall@k only)
        k = min(k, length(trueTags)); % make sure max k is the total number of instances

        trueFeatureValues = featureMatrix' * trueTags; % A column vector
        objectiveFunction = @objectiveFunctionF;
        extraPara = k;
        nSample = 1;
    elseif strcmpi(measure, 'DCG')
        
        if ischar(extraPara)
            nDocs = loadDataset(['data/' extraPara]);
        else
            nDocs = extraPara;
        end
        rowFeatureMatrix = featureMatrix;
        [nRow, nColumn] = size(rowFeatureMatrix);
%         if rem(nRow,extraPara)~=0 || extraPara > nRow % extraPara is the number of documents in each query-document pair
%             error('Number of documents is not compatible with the data.');
%         end
%         nSample = nRow/extraPara;

        if sum(nDocs) ~= nRow % extraPara is the number of documents in each query-document pair
            error('Number of documents is not compatible with the data.');
        end
        nSample = length(nDocs);


        tagList = sort(unique(trueTags));
        nRel = length(tagList); % k is the number of all relevance levels, convert any format to relevance level start from 0 to k-1
        for j = 1:nRel
           idx = (trueTags==tagList(j));
           trueTags(idx) = ones(sum(idx),1) * (j-1);
        end      
        
        trueFeatureValues = zeros(nRel * nColumn,1); %Since the expectation on samples is actually summation, we can add the true features all together here
        for j = 1:nRel
            trueFeatureValues((j-1)*nColumn+1:j*nColumn) = (rowFeatureMatrix(:,:))' * (trueTags==(j-1)); 
        end        
        trueFeatureValues = trueFeatureValues / nSample;
        
        featureMatrix = cell(nSample,1); 
        tagVector = cell(nSample,1);
        %IDCG = zeros(nSample,1);
        
        startIdx = 1;
        for i = 1:nSample
            featureMatrix{i} = rowFeatureMatrix(startIdx:startIdx+nDocs(i)-1,:); % Reshape the feature for each sample of query, document pair
            tagVector{i} = trueTags(startIdx:startIdx+nDocs(i)-1);
            startIdx = startIdx + nDocs(i);
            
           % [~,I] = sort(tagVector{i}, 'descend');
           % [~,idealPos] = sort(I);
           % IDCG(i) = computeDCG(tagVector{i},idealPos,1);
        end
        
        IDCG = ones(nSample,1);  % Optimizer DCG instead of NDCG
        %objectiveFunction = @objectiveFunctionDCG;
        objectiveFunction = @objectiveSampleFunctionDCG;
        extraPara = IDCG;
    else
        fprintf('Unrecognized measure: %s.', measure);
        return;
    end
    
    verbose = false; 
    
    %trueFeatureValues are column vectors, while featureMatrix are row
    %vectors.
    
    %options.progTol = 0; % accept any function/parameter changes
    %options.method = 'lbfgs';
    
    %[theta, score, ~, output] = minFunc(objectiveFunction, theta, options, ...
    %trueFeatureValues, featureMatrix, extraPara, regularization, valuePrecision, verbose);

     options.nSample = nSample;
     options.save_cycle = 10;
     options.LOG_PATH = 'data/sg_logging/';
     options.WORKER_NUM = 2;
     options.MINIBATCH_SIZE  = 30;
     
     [theta, score] = stochasticGradient(objectiveFunction, theta, options, ...
     trueFeatureValues, featureMatrix, extraPara, regularization, valuePrecision,verbose);
     
     output = [];
    
    score = -score;
        
end
