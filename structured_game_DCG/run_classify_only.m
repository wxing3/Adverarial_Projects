clear all;
restoredefaultpath;
addpath(genpath(pwd));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
datasetName = 'Toy2';
measure = 'DCG'; % F1, Precision, Recall (case insensitidve)
%extraPara = 2; % for precision/recall@k, k = kPercent * num_of_positive_tags
% docNumTrain = [40 40 40]';
% docNumValid = 40;
% docNumTest = [40 40]';
%extraPara = load('nDoc'); % for DCG, it is vector informing the number of documents in each query-document pair.
learnInitialTheta = true;
% learnInitialTheta = load('data/theta.mat');
% learnInitialTheta = learnInitialTheta.theta;
useQuadFeatureValues = false;
regularizationMethod = 'L2';
% use '0' for no regularization
%regularizationValues = [0, 2^-6, 2^-5, 2^-4, 2^-3, 2^-2, 2^-1, 2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6 ];
regularizationValues = [2^-3];
%regularizationValue = [2^-5,2^-4, 2^-3, 2^-2,2^-1, 2^0,2];
biasFeatureValue = 1; % value for artificial bias feature
valuePrecision = 10^-8; % the smaller the more accurate
negativeClassTags = false; % if set to 'true', learning and predicting negative instances

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

testFileNames = dir(['data/' datasetName '/*.test']);

if length(testFileNames) < 1
    error('Training file does not exist.')
end

for i = 1:length(testFileNames)
    
    testFileName = [datasetName '/' testFileNames(i).name];
    
    if strcmpi(measure, 'DCG')
        testNDocFileName = [testFileName '.ndoc'];
        if(exist(testNDocFileName, 'file') ~= 2)
            error('Number of documents with each query in test data is unknow.')
        end
    end
    
    if strcmpi(measure, 'DCG')
        extraPara = testNDocFileName;
    else
        extraPara = kPercent;
    end
    %% load weights
    [fileNameSurrfix, fileNameRemain] = strtok(flip(testFileName), '.');
    thetaFilePrefix = ['data/' flip(strrep(fileNameRemain, '/', '_')) lower(measure) '.'];
    if(useQuadFeatureValues)
        thetaFilePrefix = [thetaFilePrefix, 'quad.'];  %#ok<*UNRCH>
        useQuadFeatureValuesStr = 'true';
    else
        useQuadFeatureValuesStr = 'false';
    end
    if(negativeClassTags)
        thetaFilePrefix = [thetaFilePrefix, 'neg.'];
    end
    
    %% test
    fprintf('---------------------------------------------------------\n');
    

    [trueTags, featureMatrix] = loadDataset(['data/' testFileName], useQuadFeatureValues);
    
    
        %Use saved theta
%     load([thetaFilePrefix regularizationMethod '_' num2str(regularizationValue) '.theta.mat']);
    
    %load('data/Toy2_train_20.dcg.L2_0.125.theta.mat');
    
    theta = [0.24076 0.209906 0.735067 0.447443 0.245042 0 0 0 0 0 0.310104 0.223083 0.123091 0.213232 0.674444 -0.0196277 -0.210605 0.133091 0.71661 0.109572 0.90808 2.94522 -2.09559 0.821801 -0.456707 -0.282868 1.29312 0.275617 0.0559919 -0.198624 0.70576 0.332177 -0.108193 1.58991 -1.01019 -0.279651 0.148817 0.984773 1.93553 -0.627407 0.576064 -0.474962 0.532025 1.17849 0.155317 0 3.2661 
]';
    %Use random theta
%     numberOfFeatures = size(featureMatrix, 2);
%     if biasFeatureValue
%         numberOfFeatures = numberOfFeatures + 1;
%     end
%     nRel_real = length(sort(unique(trueTags)));
%     if nRel_real > 2
%         theta = rand(numberOfFeatures*nRel_real,1);
%     else
%         theta = rand(numberOfFeatures,1);
%     end

    %     theta = theta/10;
    
    result = classify(measure, testFileName, theta, useQuadFeatureValues, ...
        extraPara , biasFeatureValue, negativeClassTags, valuePrecision);
    
    fprintf('%s [quadratic=%s] [measure=%s] [*%s=%.6f]\n', ...
        testFileName, useQuadFeatureValuesStr, measure, regularizationMethod, regularizationValue);
    if strcmpi(measure, 'precision') || strcmpi(measure, 'recall') || strcmpi(measure, 'f1')
        fprintf('Max Prob.(%.4f) Maximizer F1, P, R:\t%.6f\t%.6f\t%.6f\n', ...
            result.maxMaximizerProb, result.maxProbMaximizerF1, result.maxProbMaximizerPrecision, result.maxProbMaximizerRecall);
        fprintf('Max Prob.(%.4f) Minimizer F1, P, R:\t%.6f\t%.6f\t%.6f\n', ...
            result.maxMinimizerProb, result.maxProbMinimizerF1, result.maxProbMinimizerPrecision, result.maxProbMinimizerRecall);
        fprintf('Best Maximizer response F1, P, R:\t%.6f\t%.6f\t%.6f\n', ...
            result.maximizerF1, result.maximizerPrecision, result.maximizerRecall);
        fprintf('Best Minimizer response F1, P, R:\t%.6f\t%.6f\t%.6f\n', ...
            result.minimizerF1, result.minimizerPrecision, result.minimizerRecall);
    elseif strcmpi(measure, 'DCG')
        fprintf('Max Prob Maximizer NDCG@1, NDCG@2, NDCG@3, NDCG@4, NDCG@5, NDCG@6, NDCG@7, NDCG@8, NDCG@9, NDCG@10, NDCG@ALL:\n\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n', ...
            result.avg_maxProbMaximizerNDCG_K(1), result.avg_maxProbMaximizerNDCG_K(2), result.avg_maxProbMaximizerNDCG_K(3), result.avg_maxProbMaximizerNDCG_K(4), ...
            result.avg_maxProbMaximizerNDCG_K(5), result.avg_maxProbMaximizerNDCG_K(6), result.avg_maxProbMaximizerNDCG_K(7), result.avg_maxProbMaximizerNDCG_K(8), result.avg_maxProbMaximizerNDCG_K(9), ...
            result.avg_maxProbMaximizerNDCG_K(10), result.avg_maxProbMaximizerScore);
        fprintf('Best Maximizer response NDCG@1, NDCG@2, NDCG@3, NDCG@4, NDCG@5, NDCG@6, NDCG@7, NDCG@8, NDCG@9, NDCG@10, NDCG@ALL:\n\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n', ...
            result.avg_maximizerNDCG_K(1), result.avg_maximizerNDCG_K(2), result.avg_maximizerNDCG_K(3), result.avg_maximizerNDCG_K(4), result.avg_maximizerNDCG_K(5), result.avg_maximizerNDCG_K(6), ...
            result.avg_maximizerNDCG_K(7), result.avg_maximizerNDCG_K(8), result.avg_maximizerNDCG_K(9), result.avg_maximizerNDCG_K(10), result.avg_maximizerScore );
         fprintf('Quasi Robust response NDCG@1, NDCG@2, NDCG@3, NDCG@4, NDCG@5, NDCG@6, NDCG@7, NDCG@8, NDCG@9, NDCG@10, NDCG@ALL:\n\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n', ...
            result.avg_quasiRobustMaximizerNDCG_K(1), result.avg_quasiRobustMaximizerNDCG_K(2), result.avg_quasiRobustMaximizerNDCG_K(3), result.avg_quasiRobustMaximizerNDCG_K(4), result.avg_quasiRobustMaximizerNDCG_K(5), result.avg_quasiRobustMaximizerNDCG_K(6), ...
            result.avg_quasiRobustMaximizerNDCG_K(7), result.avg_quasiRobustMaximizerNDCG_K(8), result.avg_quasiRobustMaximizerNDCG_K(9), result.avg_quasiRobustMaximizerNDCG_K(10), result.avg_quasiRobustMaximizerScore );

    end
    
    save([thetaFilePrefix regularizationMethod '_' num2str(regularizationValue) '.test.result.mat'], 'result');
    
    fprintf('=========================================================\n');
end