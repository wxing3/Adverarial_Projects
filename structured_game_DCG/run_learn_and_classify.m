clc;
clear all;
restoredefaultpath;
addpath(genpath(pwd));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
datasetName = 'Test';
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
%regularizationValues = [0, 2^-6, 2^-4, 2^-2, 2^0, 2];
regularizationValues = [2^-3];
biasFeatureValue = 1; % value for artificial bias feature
valuePrecision = 10^-6; % the smaller the more accurate
negativeClassTags = false; % if set to 'true', learning and predicting negative instances

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainingFileNames = dir(['data/' datasetName '/*.train']);
for i = 1:length(trainingFileNames)
    trainingFileName = [datasetName '/' trainingFileNames(i).name];
    testFileName = strrep(trainingFileName, '.train', '.test');
    
    if(useQuadFeatureValues)
        useQuadFeatureValuesStr = 'true';
    else
        useQuadFeatureValuesStr = 'false';
    end
    
    for regularizationValue = regularizationValues
        regularization = struct('method', regularizationMethod, 'parameter', regularizationValue);
        fprintf('\n%s [quadratic=%s] [measure=%s] [%s=%.6f]\n', ...
            trainingFileName, useQuadFeatureValuesStr, measure, regularizationMethod, regularizationValue);
        
        %% training
        [trainingScore, theta] = learn(measure, trainingFileName, learnInitialTheta, ...
            useQuadFeatureValues, extraPara, regularization, biasFeatureValue, negativeClassTags, valuePrecision);
        fprintf('Training score=%.6f\n', trainingScore);
        
        %% save weights
        [fileNameSurrfix, fileNameRemain] = strtok(flip(trainingFileName), '.');
        thetaFilePrefix = ['data/' flip(strrep(fileNameRemain, '/', '_')) lower(measure) '.'];
        if(useQuadFeatureValues)
            thetaFilePrefix = [thetaFilePrefix, 'quad.'];  %#ok<*UNRCH>
        end
        if(negativeClassTags)
            thetaFilePrefix = [thetaFilePrefix, 'neg.'];
        end
        save([thetaFilePrefix regularization.method '_' num2str(regularization.parameter) '.theta.mat'], 'theta');
        
        %% predicting
        result = classify(measure, testFileName, theta, useQuadFeatureValues, ...
            extraPara, biasFeatureValue, negativeClassTags, valuePrecision);
        
        %% display
        fprintf('%s [quadratic=%s] [measure=%s] [%s=%.6f]\n', ...
            testFileName, useQuadFeatureValuesStr, measure, regularizationMethod, regularizationValue);
        fprintf('Max Prob.(%.4f) Maximizer F1, P, R:\t%.6f\t%.6f\t%.6f\n', ...
            result.maxMaximizerProb, result.maxProbMaximizerF1, result.maxProbMaximizerPrecision, result.maxProbMaximizerRecall);
        fprintf('Max Prob.(%.4f) Minimizer F1, P, R:\t%.6f\t%.6f\t%.6f\n', ...
            result.maxMinimizerProb, result.maxProbMinimizerF1, result.maxProbMinimizerPrecision, result.maxProbMinimizerRecall);
        fprintf('Best Maximizer response F1, P, R:\t%.6f\t%.6f\t%.6f\n', ...
            result.maximizerF1, result.maximizerPrecision, result.maximizerRecall);
        fprintf('Best Minimizer response F1, P, R:\t%.6f\t%.6f\t%.6f\n', ...
            result.minimizerF1, result.minimizerPrecision, result.minimizerRecall);
    end
    fprintf('=========================================================\n');
end
