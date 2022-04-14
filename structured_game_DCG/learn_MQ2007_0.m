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
regularizationValues = [0];
biasFeatureValue = 1; % value for artificial bias feature
valuePrecision = 10^-6; % the smaller the more accurate
negativeClassTags = false; % if set to 'true', learning and predicting negative instances

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainingFileNames = dir(['data/' datasetName '/*.train']);

if length(trainingFileNames) < 1
    error('Training file does not exist.')
end

for i = 1:length(trainingFileNames)
    trainingFileName = [datasetName '/' trainingFileNames(i).name];
    
    if strcmpi(measure, 'DCG')
        trainingNDocFileName = [trainingFileName '.ndoc'];
        if(exist(trainingNDocFileName, 'file') ~= 2)
            error('Number of documents with each query in training data is unknow.')
        end
    end
    
    validationFileName = strrep(trainingFileName, '.train', '.valid');
    if(exist(validationFileName, 'file') ~= 2)
        validationFileName = strrep(trainingFileName, '.train', '.test');
        warning('No validation file exists, using test file [%s] directly.', validationFileName);
        if(exist(validationFileName, 'file') ~= 2)
            warning('No test file exists, using training file [%s] directly.', validationFileName);
            validationFileName = trainingFileName;
        end
    end

    if strcmpi(measure, 'DCG')
        validationNDocFileName = [validationFileName '.ndoc'];
        if(exist(validationNDocFileName, 'file') ~= 2)
            error('Number of documents with each query in validation data is unknow.')
        end
    end

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
        [trainingScore, theta, output] = learn(measure, trainingFileName, learnInitialTheta, ...
            useQuadFeatureValues, trainingNDocFileName, regularization, biasFeatureValue, negativeClassTags, valuePrecision);
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
        save([thetaFilePrefix regularization.method '_' num2str(regularization.parameter) '.theta.mat'], 'trainingScore', 'theta','output');
        
%         %% validation
%         result = classify(measure, validationFileName, theta, useQuadFeatureValues, ...
%             validationNDocFileName, biasFeatureValue, negativeClassTags, valuePrecision);
%         
%         %% display
%         fprintf('%s [quadratic=%s] [measure=%s] [%s=%.6f]\n', ...
%             validationFileName, useQuadFeatureValuesStr, measure, regularizationMethod, regularizationValue);
%         
%         if strcmpi(measure, 'precision') || strcmpi(measure, 'recall') || strcmpi(measure, 'f1')
%             fprintf('Max Prob.(%.4f) Maximizer F1, P, R:\t%.6f\t%.6f\t%.6f\n', ...
%                 result.maxMaximizerProb, result.maxProbMaximizerF1, result.maxProbMaximizerPrecision, result.maxProbMaximizerRecall);
%             fprintf('Max Prob.(%.4f) Minimizer F1, P, R:\t%.6f\t%.6f\t%.6f\n', ...
%                 result.maxMinimizerProb, result.maxProbMinimizerF1, result.maxProbMinimizerPrecision, result.maxProbMinimizerRecall);
%             fprintf('Best Maximizer response F1, P, R:\t%.6f\t%.6f\t%.6f\n', ...
%                 result.maximizerF1, result.maximizerPrecision, result.maximizerRecall);
%             fprintf('Best Minimizer response F1, P, R:\t%.6f\t%.6f\t%.6f\n', ...
%                 result.minimizerF1, result.minimizerPrecision, result.minimizerRecall);
%         elseif strcmpi(measure, 'DCG')
%             fprintf('Max Prob Maximizer NDCG@1, NDCG@2, NDCG@3, NDCG@4, NDCG@5, NDCG@6, NDCG@7, NDCG@8, NDCG@9, NDCG@10, NDCG@ALL:\n\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n', ...
%                 result.avg_maxProbMaximizerNDCG_K(1), result.avg_maxProbMaximizerNDCG_K(2), result.avg_maxProbMaximizerNDCG_K(3), result.avg_maxProbMaximizerNDCG_K(4), ...
%                 result.avg_maxProbMaximizerNDCG_K(5), result.avg_maxProbMaximizerNDCG_K(6), result.avg_maxProbMaximizerNDCG_K(7), result.avg_maxProbMaximizerNDCG_K(8), result.avg_maxProbMaximizerNDCG_K(9), ...
%                 result.avg_maxProbMaximizerNDCG_K(10), result.avg_maxProbMaximizerScore);
%             fprintf('Best Maximizer response NDCG@1, NDCG@2, NDCG@3, NDCG@4, NDCG@5, NDCG@6, NDCG@7, NDCG@8, NDCG@9, NDCG@10, NDCG@ALL:\n\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n', ...
%                 result.avg_maximizerNDCG_K(1), result.avg_maximizerNDCG_K(2), result.avg_maximizerNDCG_K(3), result.avg_maximizerNDCG_K(4), result.avg_maximizerNDCG_K(5), result.avg_maximizerNDCG_K(6), ...
%                 result.avg_maximizerNDCG_K(7), result.avg_maximizerNDCG_K(8), result.avg_maximizerNDCG_K(9), result.avg_maximizerNDCG_K(10), result.avg_maximizerNDCG_K(1) );
% 
%         end
    end
    fprintf('=========================================================\n');
end
