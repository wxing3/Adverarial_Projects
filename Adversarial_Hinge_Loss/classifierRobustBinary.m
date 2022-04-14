function trainedParas = classifierRobustBinary(training,featureFunc,typeLoss)

% Robust classification using hinge loss.
% Inputs:
%   training - structure: training data
%   featureFuncs - cell nStat*1: function handles which provide valid
%   features in statistics constrain.
%   typeLoss - string: loss function type
% Outputs:
%   trainedParas - cell 3*1: contain all of the given and trained parameters
%   It contains:
%     featureFuncs - cell nStat*1: function handles which provide valid
%     features in statistics constrain.
%     theta - 1 * sum(length(statistics)): All the Lagrauge coefficients of
%     the statistic constrains. Concatenated into one single vector.
%     typeLoss - string: loss function type
%     loss - 1*1: loss
%     nIter - 1*1: iterations of the optimizer used for training.
%
% Available types are: 'hinge'
%
% Training data should both be a structure with
%   y nSample*1 vector: lables
%   x nSample*nDimension matrix: samples
%
% nSample is the number of samples and nDimension is the dimension of feature
% space, nStat is number of statistic funcitons.
%
% Created by Wei Xing @ UIC , 2014-07-03
% Updated by Wei Xing @ UIC , 2014-09-24

% [nSample,nDimension] = size(training.x);
% sortedLabelValues  = sort(unique(training.y));
% nClasses           = length(sortedLabelValues);

trainedParas = cell(3,1);
trainedParas{1} = featureFunc;
trainedParas{3} = typeLoss;

nStat = length(featureFunc);

switch typeLoss
    case {'hinge'}
        
        knownParas = cell(3,1);
        [unqSamples,unqLabels,freq,~] = sampleFreq(training.x,training.y); %
        sortedLabelValues  = sort(unique(unqLabels));
        nUnqSample = length(freq);
        nClasses = length(sortedLabelValues); % Should Always be 2 in this case
        if nClasses ~= 2
            fprintf('ERROR: current hinge loss robust classfification code only allows 2 class problems or a pairwise voting classifier based on SVM\n');
            pause
            return;
        else
            indices1 = training.y == sortedLabelValues(1);
            indices2 = training.y == sortedLabelValues(2);
            training.y(indices1) = -1;
            training.y(indices2) = 1;
        end
        features = featureEval(unqSamples,unqLabels,featureFunc);
        
        nFeature = size(features,1)/nClasses;
        statistics = zeros(nFeature,size(unqSamples,2));
        for i = 1:size(unqSamples,2)
            idx = find(sortedLabelValues == unqLabels(i));
            statistics(:,i) = features(nFeature*(idx-1)+1:nFeature*idx,i);
        end
        
        knownParas{1} = freq;
        knownParas{2} = zeros(nFeature,nUnqSample); % Feature difference
        knownParas{3} = zeros(nFeature,nUnqSample); % Feature sum
        
        for i = 1:nUnqSample
            knownParas{2}(:,i) = features(nFeature+1:2*nFeature,i) - features(1:nFeature,i);
            knownParas{3}(:,i) = features(nFeature+1:2*nFeature,i) + features(1:nFeature,i);
        end
        knownParas{4} = statistics;
        stepParas= cell(3,1);
        
        % Setting step size to make step change according to the data.
        
        maxs = max(max(abs(statistics)));
        
        stepParas{1} = maxs; stepParas{2} = 10; stepParas{3} = 2;  % For mix gradient method
%         stepParas{1} = maxs; stepParas{2} = 1000; stepParas{3} = 0.5; % For sub gradient method
        
        initParas = ones(nFeature,1);
        
%         [theta,loss,nIter] = optimizerMixGradient(@gradientFuncHingeBinary,@objectFuncHingeBinary,initParas,knownParas,stepParas,true);
%         [theta,loss,nIter] = optimizerMixGradient(@gradientFuncHingeBinary_L1,@objectFuncHingeBinary_L1,initParas,knownParas,stepParas,true); % With L1 regulazation
        %         theta = optimizerSubGradient(func,initParas,knownParas,stepParas);
        [theta,loss,nIter] = optimizerStochasticGradient(@gradientFuncHingeBinary_L1,@gradientFuncHingeBinarySample_L1,@objectFuncHingeBinary,initParas,knownParas,stepParas,true);
        
        trainedParas{2} = theta;
        trainedParas{4} = loss;
        trainedParas{5} = nIter;
end

end

