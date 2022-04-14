function trainedParas = classifierRobustBinary_old(training,featureFunc,typeLoss)

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

% figure out a few things
sortedLabelValues  = sort(unique(training.y));
nClasses           = length(sortedLabelValues);

switch typeLoss
    case {'hinge'}
        
                
        % convert labels to +1 or -1
        if nClasses > 2
            fprintf('ERROR: current hinge loss robust classfification code only allows 2 class problems or a pairwise voting classifier based on SVM\n');
            pause
            return;
        else
            indices1 = training.y == sortedLabelValues(1);
            indices2 = training.y == sortedLabelValues(2);
            training.y(indices1) = -1;
            training.y(indices2) = 1;
        end
       
        knownParas = cell(3,1);
        [unqSample,freq,~] = sampleFreq(training.x); % unqSampel is a
        features = featureEval(unqSample,training.y,featureFunc);         
        knownParas{1} = freq;
        knownParas{2} = cell(nStat,1);
        knownParas{3} = cell(nStat,1);
        statistics = statisticEval(training,featureFunc);
        nUnqSample = length(freq);
        nFeatureDimension = zeros(nStat,1);
        for i = 1:nUnqSample
            for k = 1:nStat
                if i == 1
                    nFeatureDimension(k) = length(statistics{k});
                    knownParas{2}{k} = zeros(nUnqSample,nFeatureDimension(k));
                    knownParas{3}{k} = zeros(nUnqSample,nFeatureDimension(k));
                end
                knownParas{2}{k}(i,:) = features{k}(i,nFeatureDimension(k)+1:2*nFeatureDimension(k)) - features{k}(i,1:nFeatureDimension(k)); % Feature difference with label 1 and -1
                knownParas{3}{k}(i,:) = features{k}(i,nFeatureDimension(k)+1:2*nFeatureDimension(k)) + features{k}(i,1:nFeatureDimension(k)); % Feature sum
            end
        end
        knownParas{4} = statistics;
        stepParas= cell(3,1);
        
        % Setting step size to make step change according to the data.
        maxs = 0;
        for k = 1:nStat
            maxs_local = max(abs(statistics{k}));
            if maxs_local > maxs
                maxs = maxs_local;
            end
        end
        
        stepParas{1} = maxs; stepParas{2} = 10; stepParas{3} = 2;
        
        
        thetaDimension = 0;
        for k = 1:nStat
            thetaDimension = thetaDimension + length(statistics{k});
        end
        initParas = ones(1,thetaDimension);
        [theta,loss,nIter] = optimizerMixGradient(@gradientFuncHingeBinary,@objectFuncHingeBinary,initParas,knownParas,stepParas,false);
%         theta = optimizerSubGradient(func,initParas,knownParas,stepParas);
        
        
        
        % Dispatch the concatenated Langrage paremeters to a cell array in
        % which differen parameters of different statistic constrains are
        % put separately.
        paras = cell(nStat,1);
        lastIdx = 0;
        for k = 1:nStat
            nFeatureDimension = length(statistics{k});
            paras{k} = theta(lastIdx+1:lastIdx+nFeatureDimension);
            lastIdx = lastIdx + nFeatureDimension;
        end
        
        trainedParas{2} = paras;
        trainedParas{4} = loss;
        trainedParas{5} = nIter;
end

end

