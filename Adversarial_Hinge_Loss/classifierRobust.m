function trainedParas = classifierRobust(training,featureFunc,typeLoss)

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
        [unqSample,freq,~] = sampleFreq(training.x);
        knownParas{1} = freq;
        knownParas{2} = featureEval(unqSample,training.y,featureFunc);
        statistics = statisticEval(training,featureFunc);
        knownParas{3} = statistics;
        stepParas= cell(3,1);
        
        % Setting step size to make step change according to the data.
        maxs = 0;
        for k = 1:nStat
            maxs_local = max(abs(statistics{k}));
            if maxs_local > maxs
                maxs = maxs_local;
            end
        end
        
        stepParas{1} = maxs/1000; stepParas{2} = 2; stepParas{3} = 0.5;
        
        
        thetaDimension = 0;
        for k = 1:nStat
            thetaDimension = thetaDimension + length(statistics{k});
        end
        initParas = ones(1,thetaDimension);
        theta = optimizerMixGradient(@gradientFuncHinge,@objectFuncHinge,initParas,knownParas,stepParas,true);
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
end

end

