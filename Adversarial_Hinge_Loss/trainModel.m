function model =  trainModel(varargin)
% General training function. Details vary according to different methods.
% Inputs:
%   varargin{1} training  - structure: training data
%   varargin{2} type - string: options for different models.
%   varargin{3} model parameters - cell: options for the particular
%   model.
% Outputs:
%   model struct: model parameters whose size vary according to
%   different models. It contains:
%     type - string: type of the model
%     when - time: when it is trained
%     modelParas: model parameters
%     trainingDataInfo - struct: infomatoin of training set which includes
%               nSample             = nSample
%               nDimension          = nDimension
%               nClasses            = nClasses
%               sortedLabelValues   = sortedLabelValues
%               classPriors         = prior of each class
%               idxSamplesWithLabel = index of samples for each class
%     trainedParas - struct:
%
%      --------------------------------------------------
%                trainedParas = cell(nClasses+3,1);
% 
%                Cells 1 to nclasses - contain the generative model for each class
%                Cell nclasses+1 - contains the discriminative model, if any
%                Cell nclasses+2 - contains any extra information, model specific
% 
%                Generative model - each cell contains a cell array
%                where each cell has one parameter, e.g.
% 
%                 trainedParas{1}{1} = mean class 1
%                 trainedParas{1}{2} = stdev class 1
%                 trainedParas{1}{3} = prior prob class 1
% 
%                Discriminative model - a cell array of sets of weights
%                 trainedParas{nclasses+1} = cell(1,1);
% 
%                Other classifier specific information
%                 model{nclasses+2}
%      ----------------------------------------------------
%      * This part is inspired by fpereira and wew
%
% Available types are: 'robust'
%
% Training data should both be a structure with
%   y nSample*1 vector: lables
%   x nSample*nDimension matrix: samples
%
% nSample is the number of samples and nFeature is the dimension of feature
% space, nStat is number of statistic funcitons.
% the number of statistics constrains
%
% Created by Wei Xing @ UIC , 2014-07-03
% Updated by Wei Xing @ UIC , 2014-09-22

l = length(varargin);

if l < 1
    help trainModel;
    return
end

training   = varargin{1};

if l < 2
    type = 'robust';
else
    type = varargin{2};
end

if l > 2
    modelParas = varargin{3};
else
    modelParas = cell(0);
end

% figure out a few things
sortedLabelValues  = sort(unique(training.y));
nClasses           = length(sortedLabelValues);
[nSample,nDimension] = size(training.x);
nPara              = length(modelParas);
labels = training.y;

% find the indices of examples with each label
idxSamplesWithLabel = cell(nClasses,1);
for l = 1:nClasses
    label = sortedLabelValues(l);
    idxSamplesWithLabel{l} = find(labels == label);
end

trainedParas = cell(nClasses+3,1);


switch type
    case {'robust'}
        
%         trainedParas{nClasses+1} = classifierRobust(training,modelParas{1},modelParas{2});
        trainedParas{nClasses+1} = classifierRobustBinary(training,modelParas{1},modelParas{2});
        
    case {'libsvm'}
        
        % uses the MATLAB wrapper for Thorsten Joachim's libsvm
        % created by Tom Briggs (thb@ship.edu).
        
        %   'Kernel'         -t       {0..4}, default value 1
        %                             Type of kernel function:
        %                             0: linear
        %                             1: polynomial (s a*b+c)^d
        %                             2: radial basis function exp(-gamma ||a-b||^2)
        %                             3: sigmoid tanh(s a*b + c)
        %                             4: user defined kernel from kernel.h
        %   'KernelParam'    -d, -g, -s, -r, -u
        %                             Depending on the kernel, this vector
        %                             contains [d] for polynomial kernel, [gamma]
        %                             for RBF, [s, c] for tanh kernel, string for
        %                             user-defined kernel
        
        %   fprintf('trainClassifier: using %s with parameters\n',classifierType);disp(classifierParameters);
        
        if nPara
            kernelParams = modelParas{1}; % a vector of parameter values
        else
            % default to a linear kernel
            kernelParams = '-t 0';
        end
        
        % convert labels to +1 or -1
        if nClasses > 2
            fprintf('ERROR: current SVM code only allows 2 class problems or a pairwise voting classifier based on SVM\n');
            pause
            return;
        else
            indices1 = training.y == sortedLabelValues(1);
            indices2 = training.y == sortedLabelValues(2);
            labels(indices1) = -1;
            labels(indices2) = 1;
            
        end
        
        % store model in the "discriminative" part
        trainedParas{nClasses+1} = svmtrain(labels, training.x', kernelParams); % SVM treat each row as a sample
end


%
% Store training set information
%

switch type
 case {'knn','libsvm','pairwise','neural','nnets','svm','logisticRegression','robust'}

  % Training Set information
  trainingDataInfo.nExamples            = nSample;
  trainingDataInfo.nFeatures            = nDimension;
  trainingDataInfo.nClasses             = nClasses;
  trainingDataInfo.sortedLabelValues    = sortedLabelValues;
  trainingDataInfo.classPriors          = zeros(nClasses,1);
  for l=1:nClasses
    trainingDataInfo.classPriors(l) = length(find(training.y==sortedLabelValues(l)));
  end
  trainingDataInfo.classPriors = trainingDataInfo.classPriors/nSample;            
  
 otherwise
  % the bayesian classifiers set it up inside classifierBayes
  trainingDataInfo = trainedParas{nClasses+2};
end

trainingDataInfo.idxSamplesWithLabel = idxSamplesWithLabel;

model.type = type;
model.when = datestr(now);
model.modelParas = modelParas;
model.trainingDataInfo = trainingDataInfo;
model.trainedParas = trainedParas;

end

