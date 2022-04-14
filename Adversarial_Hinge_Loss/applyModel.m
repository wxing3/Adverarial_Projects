function result =  applyModel(varargin)
% General prediction function. Details vary according to different methods.
%
% Inputs:
%   varargin{1} testing - structure: testing data
%   varargin{2} model - struct: model parameters whose size vary according
%       to different models. It contains:
%          type: same as input
%          trainedParas: contain all of the given and trained parameters
% Outputs:
%   result - struct: it contains
%     scores - nSample*nClasses: a matrix of scores of the labelling over
%     all the classes present in the test set. The columns of that matrix
%     correspond to the labels sorted in increasing order of their numeric
%     values.
%     acc - the accuracy of the model using highest score as the predicted
%     label
%
% Testing data should both be a structure with
%   y nSample*1 vector: lables
%   x nSample*nDimension matrix: samples
%
% nSample is the number of samples and nDimension is the dimension of feature
% space, nStat is number of statistic funcitons.
% the number of statistics constrains
%
% Created by Wei Xing @ UIC , 2014-07-03
% Updated by Wei Xing @ UIC , 2014-09-22

l = length(varargin);

if l < 2
    help trainModel;
    return
end

testing = varargin{1};
model   = varargin{2};

% extract info from model
nClasses          = model.trainingDataInfo.nClasses;
sortedLabelValues = model.trainingDataInfo.sortedLabelValues;
type              = model.type;

nTestingSample = length(testing.y);
scores = zeros(nTestingSample,nClasses);

switch type
    case {'robust'}
        paras = model.trainedParas{nClasses+1};
%         ypred = predictRobust(testing.x,sortedLabelValues,paras{1},paras{2},paras{3});
        ypred = predictRobustBinary(testing.x,sortedLabelValues,paras{1},paras{2},paras{3});
        
        acc = sum(ypred == testing.y)/nTestingSample;
        
        % each entry in ypred is negative (-1 prediction) or positive
        % (+1 prediction) , convert to 1 2
        indicesA = ypred <= 0;
        indicesB = ypred > 0;
        ypred(indicesA) = 1;
        ypred(indicesB) = 2;
        
        % Create a score array with one row per example, with one
        % column per label (in sortedLabelValues order). In that row,
        % the predicted label has value 1 and the others 0
        
        indices = sub2ind([nTestingSample,nClasses],(1:nTestingSample)',ypred);
        scores(indices) = 1;
        
    case {'libsvm'}
        % uses the MATLAB wrapper for Thorsten Joachim's libsvm
        % See SVM/svml.m for details on options available. For now,
        % we'll have to settle for picking a kernel and its parameters
        % (follows SVMdemsvml1.m example)
        
        if nClasses > 2
            fprintf('ERROR: current SVM code only allows 2 class problems or a pairwise voting classifier based on SVM\n');
            pause; return;
        end
        
        % The real label can be arbitrary here as we will evaluate the prediction
        % later.
        [ypred,~,~] = svmpredict( testing.y, testing.x', model.trainedParas{nClasses+1},'-q'); %SVM treat each row as a sample
        
        acc = sum(ypred == testing.y)/nTestingSample;
        
        % each entry in ypred is negative (-1 prediction) or positive
        % (+1 prediction) , convert to 1 2
        indicesA = ypred <= 0;
        indicesB = ypred > 0;
        ypred(indicesA) = 1;
        ypred(indicesB) = 2;
        
        % Create a score array with one row per example, with one
        % column per label (in sortedLabelValues order). In that row,
        % the predicted label has value 1 and the others 0
        scores = zeros(nTestingSample,nClasses);
        indices = sub2ind([nTestingSample,nClasses],(1:nTestingSample)',ypred);
        scores(indices) = 1;
end

result.scores = scores;
result.acc = acc;

end





