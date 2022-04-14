function [x,k] = optimizerSubGradient(varargin)
% Using sub-gradient method to find out the optimal solution of convex
% minimal point optimization problems. 
%
% Inputs:
%
% The varibles should be inputted in the order of:
% subGradFunc,initParas,knownParas,stepParas,objectFunc
%
% subGradFunc - fucntion: It should be function handle which provide the
% sub-gradient. Only the first parameter in this function is treated as
% unknown, all the order parameters should be provided in knowParas. And
% the first parameter must be a vector.
%
% knowParas - cell nFuncPara-1 * 1: This cell array contains all the
% known parameters. The sub-gradiant function should handle it.
%
% initParas: initial value of unknow parameters.
%
% stepParas - cell 2 * 1: Parameters that control the steps. e
% only support step with form a/b+k, where a,b should be give in the
% first and second part of stepPara and k is the current iteration
% numbers.
%
% objectFunc - function:It should be function handle of the object
% funciton. Only the first parameter in this function is treated as
% unknown, all the order parameters should be provided in knowParas. And
% the first parameter must be a vector.
%
% Outputs:
% x - nUnkownPara*1: final result(the unkown parameters) of the
% optimiztion problem.
% k : number of iterations
%
% Created by Wei Xing @ UIC , 2014-07-04
% Updated by Wei Xing @ UIC , 2014-09-18


l = length(varargin);

stepParas = cell(2,1);
stepParas{1} = 1;
stepParas{2} = 1000;
isEstimateAll = true;
isNeedCheck = false;

if l < 2
    fprintf('ERROR: A sub-gradient function and initial values are needed.\n');
    return;
end

subGradFunc = varargin{1};
initParas = varargin{2};

if l >= 3
    isEstimateAll = false;
    knownParas = varargin{3};
end

if l >=4
    stepParas = varargin{4};
end

if l >= 5
    objFunc = varargin{5};
    isNeedCheck = true;
    fileID = fopen('log_subgradient','w');
end

tolerance = 1e-6;
maxIter = 1e100;
previousParas = inf*ones(1,length(initParas));
currentParas = initParas;
nUnknownParas = length(initParas);
k = 0;
x= currentParas;
% epsilon = 1e-6;
epsilon = 1e-8;

while norm(currentParas-previousParas,2) > tolerance && k<=maxIter
    
    step = stepParas{1} / (stepParas{2} + k);
    
    if isEstimateAll
        g = feval(subGradFunc,currentParas);
    else
        g = feval(subGradFunc,currentParas,knownParas);
    end
    
    previousParas = currentParas;
    currentParas = currentParas - step * g; % Gradient direction increase the function while negative gradient direction decrease the function.
    
    if isNeedCheck
        fprintf(fileID,'Iteration %i\n',k+1);
        fprintf(fileID,'x:');
        fprintf(fileID,'%f ',previousParas); % When fprintf accept an array, it will actually run fprintf for each element.
        fprintf(fileID,'\n');
        fprintf(fileID,'Gradient:');
        fprintf(fileID,'%f ',g);
        fprintf(fileID,'\n');
        fprintf(fileID,'Gradient mode: %f\n',norm(g,2));
        fprintf(fileID,'Object function: %f\n',objFunc(previousParas,knownParas));
%         fprintf(fileID,'Difference between real objFunc and estimated objFunc: %f\n',objectFunc1D(previousParas)-objFunc(previousParas,knownParas));
        
        estimatedGradient = zeros(1,nUnknownParas);
        for i = 1:nUnknownParas
            posIdx = zeros(1,nUnknownParas);
            posIdx(i) = 1;
            if isEstimateAll
                estimatedGradient(i) = (objFunc(previousParas + posIdx * epsilon)- objFunc(previousParas))/epsilon;
            else
                estimatedGradient(i) = (objFunc(previousParas + posIdx * epsilon,knownParas)- objFunc(previousParas,knownParas))/epsilon;
            end
        end
        fprintf(fileID,'Forward difference:');
        fprintf(fileID,'%f ',estimatedGradient-g); % The differece is: estimated gradient - gradient function result
        fprintf(fileID,'\n');
        
        estimatedGradient = zeros(1,nUnknownParas);
        for i = 1:nUnknownParas
            posIdx = zeros(1,nUnknownParas);
            posIdx(i) = 1;
            if isEstimateAll
                estimatedGradient(i) = (objFunc(previousParas)- objFunc(previousParas - posIdx * epsilon))/epsilon;
            else
                estimatedGradient(i) = (objFunc(previousParas,knownParas) - objFunc(previousParas - posIdx * epsilon,knownParas))/epsilon;
            end
        end
        fprintf(fileID,'Backward difference:');
        fprintf(fileID,'%f ',estimatedGradient-g); % The differece is: estimated gradient - gradient function result
        fprintf(fileID,'\n');
        
        fprintf(fileID,'\n');
    end
    
    k = k + 1;
end

if k > maxIter
    fprintf('ERROR: Sub-gradient method cannot converge in given iteration times.\n');
    pause; return;
end

x= currentParas;

if isNeedCheck
    fclose(fileID);
end

