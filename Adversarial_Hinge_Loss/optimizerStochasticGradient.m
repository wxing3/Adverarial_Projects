function [minParas,minF,k] = optimizerStochasticGradient(varargin)
% Using sub-gradient method to find out the optimal solution of convex
% minimal point optimization problems.
%
% Inputs:
%
% The varibles should be inputted in the order of:
% subGradFunc,initParas,knownParas,stepParas,objectFunc
%
%
% subGradFunc - fucntion: It should be function handle which provide the
% sub-gradient. Only the first parameter in this function is treated as
% unknown, all the order parameters should be provided in knowParas. And
% the first parameter must be a vector.
%
% sampleSubGradFunc - fucntion: The same as subGradFunc except that
%
% objectFunc - function:It should be function handle of the object
% funciton. Only the first parameter in this function is treated as
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
% isNeedCheck boolean: indicate whether a log is needed
%
% Outputs:
% x - nUnkownPara*1: final result(the unkown parameters) of the
% optimiztion problem.
% k : number of iterations
%
% Created by Wei Xing @ UIC , 2014-11-26
% Updated by Wei Xing @ UIC , 2014-11-26

tic;

l = length(varargin);

stepParas = cell(2,1);
stepParas{1} = 1;
stepParas{2} = 1000;
stepParas{3} = 1;
isEstimateAll = true;
isNeedCheck = false;

if l < 2
    fprintf('ERROR: A sub-gradient function and initial values are needed.\n');
    return;
end

subGradFunc = varargin{1};
sampleSubGradFunc = varargin{2};
objFunc = varargin{3};
initParas = varargin{4};

if l >= 5
    isEstimateAll = false;
    knownParas = varargin{5};
end

if l >=6
    stepParas = varargin{6};
end

if l >= 7
    isNeedCheck = varargin{7};
end

tolerance = 1e-3;
maxIter = 1e5;
epsilon = 1e-8;

minParas = inf*ones(1,length(initParas));
f = 0;
minF = inf;

gs = 0;

currentParas = initParas;
nUnknownParas = length(initParas);

nSample = length(knownParas{1});
sampleListCount = nSample;
step = 0;

k = 0;

if isNeedCheck
    fileID = fopen('log_stocsubgradient','w');
    posIdx = zeros(nUnknownParas,1);
    estimatedGradient = zeros(nUnknownParas,1);
    if nUnknownParas >= 3
        scatter3(initParas(1),initParas(2),initParas(3),'b.');
    elseif nUnknownParas >=2
        plot(initParas(1),initParas(2),'b.');
    else
        plot(initParas,0,'b.');
    end
    hold on;
    grid on;
end

while (norm(gs) < tolerance || norm(f-minF) > tolerance) && k<=maxIter
    
    if sampleListCount >= nSample;
        sampleList = randperm(nSample);
        sampleListCount = 1;
        sampleIdx = sampleList(sampleListCount);
    else
        sampleListCount = sampleListCount + 1;
        sampleIdx = sampleList(sampleListCount);
    end
    
    if isEstimateAll
        gs = feval(sampleSubGradFunc,currentParas,sampleIdx,-tolerance);
        g = feval(subGradFunc,currentParas,tolerance);
        f = feval(objFunc,currentParas);
    else
        gs = feval(sampleSubGradFunc,currentParas,sampleIdx,-tolerance,knownParas);
        g = feval(subGradFunc,currentParas,tolerance,knownParas);
        f = feval(objFunc,currentParas,knownParas);
    end
    
    if f < minF
        minParas = currentParas;
    end
    
    if norm(g) < tolerance
        break
    end
    
    k = k + 1;
    
    if isNeedCheck
        if norm(gs) > tolerance
            if nUnknownParas >= 3
                scatter3(currentParas(1),currentParas(2),currentParas(3),1,[k/(maxIter+1),0,1-k/(maxIter+1)]);
            elseif nUnknownParas >=2
                plot(currentParas(1),currentParas(2),'Color',[k/(maxIter+1),0,1-k/(maxIter+1)]);
            else
                plot(currentParas,0,'Color',[k/(maxIter+1),0,1-k/(maxIter+1)]);
            end
        end
        fprintf(fileID,'Iteration %i\n',k);
        fprintf(fileID,'x:');
        fprintf(fileID,'%f ',currentParas); % When fprintf accept an array, it will actually run fprintf for each element.
        fprintf(fileID,'\n');
        fprintf(fileID,'Gradient:');
        fprintf(fileID,'%f ',g);
        fprintf(fileID,'\n');
        fprintf(fileID,'Sample Gradient on Sample %i:',sampleIdx);
        fprintf(fileID,'%f ',gs);
        fprintf(fileID,'\n');
        fprintf(fileID,'Step size: %f\n',step);
        fprintf(fileID,'Object function: %f\n',objFunc(currentParas,knownParas));
        %         fprintf(fileID,'Difference between real objFunc and estimated objFunc: %f\n',objectFunc1D(previousParas)-objFunc(previousParas,knownParas));
        
        for i = 1:nUnknownParas
            posIdx(i) = 1;
            if isEstimateAll
                estimatedGradient(i) = (objFunc(currentParas + posIdx * epsilon)- objFunc(currentParas))/epsilon;
            else
                estimatedGradient(i) = (objFunc(currentParas + posIdx * epsilon,knownParas)- objFunc(currentParas,knownParas))/epsilon;
            end
        end
        fprintf(fileID,'Forward difference:');
        fprintf(fileID,'%f ',estimatedGradient-g); % The differece is: estimated gradient - gradient function result
        fprintf(fileID,'\n');
        
        for i = 1:nUnknownParas
            posIdx(i) = 1;
            if isEstimateAll
                estimatedGradient(i) = (objFunc(currentParas)- objFunc(currentParas - posIdx * epsilon))/epsilon;
            else
                estimatedGradient(i) = (objFunc(currentParas,knownParas) - objFunc(currentParas - posIdx * epsilon,knownParas))/epsilon;
            end
        end
        fprintf(fileID,'Backward difference:');
        fprintf(fileID,'%f ',estimatedGradient-g); % The differece is: estimated gradient - gradient function result
        fprintf(fileID,'\n');
        
        fprintf(fileID,'\n');
    end
    
    currentParas = currentParas - step * gs; % Gradient direction increase the function while negative gradient direction decrease the function.
    
    if norm(gs) > tolerance
        step = stepParas{1} / (stepParas{2} + stepParas{3}*k);
    end
end

if isNeedCheck
    fclose(fileID);
end

if isEstimateAll
    f = feval(objFunc,currentParas);
else
    f = feval(objFunc,currentParas,knownParas);
end

if f < minF
    minParas = currentParas;
    minF = f;
end

if k > maxIter
    fprintf('ERROR: Sub-gradient method cannot converge in given iteration times.\n');
    toc
    pause; return;
end

toc