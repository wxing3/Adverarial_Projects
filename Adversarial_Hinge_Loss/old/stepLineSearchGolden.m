function x = stepLineSearchGolden(varargin)
% Using Golden Section method to find out the optimal next point for
% gradient methods which try to find minimal points.
%
% Inputs:
%
% The varibles should be inputted in the order of:
% objectFunc,initParas
%
% objectFunc - function:It should be function handle of the object
% funciton. Only the first parameter in this function is treated as
% unknown, all the order parameters should be provided in knowParas. And
% the first parameter must be a vector.
%
% initParas - 2 * nDimension: They are the initial points where the
% function on the line connectting the two points is unimodel with a
% minimal value. This can be ensured if the object fucntion is convex and
% another point p3 besides these two points p1 and p2 are found such that
% p1, p2, p3 should be in a line, and objFunc(p1) and objFunc(p2) need to
% both be greater than objFunc(p3). 
%
% knowParas - cell nFuncPara-1 * 1: This cell array contains all the
% known parameters. The object function should handle it.
%
% Outputs:
% x : optimal next point 
%
% Created by Wei Xing @ UIC , 2014-09-13
% Updated by Wei Xing @ UIC , 2014-09-19

% input processing
l = length(varargin);
isEstimateAll = true; % Flag of whether all parameters are unknown.

if l < 2
    fprintf('ERROR: Insufficient input arguements.\n');
    return;
end

if l >= 3
    isEstimateAll = false;
    knownParas = varargin{3};
end

% initialize the golden rational
phi = (1+sqrt(5))/2;
rephi = 2 - phi;

tol = 1e-6;
maxIter = 1e100;
k = 0; % Count iterations

objFunc = varargin{1};
grandParas = varargin{2}(1,:);
previousParas = varargin{2}(2,:);
currentParas = grandParas + rephi * (previousParas - grandParas);

nextParas = currentParas + rephi * (previousParas - currentParas);

if isEstimateAll
    currentValue = objFunc(currentParas);
    nextValue = objFunc(currentParas);
else
    currentValue = objFunc(currentParas,knownParas);
    nextValue = objFunc(nextParas,knownParas);
end

while norm(grandParas-previousParas,2) >= tol * (norm(currentParas)+norm(nextParas)) && k<=maxIter
    a = norm(currentParas-grandParas,2);
    b = norm(previousParas-currentParas,2);
    
    if nextValue == currentValue
        break
    elseif nextValue < currentValue
        if b > a
            grandParas = currentParas;
            currentParas = nextParas;
            currentValue = nextValue;
            nextParas = nextParas + rephi * (previousParas - nextParas);            

        else
            previousParas = currentParas;
            currentParas = nextParas;
            currentValue = nextValue;
            nextParas = nextParas - rephi * (nextParas - grandParas);

        end
    else
        if b > a
            previousParas = nextParas;
            nextParas = currentParas - rephi * (currentParas - grandParas);
        else
            grandParas = nextParas;
            nextParas = currentParas + rephi * (previousParas - currentParas);
        end
    end
    
    if isEstimateAll
        nextValue = objFunc(nextParas);
    else
        nextValue = objFunc(nextParas,knownParas);
    end
    
    k = k + 1;
end

if k > maxIter
    fprintf('ERROR: Golden Ration linear search method cannot converge in given iteration times.\n');
    pause; return;
end

x = (grandParas + previousParas)/2;












