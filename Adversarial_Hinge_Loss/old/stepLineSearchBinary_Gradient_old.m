function x = stepLineSearchBinary_Gradient_old(varargin)
% Using binary method to find out the optimal point on one of the
% boundaries of the non optimal point in a one dimension convex
% optimization problem.
%
% Inputs:
%
% The varibles should be inputted in the order of:
% gradFunc, optPoint, nonOptPoint, knownParas
%
% gradFunc - function:It should be function handle of the gradient or
% subgradient of the function. Only the first parameter in this function is
% treated as unknown, all the order parameters should be provided in
% knowParas. And the first parameter must be a vector.
%
% initParas - 2 * nDimension: They are the initial points where the
% function on the line connectting the two points is unimodel with a
% minimal value.
%
% tolerace - 1 * 1 (1e-8): Error tolerance of the model which use this method
%
% knowParas - cell nFuncPara-1 * 1: This cell array contains all the
% known parameters. The sub-gradiant function should handle it. If it is
% not provided, it is assumed that all the needed additional parameters are
% already in the gradient function.
%
% Outputs:
% x : optimal next point
%
% Created by Wei Xing @ UIC , 2014-11-18
% Updated by Wei Xing @ UIC , 2014-11-19

% input processing
l = length(varargin);
isEstimateAll = true; % Flag of whether all parameters are unknown.
tolerance = 1e-8;

if l < 2
    fprintf('ERROR: Insufficient input arguements.\n');
    return;
end

gradFunc = varargin{1};

if l >= 3
    tolerance = varargin{3}; % Binary search need to be more precise to let gradient funciton decide the boundary.
end

if l >= 4
    isEstimateAll = false;
    knownParas = varargin{4};
end


maxIter = 1e10;
k = 0; % Count iterations

d = varargin{2}(2,:) - varargin{2}(1,:);
d = d/norm(d);

if isEstimateAll
    g1 = gradFunc(varargin{2}(1,:),0); % In binary search, the tolerance of boundary should be 0, so that the minimal point can get the real minimal point, otherwise the minimal point will become one of theat * featureDif = 2 + tol and theat * featureDif = 2 - tol
    g2 = gradFunc(varargin{2}(2,:),0);
else
    g1 = gradFunc(varargin{2}(1,:),0,knownParas);
    g2 = gradFunc(varargin{2}(2,:),0,knownParas);
end

if norm(g1 * d') <= tolerance % This means first point is an optimal point.
    if norm(g2 * d') <= tolerance
        x = varargin{2}(1,:);
        fprintf('WARNING: These two points have the same object function value.\n');
        return;
    elseif g2 * d' > 0
        left = varargin{2}(2,:);
        right = varargin{2}(1,:);
        d = -d; % Keep direction from left to right, so that _| shape model becomes |_ shape, and we always find the left boundary of the region whose graident is 0.
    else
        fprintf('ERROR: Opitimal point does not lie between the two given points.\n');
        pause; return;
    end
elseif g1 * d' < 0
    if norm(g2*d') > tolerance && g2 * d' < 0
        fprintf('ERROR: Opitimal point does not lie between the two given points.\n');
        pause; return;
    else
        left = varargin{2}(1,:);
        right = varargin{2}(2,:);
    end
else
    fprintf('ERROR: Opitimal point does not lie between the two given points.\n');
    pause; return;
end

% tol = tolerance; % Tolance to find the minimal point.
%
% while 1 % Find a minimal point, than use hyperplane information to get better tolerance until using tolerance in the model can find the hyperplane.
%
% % better ending critia if optimal point is near 0
% %      * min(1,(norm(left)+norm(right)))
%
%     while norm(left - right) >= tol && k<=maxIter %This can be very slow if left or right is too small
%
%         middle = (right + left)/2;
%
%         if isEstimateAll
%             middleG = gradFunc(middle,0);
%         else
%             middleG = gradFunc(middle,0,knownParas);
%         end
%
%     if norm(middleG * d') <= tolerance
%         right = middle;
%     elseif middleG * d' > 0
%         right = middle;
%     else
%         left = middle;
%     end
%
%         k = k + 1;
%     end
%
%     x = (left + right)/2;
%
%     [~,hyperPlanes] = gradFunc(x,tolerance,knownParas);
%     nBoundaries = length(hyperPlanes{1});
%     if nBoundaries > 0 % Point got is accurate enough to let the opimizer find the hyperplane it is on.
%         break;
%     end
%
%     etol = tol; % Expanded tolerance to find the hyperplane
%     nBoundaries = 0;
%     while nBoundaries == 0
%     [~,hyperPlanes] = gradFunc(x,etol,knownParas); % Use high tolerance to get the nearest hyperplane, which will give a bound of how accuracy the minimal point should be so that we can make sure it is on a hyperplane in optimizer.
%     minCosin = 1;
%     nBoundaries = length(hyperPlanes{1});
%     etol = etol * 10;
%     end
%
%     for i = 1:nBoundaries
%         v = hyperPlanes{2}(:,i); % This is a colume vector.
%         cosin = abs(d * v / norm(v)); % For the give direction that the point x can move around, in w * x' = b if b has a tolerance of tol, than x with tolerance tol/cos(w,d) can ensure w * x' within b's tolerance.
%         if cosin < minCosin
%             minCosin = cosin;
%         end
%     end
%     tol = tol * minCosin * 0.9;
% end


tol = tolerance^2; % Tolance to find the minimal point should be smaller than the tolerance of the optimizer.

while norm(left - right) >= tol && k<=maxIter %This can be very slow if left or right is too small
    
    middle = (right + left)/2;
    
    if isEstimateAll
        middleG = gradFunc(middle,0);
    else
        middleG = gradFunc(middle,0,knownParas);
    end
    
    if norm(middleG * d') <= tolerance
        right = middle;
    elseif middleG * d' > 0
        right = middle;
    else
        left = middle;
    end
    
    k = k + 1;
end

x = (left + right)/2;

if k > maxIter
    fprintf('ERROR: Binary search method cannot converge in given iteration times.\n');
    pause; return;
end












