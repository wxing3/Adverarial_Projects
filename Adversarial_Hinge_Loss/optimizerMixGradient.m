function [minParas,minF,k] = optimizerMixGradient(varargin)
% Using gradient method with line search to find out the optimal
% solution of convex minimal point optimization problems. It contains 3
% parts:
% 1. A base step size used for basic subgradient method
% 2. Accelerated step size for hing-loss problem whose solution subspace is
% a set of continuous hyperplanes. If the gradient does no change, step size
% will increase, otherwise update and use the base step size.
% 3. When we find a bound on a line contianing a unimold, Gold Section line
% search method is used to further increase the speed.
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
% objectFunc - function:It should be function handle of the object
% funciton. Only the first parameter in this function is treated as
% unknown, all the order parameters should be provided in knowParas. And
% the first parameter must be a vector.
%
% initParas: initial value of unknow parameters.
%
% knowParas - cell nFuncPara-1 * 1: This cell array contains all the

% known parameters. The sub-gradiant function should handle it.
%
% stepParas - cell 3*1: A cell array whose first element is base step size.
% The first element is the numerator. The second element is the decreasing
% rate of base step size. The third element is the increasing rate when
% the algorithm steped into acceleration step.  basestap = stepParas{1} /
% (1000 + stepParas{2} * k}.
%
% isNeedCheck - boolean: Parameters controlling whether the log is needed.
%
% Outputs:
% x - nUnkownPara*1: final result(the unkown parameters) of the
% optimiztion problem.
% minF - 1: The optimal value of the object function.
% k : number of iterations
%
% Created by Wei Xing @ UIC , 2014-09-20
% Updated by Wei Xing @ UIC , 2014-11-20

tic;

import java.util.LinkedList

l = length(varargin);

stepParas = cell(2,1);
stepParas{1} = 1;
stepParas{2} = 10;
stepParas{3} = 2;

isEstimateAll = true;
isNeedCheck = false;

if l < 3
    fprintf('ERROR: Insufficient input arguments.\n');
    return;
end


subGradFunc = varargin{1};
objFunc = varargin{2};
initParas = varargin{3};

if l >= 4
    isEstimateAll = false;
    knownParas = varargin{4};
end

if l >= 5
    stepParas = varargin{5};
    if stepParas{1} < 0
        fprintf('ERROR: Base step size must be positive.\n');
        return;
    end
    if stepParas{3} < 1
        fprintf('ERROR: Increasing rate must in (1,Inf).\n');
        return;
    end
end

if l >= 6
    isNeedCheck = varargin{6};
end



% Fixed model parameters.
tolerance = 1e-6;
maxIter = 1e10;
epsilon = 1e-8; % Probe distance to estimate the real gradient value.

% Initialize gradient method parameters
k = 0;
nUnknownParas = length(initParas);
currentParas = initParas;
basestep = stepParas{1}/1000;
step = basestep;

% The following 3 parameters are intialized to voilate binary search condition in first two steps
previousG = zeros(nUnknownParas,1); % Previous greadient function value
grandG = previousG; % Gradient function value 2 steps ago
previousF = -inf; % Previous object function value
grandF = previousF; % Object function value 2 steps ago
previousParas = zeros(nUnknownParas,1); % Previous unknown parameters value
grandParas = previousParas; % Unknown parameters value 2 steps ago.
previousD = zeros(nUnknownParas,1); % Direction (used to find next point) that previous point has.
grandD = previousD; % Direction 2 steps age.

if isEstimateAll
    [g, hyperPlanes] = subGradFunc(currentParas,tolerance);
    f = objFunc(currentParas);
else
    [g, hyperPlanes] = subGradFunc(currentParas,tolerance,knownParas);
    f = objFunc(currentParas,knownParas);
end

nBoundaries = length(hyperPlanes{1});
d = g/norm(g); % Normalized direction of the gradient.

minParas = currentParas;
minF = f; % Currrent minimal f

updateFlag = 1;
isPlaneConstrained = false; % Showing whether current optimazation is constrained on a hyperplane.
isGradientPerpendicular = false; % Showing whether
nUnconstrained = 0; % Number of repiting unconstrained binary search
maxUnconstrained = 20; % If nUnconstrained exceed this number, a forced hyperplane constrained search is done.

if isNeedCheck
    fileID = fopen('log_subgradient','w');
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

nUnqSample = length(knownParas{1});

while norm(g) > tolerance && k <= maxIter
    
    if isNeedCheck % Check the correctness of the gradient and object function value of current point.
        
        fprintf(fileID,'Iteration %i\n',k);
        switch updateFlag
            case 1
                updateMethod = 'Subgradient';
                if nUnknownParas >= 3
                    scatter3(currentParas(1),currentParas(2),currentParas(3),'b.');
                elseif nUnknownParas >=2
                    plot(currentParas(1),currentParas(2),'b.');
                else
                    plot(currentParas,0,'b.');
                end
            case 2
                updateMethod = 'Binary Search';
                if nUnknownParas >= 3
                    scatter3(currentParas(1),currentParas(2),currentParas(3),'ro');
                elseif nUnknownParas >=2
                    plot(currentParas(1),currentParas(2),'ro');
                else
                    plot(currentParas,0,'ro');
                end
            case 3
                updateMethod = 'Hyperplane Constrain';
                if nUnknownParas >= 3
                    scatter3(currentParas(1),currentParas(2),currentParas(3),'g*');
                elseif nUnknownParas >=2
                    plot(currentParas(1),currentParas(2),'g*');
                else
                    plot(currentParas,0,'g*');
                end
        end
        fprintf(fileID,['Update method used to get this result is:' updateMethod '\n']);
        fprintf(fileID,'x:');
        fprintf(fileID,'%f ',currentParas); % When fprintf accept an array, it will actually run fprintf for each element.
        fprintf(fileID,'\n');
        fprintf(fileID,'Gradient:');
        fprintf(fileID,'%f ',g);
        fprintf(fileID,'\n');
        %         fprintf(fileID,'Gradient mode: %f\n',norm(g,2));
        fprintf(fileID,'Current base step size and step size: %e %e\n',basestep,step);
        fprintf(fileID,'Last direction:');
        fprintf(fileID,'%f ',d);
        fprintf(fileID,'\n');
        fprintf(fileID,'Number of hyperplanes that the point is on: %i\n',nBoundaries);
        fprintf(fileID,'Current minimal value: %e\n',minF);
        fprintf(fileID,'Object function value: %e\n',f);
        %         fprintf(fileID,'Difference between real objFunc and estimated objFunc: %f\n',objectFunc1D(previousParas)-objFunc(previousParas,knownParas));
        
        for i = 1:nUnknownParas
            posIdx = zeros(nUnknownParas,1);
            posIdx(i) = 1;
            if isEstimateAll
                estimatedGradient(i) = (objFunc(currentParas + posIdx * epsilon)- objFunc(currentParas))/epsilon;
            else
                estimatedGradient(i) = (objFunc(currentParas + posIdx * epsilon,knownParas)- objFunc(currentParas,knownParas))/epsilon;
            end
        end
        fprintf(fileID,'Forward difference:');
        fprintf(fileID,'%e ',estimatedGradient-g); % The differece is: estimated gradient - gradient function result
        fprintf(fileID,'\n');
        
        for i = 1:nUnknownParas
            posIdx = zeros(nUnknownParas,1);
            posIdx(i) = 1;
            if isEstimateAll
                estimatedGradient(i) = (objFunc(currentParas)- objFunc(currentParas - posIdx * epsilon))/epsilon;
            else
                estimatedGradient(i) = (objFunc(currentParas,knownParas) - objFunc(currentParas - posIdx * epsilon,knownParas))/epsilon;
            end
        end
        fprintf(fileID,'Backward difference:');
        fprintf(fileID,'%e ',estimatedGradient-g); % The differece is: estimated gradient - gradient function result
        fprintf(fileID,'\n');
        
        fprintf(fileID,'\n');
    end
    
    if k >= 2 % Starting from the third step (minimal steps needed to check binary search condition), check and change the updating method that should be used at this step.
        switch updateFlag % This is previous update step.
            case 1 % Normal subgradient method
                if abs(g' * d) <= tolerance % If current gradient is perpendicular to current direction,
                    if nBoundaries > 0 %If current point is on a hyperplanetry, then constrain the movement on the hyperplane directly, to prevent step into this region again.
                        hyperPlaneList = LinkedList();
                        for j = 1:nBoundaries
                            hyperPlaneList.push(hyperPlanes{2}(:,j));
                        end
                        constrainedStartPoint.x = currentParas;
                        constrainedStartPoint.g = g;
                        constrainedStartPoint.f = f;
                        constrainedStartPoint.px = previousParas;
                        constrainedStartPoint.pg = previousG;
                        constrainedStartPoint.pf = previousF;
                        updateFlag = 3;
                    else %If current point isn't on a hyperplanetry, then use Binary Boundary search to find nearest point in previous path that has a gradient which is not perpendicular to currrent direction. This point must on a hyperplane.
                        isGradientPerpendicular = true;
                        leftbound = previousParas;
                        rightbound = currentParas;
                        updateFlag = 2;
                    end
                else
                    if norm(d - previousD) <= tolerance && norm(d - grandD) <= tolerance % If there continuous points are not in a line, then the following judgement cannot be true.
                        if norm(grandG) > tolerance && ((previousG' * d > 0 && g' * d < 0 || previousG' * d < 0 && g' * d > 0 )) && norm(previousG - g) > tolerance && norm(previousG - grandG) <= tolerance  % Condition that a unimodal segment between two points is ensured.
                            leftbound = grandParas;
                            rightbound = currentParas;
                            updateFlag = 2;
                        elseif norm(grandG) > tolerance && ((previousG' * d < 0 && grandG' * d > 0) || (previousG' * d > 0 && grandG' * d < 0)) && norm(grandG - previousG) > tolerance && (norm(previousG - g) <= tolerance || norm(grandG - g) <= tolerance)
                            leftbound = grandParas;
                            rightbound = previousParas;
                            updateFlag = 2;
                        end
                    end
                end
            case 2 %Binary Search method
                if nBoundaries == 0
                    fprintf('ERROR: Binary search method failed to find a point on the hyperplane.\n');
                    pause;
                    return;
                end
                
                if isPlaneConstrained % Leave the work for case 3 to judge whether optimal value is achieved.
                    % What if Gradient is perpendicular on constrained optimization?
                    updateFlag = 3;
                elseif isGradientPerpendicular
                    if nBoundaries > 0 % If the nearest non perpendicular on previous path is on the bound, try to constrain the movement on the hyperplane directly, to prevent step into this region again.
                        hyperPlaneList = LinkedList();
                        for j = 1:nBoundaries
                            hyperPlaneList.push(hyperPlanes{2}(:,j));
                        end
                        constrainedStartPoint.x = currentParas;
                        constrainedStartPoint.g = g;
                        constrainedStartPoint.f = f;
                        constrainedStartPoint.px = previousParas;
                        constrainedStartPoint.pg = previousG;
                        constrainedStartPoint.pf = previousF;
                        updateFlag = 3;
                    else % If previous condition is incorrect, then the code must have some problem!
                        fprintf('ERROR: No gradients of two points got from two countinuous steps can be both perpendicular to current direction after a binary boundary search.\n');
                        pause;
                        return;
                    end
                    isGradientPerpendicular = false;
                else
                    if f > minF || nUnconstrained > maxUnconstrained% If using binary search does not yield better result.
                        hyperPlaneList = LinkedList();
                        for j = 1:nBoundaries % This constant as specified by
                            hyperPlaneList.push(hyperPlanes{2}(:,j));
                        end
                        constrainedStartPoint.x = currentParas;
                        constrainedStartPoint.g = g;
                        constrainedStartPoint.f = f;
                        constrainedStartPoint.px = previousParas;
                        constrainedStartPoint.pg = previousG;
                        constrainedStartPoint.pf = previousF;
                        updateFlag = 3;
                    else % Otherwise, just do normal gradient
                        d = g/norm(g); % Update search direction with current gradient.
                        updateFlag = 1;
                        nUnconstrained = nUnconstrained + 1;
                    end
                end
                
            case 3 % Gradient constrained on a hyperplane.
                if isPlaneConstrained % Start doing a normal gradient search constraining the movement on a hyperlane.
                    updateFlag = 1;
                else % The optimal value constrained on a hyperplane is got.
                    if f < minF % Improvement achieved by moving on a hyperplane
                        d = g/norm(g,2); % Update search direction with current gradient.
                        updateFlag = 1;
                    else
                        if hyperPlaneList.size() == 0 % If constrain on all the hyperplane does not make any progress.
                            break; % Optimal value is found.
                        else % Restore the point that generated current hyperPlaneList, and try another hyperplane
                            currentParas = constrainedStartPoint.x;
                            g = constrainedStartPoint.g;
                            f = constrainedStartPoint.f;
                            previousParas = constrainedStartPoint.px;
                            previousG = constrainedStartPoint.pg;
                            previousF = constrainedStartPoint.pf;
                        end
                    end
                    
                end
        end
    end
    
    
    if f < minF % Use f from last time to update minF
        minF = f;
        minParas = currentParas;
    end
    
    switch updateFlag
        case 1 %Gradient direction will not change in this step
            
            if norm(g - previousG) <= tolerance %Acceleration
                step = step * stepParas{3};
            else % Acceleration stop when another hyperplane is reached.
                step = basestep;
                basestep = stepParas{1} / (1000 + stepParas{2} * k);
                basestep = max([basestep, tolerance]);
            end
            
            grandParas = previousParas;
            previousParas = currentParas; % The currentParas is real current, while g and f are previous values.
            
            g_line = (d' * g) * d; % Gradient constrained on a line.
            currentParas = currentParas - step * g_line; % Finding next point. Gradient direction increase the function while negative gradient direction decrease the function.
            
        case 2 % Each 2 dimension minimal problem must ended with binary search method.
            
            grandParas = previousParas;
            previousParas = currentParas; % The currentParas is real current, while g and f are previous values.
            
            if isEstimateAll
                currentParas = stepLineSearchBinary_Gradient(subGradFunc,[leftbound rightbound],tolerance);
            else
                currentParas = stepLineSearchBinary_Gradient(subGradFunc,[leftbound rightbound],tolerance,knownParas);
            end
            
        case 3 % If binary search cannot find better result, we will try to constrain the problem on one of the hyperplanes.
            if isPlaneConstrained
                isPlaneConstrained = false; % Do nothing. Let the checking process test whether improvement is achieved first.
            else
                hyperPlane = hyperPlaneList.pop();
                normV = hyperPlane(1:length(hyperPlane)-1); % Normal vector of the hyperplane, need to discard the last constant, a colume vector
                normV = normV / norm(normV,2);
                g_hyperplane = g - (g' * normV) * normV; % Projection of the gradient on given hyperplane.
                
                d = g_hyperplane / norm(g_hyperplane,2);
                
                isPlaneConstrained = true;
                nUnconstrained = 0; %Whenever a constrained update is done, nUnconstrained should be set to 0.
            end
        otherwise
            fprintf('ERROR: Illegal updating strategy.\n');
            return;
    end
    
    if updateFlag ~= 3 % If unknown parameters are updated in case 1 or case 2, then update the corresponding gradient and object function value.
        grandF = previousF;
        previousF = f;
        grandG = previousG;
        previousG = g;
        
        if isPlaneConstrained %If optimization is not constrained on the hyperplane, there is no need to use subgradient, the point can choose the gradient from any nearby regions.
            tol = -tolerance;
        else
            tol = tolerance;
        end
        
        if isEstimateAll
            [g, hyperPlanes] = subGradFunc(currentParas,tol);
            f = objFunc(currentParas);
        else
            [g, hyperPlanes] = subGradFunc(currentParas,tol,knownParas);
            f = objFunc(currentParas,knownParas);
        end
        
        nBoundaries = length(hyperPlanes{1});
        
        % The new trained point should have the same direction at this
        % time, any change later will be recorded in previousD
        grandD = previousD;
        previousD = d;
        
    end
    
    k = k + 1;
end

if isNeedCheck
    fclose(fileID);
end

% Update optimal value with the newest result.
if f < minF % Use f from last time to update minF
    minF = f;
    minParas = currentParas;
end

if k > maxIter
    fprintf('ERROR: Sub-gradient method cannot converge in given iteration times.\n');
    pause;
    return;
end

toc


