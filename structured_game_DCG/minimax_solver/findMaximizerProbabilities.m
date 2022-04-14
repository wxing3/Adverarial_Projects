function [p, v, errorFlag] = findMaximizerProbabilities(scoreMatrix, calledTimes, verbose)
    if nargin <3
        verbose = false;
    end

    if nargin < 2
        calledTimes = 1;
    end
    
    neg = min(min(scoreMatrix)); % this method requires all the values are positive
    if (neg <= 0)
        scoreMatrix = scoreMatrix + (1 - neg);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if verbose
        if(calledTimes == 0)
            fprintf('\t');
        elseif(mod(calledTimes, 100) == 0)
            fprintf('\n\t');
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    timeout = 60; % seconds;
    global lpSolver;
    
    if(~isempty(lpSolver) && strcmp(lpSolver, 'gurobi'))
        [x, errorFlag] = gurobiWrapper(scoreMatrix, timeout,verbose);
    else
        [x, errorFlag] = lpsolveWrapper(scoreMatrix, timeout,verbose);
    end
    
    v = 1 / sum(x);
    p = x * v;
    
    if(neg <= 0)
        v = v - (1 - neg);
    end
end

%%
function [x, errorFlag] = lpsolveWrapper(scoreMatrix, timeout, verbose)
    if verbose
        fprintf('.');
    end
    errorFlag = 0;
    
    % m-rows * n-cols
    [m, n] = size(scoreMatrix);
    
    % http://web.mit.edu/lpsolve/doc/MATLAB.htm
    f = -ones(m, 1);
    b = -ones(n, 1);
    lb = zeros(size(f));
    e = -ones(n, 1); % less than
    
    % max v = f'*x
    lp = lp_maker(f, -scoreMatrix', b, e, lb);
    mxlpsolve('set_timeout', lp, timeout);
    try
        solveStat = mxlpsolve('solve', lp);
        x = mxlpsolve('get_variables', lp);
    catch
        warning('Error in lp_solve!');
        errorFlag = 1;
        x = ones(m,1) / m;
        mxlpsolve('delete_lp', lp);
        return
    end
    if(solveStat ~= 0)
        warning(['Ended unnormally: solveStat=' num2str(solveStat)]);
        errorFlag = 1;
    end
    % fprintf('After calling the solver %s\n', datestr(now));
    mxlpsolve('delete_lp', lp);
end

%%
function [x, errorFlag] = gurobiWrapper(scoreMatrix, timeout, verbose)
    if verbose
        fprintf('*');
    end
    
    errorFlag = 0;
    
    % m-rows * n-cols
    [m, n] = size(scoreMatrix);
    
    clear model;
    % c'*x; A*x > b
    model.obj = ones(1, m); % c
    model.A = sparse(scoreMatrix'); % A
    model.rhs = ones(n, 1); % b
    model.sense = '>';
    % default lower bounds are 0's
    % default upper bounds are infinites
    % default is minimization
    clear params;
    params.timeLimit = timeout;
    params.outputFlag = 0;
    
    try
        result = gurobi(model, params);
    catch
        warning('Error in gurobi!');
        errorFlag = 1;
        x = ones(m,1) / m;
        return
    end
    
    if(~strcmp(result.status, 'OPTIMAL'))
        warning(['Ended unnormally: status=' result.status]);
        errorFlag = 1;
        if ~isfield(result, 'x')
            x = ones(m,1) / m;
            return
        end
    end
    x = result.x;
end
