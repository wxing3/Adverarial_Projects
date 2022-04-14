function initialize(measure)
    fprintf('\nInitializing system to optimize [%s]... ', measure);
    
    clear gfm; % remove the persistent variable W
    
    global findBestMaximizerResponseAction findBestMinimizerResponseAction;
    global buildInitialActions computeScore;
    global buildFullMaximizerActions buildFullMinimizerActions;
    global optimizer buildGameMatrix;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    global lpSolver; % indicate which LP solver to use
    lpSolver = 'lp_solve'; 
    %lpSolver = 'gurobi';
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(strcmpi(measure, 'f1'))
        findBestMaximizerResponseAction = @findBestMaximizerF1ResponseAction;
        findBestMinimizerResponseAction = @findBestMinimizerF1ResponseAction;
        buildInitialActions = @buildInitialF1Actions;
        computeScore = @computeF1;
        optimizer = @optimizerF;
        buildFullMaximizerActions = @buildFullMaximizerFActions;
        buildFullMinimizerActions = @buildFullMinimizerFActions;
        buildGameMatrix = @buildFGameMatrix;
    elseif(strcmpi(measure, 'precision'))
        findBestMaximizerResponseAction = @findBestMaximizerPrecisionResponseAction;
        findBestMinimizerResponseAction = @findBestMinimizerPrecisionResponseAction;
        buildInitialActions = @buildInitialPrecisionActions;
        computeScore = @computePrecision;
        optimizer = @optimizerF;
        buildFullMaximizerActions = @buildFullMaximizerFActions;
        buildFullMinimizerActions = @buildFullMinimizerFActions;
        buildGameMatrix = @buildFGameMatrix;
    elseif(strcmpi(measure, 'recall'))
        findBestMaximizerResponseAction = @findBestMaximizerRecallResponseAction;
        findBestMinimizerResponseAction = @findBestMinimizerRecallResponseAction;
        buildInitialActions = @buildInitialRecallActions;
        computeScore = @computeRecall;
        optimizer = @optimizerF;
        buildFullMaximizerActions = @buildFullMaximizerFActions;
        buildFullMinimizerActions = @buildFullMinimizerFActions;
        buildGameMatrix = @buildFGameMatrix;
    elseif(strcmpi(measure, 'DCG'))
        findBestMaximizerResponseAction = @findBestMaximizerDCGResponseAction;
        findBestMinimizerResponseAction = @findBestMinimizerDCGResponseAction;
        buildInitialActions = @buildInitialDCGActions;
        computeScore = @computeDCG;
        optimizer = @optimizerDCG;
        buildFullMaximizerActions = @buildFullMaximizerDCGActions;
        buildFullMinimizerActions = @buildFullMinimizerDCGActions;
        buildGameMatrix = @buildDCGGameMatrix;
    else
        error('Unsupported measure [%s]. Available measures are [F1, Precision, Recall]', measure);
    end
    
    fprintf('done\n');
end

