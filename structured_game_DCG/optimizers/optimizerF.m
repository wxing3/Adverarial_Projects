function [scoreMatrix, ...
        minimizerActions, maximizerActions, ...
        minimizerProbabilities, maximizerProbabilities, ...
        minimizerValue, maximizerValue, flag] = optimizerF(lagrangeMultipliers, k, valuePrecision, verbose)
    if nargin  < 4
        verbose = false;
    end
    
    % This version stops when there's no new action anymore.
    if (~isvector(lagrangeMultipliers))
        error('Input "lagrangeMultipliers" must be a vector.');
    end
    
    global findBestMaximizerResponseAction findBestMinimizerResponseAction;
    global buildInitialActions;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    timesLimit = 100;
    flag = 0;

    numOfBits = length(lagrangeMultipliers);
    % each column is a action
    [minimizerActions, maximizerActions] = buildInitialActions(numOfBits, k);
    % initialize score matrix
    scoreMatrix = buildScoreMatrix(minimizerActions, maximizerActions, lagrangeMultipliers);
    
    done = false;
    calledTimes = 0;
    while ~done && calledTimes <= timesLimit 
        % find and add minimizer's response %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [maximizerProbabilities, maximizerValue] = findMaximizerProbabilities(scoreMatrix, calledTimes);
        calledTimes = calledTimes + 1;
        
        [nonZeroMaximizerProbabilities, nonZeroProbMaximizerActions] = ...
            findNonZeroProbabilitiesAndActions(maximizerProbabilities, maximizerActions, valuePrecision);
        
        [bestMinimizerResponseAction, bestMinimizerResponse] = findBestMinimizerResponseAction(...
            nonZeroMaximizerProbabilities, nonZeroProbMaximizerActions, lagrangeMultipliers, k);
        % fprintf('maximizerValue=%.6f\tbestMinimizerResponse=%.6f\n', maximizerValue, bestMinimizerResponse);
        
        reachedBestMinimizerValue = abs(maximizerValue - bestMinimizerResponse) <= valuePrecision;
        if (~reachedBestMinimizerValue)
            minimizerAlreadyExists = doesActionAlreadyExist(minimizerActions, bestMinimizerResponseAction);
            if (~minimizerAlreadyExists)
                % append at the end
                minimizerActions = [minimizerActions, bestMinimizerResponseAction]; %#ok<*AGROW>
                % bestMinimizerResponseScores is one column
                bestMinimizerResponseScores = computeMinimizerScores(...
                    bestMinimizerResponseAction, maximizerActions, lagrangeMultipliers);
                % append column at the end of scoreMatrix
                scoreMatrix = [scoreMatrix, bestMinimizerResponseScores];
            else
                reachedBestMinimizerValue = true; % no further minimizer action was inserted
                if verbose
                    fprintf('<');
                end
            end
        else
            if verbose
                fprintf('v');
            end
        end
        
        % find and add maximizer's response %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [minimizerProbabilities, minimizerValue] = findMinimizerProbabilities(scoreMatrix, calledTimes);
        calledTimes = calledTimes + 1;
        
        [nonZeroMinimizerProbabilities, nonZeroProbMinimizerActions] = ...
            findNonZeroProbabilitiesAndActions(minimizerProbabilities, minimizerActions, valuePrecision);
        
        [bestMaximizerResponseAction, bestMaximizerResponse] = findBestMaximizerResponseAction(...
            nonZeroMinimizerProbabilities, nonZeroProbMinimizerActions, lagrangeMultipliers, k);
        % fprintf('minimizerValue=%.6f\tbestMaximizerResponse=%.6f\n', minimizerValue, bestMaximizerResponse);
        
        reachBestMaximizerValue = abs(minimizerValue - bestMaximizerResponse) <= valuePrecision ...
            && abs(minimizerValue - maximizerValue) <= valuePrecision;
        if(~reachBestMaximizerValue)
            maximizerAlreadyExists = doesActionAlreadyExist(maximizerActions, bestMaximizerResponseAction);
            if (~maximizerAlreadyExists)
                % append at the end
                maximizerActions = [maximizerActions, bestMaximizerResponseAction];
                % bestMaximizerResponseScores is one row
                bestMaximizerResponseScores = computeMaximizerScores(...
                    bestMaximizerResponseAction, minimizerActions, lagrangeMultipliers);
                % append row at the end of scoreMatrix
                scoreMatrix = [scoreMatrix; bestMaximizerResponseScores];
            else
                reachBestMaximizerValue = true; % no further maximizer action was inserted
                if verbose
                    fprintf('>');
                end
            end
        else
            if verbose
                fprintf('^');
            end
        end
        
        done = (reachedBestMinimizerValue && reachBestMaximizerValue);
    end

    fprintf('\n');    

    if calledTimes > timesLimit
        flag = 1;
        warning('The game value failed to converge in %d expansion steps', timesLimit);
    end
end

%%
function minimizerScoreColumn = computeMinimizerScores(...
        minimizerAction, maximizerActions, lagrangeMultipliers)
    minimizerScoreColumn = buildScoreMatrix(minimizerAction, maximizerActions, lagrangeMultipliers);
    assert(iscolumn(minimizerScoreColumn), '"scoreMatrix" here must be a column.');
end

%%
function maximizerScoreRow = computeMaximizerScores(...
        maximizerAction, minimizerActions, lagrangeMultipliers)
    maximizerScoreRow = buildScoreMatrix(minimizerActions, maximizerAction, lagrangeMultipliers);
    assert(isrow(maximizerScoreRow), '"scoreMatrix" here must be a row.');
end

%%
function exists = doesActionAlreadyExist(originalSortedActions, actionToInsert)
    protentialExistingLogicalIndices = ismember(originalSortedActions', actionToInsert', 'rows');
    sumOfProtentialExistingLogicalIndices = sum(protentialExistingLogicalIndices);
    if(sumOfProtentialExistingLogicalIndices > 1)
        display(originalSortedActions);
        display(actionToInsert);
        error('Should not alrady had more than one identical action.');
    end
    
    if(sumOfProtentialExistingLogicalIndices == 1) % already exist
        exists = true;
    else
        exists = false;
    end
end

%%
function scoreMatrix = buildScoreMatrix(minimizerActions, maximizerActions, lagrangeMultipliers)
    global computeScore;
    
    % number of columns is the count of actions.
    numOfMinimizerActions = size(minimizerActions, 2);
    numOfMaximizerActions = size(maximizerActions, 2);
    
    scoreMatrix = zeros(numOfMaximizerActions, numOfMinimizerActions);
    
    for minimizerIndex = 1:numOfMinimizerActions % column index in scoreMatrix
        minimizerAction = minimizerActions(:, minimizerIndex);
        lagrangeMultiplierSum = sum(lagrangeMultipliers(minimizerAction == 1));
        
        for maximizerIndex = 1:numOfMaximizerActions % row index in scoreMatrix
            maximizerAction = maximizerActions(:, maximizerIndex);
            score = computeScore(minimizerAction, maximizerAction);
            % 'lagrangeMultipliers' are subtracted from each minimizer column.
            scoreMatrix(maximizerIndex, minimizerIndex) = score - lagrangeMultiplierSum;
        end
    end
end
