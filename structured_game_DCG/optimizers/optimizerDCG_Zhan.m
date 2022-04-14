function [scoreMatrix, ...
        minimizerActions, maximizerActions, ...
        minimizerProbabilities, maximizerProbabilities, ...
        minimizerValue, maximizerValue, flag] = optimizerDCG(lagrangeMultipliers, IDCG, valuePrecision, verbose)
    % This version stops when there's no new action anymore.
%     if (~isvector(lagrangeMultipliers))
%         error('Input "lagrangeMultipliers" must be a vector.');
%     end
    if nargin < 4
        verbose = false;
    end

    global findBestMaximizerResponseAction findBestMinimizerResponseAction;
    global buildInitialActions;
    
%     buildInitialActions = @buildInitialDCGActions;
%     findBestMaximizerResponseAction = @findBestMaximizerDCGResponseAction;
%     findBestMinimizerResponseAction = @findBestMinimizerDCGResponseAction;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [nDoc, nRel] = size(lagrangeMultipliers);
    
    if nRel == 1
        lagrangeMultipliers = [zeros(nDoc,1) lagrangeMultipliers]; % For binary case, only documents with label 1 has valid features
    end
    
    % each column is a action
    
    [minimizerActions, maximizerActions] = buildInitialActions(nDoc,nRel, false);

    % initialize score matrix
    
    timesLimit = nDoc^2;
    flag = 0;
    
    scoreMatrix = buildScoreMatrix(minimizerActions, maximizerActions, lagrangeMultipliers, IDCG);
    
    done = false;
    calledTimes = 0;
    
    if 0
        [minimizerActions, maximizerActions] = buildInitialActions(nDoc,nRel, true);
        scoreMatrix = buildScoreMatrix(minimizerActions, maximizerActions, lagrangeMultipliers, IDCG);
        [maximizerProbabilities, maximizerValue, errorFlag1] = findMaximizerProbabilities(scoreMatrix, calledTimes, verbose);
        [minimizerProbabilities, minimizerValue, errorFlag2] = findMinimizerProbabilities(scoreMatrix, calledTimes, verbose);
        done = true;
    end 
    
    while ~done
        % find and add minimizer's response %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [maximizerProbabilities, maximizerValue, errorFlag1] = findMaximizerProbabilities(scoreMatrix, calledTimes, verbose);
        calledTimes = calledTimes + 1;
        
        [nonZeroMaximizerProbabilities, nonZeroProbMaximizerActions] = ...
            findNonZeroProbabilitiesAndActions(maximizerProbabilities, maximizerActions, valuePrecision);
        
        nonZeroMaximizerProbabilities = nonZeroMaximizerProbabilities / sum(nonZeroMaximizerProbabilities);
        
        [bestMinimizerResponseAction, bestMinimizerResponse] = findBestMinimizerResponseAction(...
            nonZeroMaximizerProbabilities, nonZeroProbMaximizerActions, lagrangeMultipliers, IDCG);
        
%         fprintf('bestMinimizerResponseAction=');
%         fprintf('%.6f\t',bestMinimizerResponseAction);
%         fprintf('\nmaximizerValue=%.6f\tbestMinimizerResponse=%.6f\n', maximizerValue, bestMinimizerResponse);
        
        reachedBestMinimizerValue = abs(maximizerValue - bestMinimizerResponse) <= valuePrecision;
        if (~reachedBestMinimizerValue)
            minimizerAlreadyExists = doesActionAlreadyExist(minimizerActions, bestMinimizerResponseAction);
            if (~minimizerAlreadyExists)
                % append at the end
                minimizerActions = [minimizerActions, bestMinimizerResponseAction]; %#ok<*AGROW>
                % bestMinimizerResponseScores is one column
                bestMinimizerResponseScores = computeMinimizerScores(...
                    bestMinimizerResponseAction, maximizerActions, lagrangeMultipliers,IDCG);
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
        [minimizerProbabilities, minimizerValue, errorFlag2] = findMinimizerProbabilities(scoreMatrix, calledTimes, verbose);
        calledTimes = calledTimes + 1;
        
        if errorFlag1 || errorFlag2
            flag = 2;
            warning('The game value cannot be solved.');
            break;
        end
        
        [nonZeroMinimizerProbabilities, nonZeroProbMinimizerActions] = ...
            findNonZeroProbabilitiesAndActions(minimizerProbabilities, minimizerActions, valuePrecision);
        
        nonZeroMinimizerProbabilities = nonZeroMinimizerProbabilities / sum(nonZeroMinimizerProbabilities);
        
        [bestMaximizerResponseAction, bestMaximizerResponse] = findBestMaximizerResponseAction(...
            nonZeroMinimizerProbabilities, nonZeroProbMinimizerActions, lagrangeMultipliers, IDCG);

%         fprintf('bestMaximizerResponseAction=');
%         fprintf('%.6f\t',bestMaximizerResponseAction);
%         fprintf('\nminimizerValue=%.6f\tbestMaximizerResponse=%.6f\n', minimizerValue, bestMaximizerResponse);
  
        reachBestMaximizerValue = abs(minimizerValue - bestMaximizerResponse) <= valuePrecision ...
            && abs(minimizerValue - maximizerValue) <= valuePrecision;
        if(~reachBestMaximizerValue)
            maximizerAlreadyExists = doesActionAlreadyExist(maximizerActions, bestMaximizerResponseAction);
            if (~maximizerAlreadyExists)
                % append at the end
                maximizerActions = [maximizerActions, bestMaximizerResponseAction];
                % bestMaximizerResponseScores is one row
                bestMaximizerResponseScores = computeMaximizerScores(...
                    bestMaximizerResponseAction, minimizerActions, lagrangeMultipliers,IDCG);
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
        
        if calledTimes > timesLimit
            flag = 1;
            warning('The game value failed to converge in %d expansion steps', timesLimit);
            break;
        end
    end
    if verbose
        fprintf('\n');
    end
        
%   View size of final game matrix  
%   fprintf('CandidateMinizerActions=%d\tCandidateMaxizerActions=%d\n', length(minimizerProbabilities), length(maximizerProbabilities));

end

%%
function minimizerScoreColumn = computeMinimizerScores(...
        minimizerAction, maximizerActions, lagrangeMultipliers,IDCG)
    minimizerScoreColumn = buildScoreMatrix(minimizerAction, maximizerActions, lagrangeMultipliers, IDCG);
    assert(iscolumn(minimizerScoreColumn), '"scoreMatrix" here must be a column.');
end

%%
function maximizerScoreRow = computeMaximizerScores(...
        maximizerAction, minimizerActions, lagrangeMultipliers,IDCG)
    maximizerScoreRow = buildScoreMatrix(minimizerActions, maximizerAction, lagrangeMultipliers, IDCG);
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
function scoreMatrix = buildScoreMatrix(minimizerActions, maximizerActions, lagrangeMultipliers, IDCG)
    global computeScore;
    
    % number of columns is the count of actions.
    numOfMinimizerActions = size(minimizerActions, 2);
    numOfMaximizerActions = size(maximizerActions, 2);
    
    scoreMatrix = zeros(numOfMaximizerActions, numOfMinimizerActions);
    
    for minimizerIndex = 1:numOfMinimizerActions % column index in scoreMatrix
        minimizerAction = minimizerActions(:, minimizerIndex);
        
        % The same minimizer action have the same lagrangeMultiplierSum
        lagrangeMultiplierSum = 0;
        for i = 1:size(lagrangeMultipliers,1)
            lagrangeMultiplierSum = lagrangeMultiplierSum + lagrangeMultipliers(i,minimizerAction(i)+1);
        end
        
        for maximizerIndex = 1:numOfMaximizerActions % row index in scoreMatrix
            maximizerAction = maximizerActions(:, maximizerIndex);
          
            score = computeScore(minimizerAction, maximizerAction, IDCG);
            
            % 'lagrangeMultipliers' are subtracted from each minimizer column.
            scoreMatrix(maximizerIndex, minimizerIndex) = score - lagrangeMultiplierSum;
        end
    end
end
