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
    global buildGameMatrix;
    
%     buildInitialActions = @buildInitialDCGActions;
%     findBestMaximizerResponseAction = @findBestMaximizerDCGResponseAction;
%     findBestMinimizerResponseAction = @findBestMinimizerDCGResponseAction;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [nDoc, nRel] = size(lagrangeMultipliers);
    
    if nRel == 1
        lagrangeMultipliers = [zeros(nDoc,1) lagrangeMultipliers]; % For binary case, only documents with label 1 has valid features
        nRel_real = 2;
    else
        nRel_real = nRel;
    end
    
    % each column is a action
    
    [minimizerActions, maximizerActions] = buildInitialActions(nDoc,nRel_real);

    % initialize score matrix
    
    timesLimit = nDoc^3;
    flag = 0; % 0 - success; 1 - reach game matrix size limit; 2 - 
    
    scoreMatrix = buildGameMatrix(minimizerActions, maximizerActions, lagrangeMultipliers, IDCG);
    
    done = false;
    calledTimes = 0;
    while ~done
        % find and add minimizer's response %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [maximizerProbabilities, maximizerValue, errorFlag1] = findMaximizerProbabilities(scoreMatrix, calledTimes, verbose);
        calledTimes = calledTimes + 1;
        
        if errorFlag1
            flag = 2;
            warning('The game value cannot be solved.');
            break;
        end
        
        [nonZeroMaximizerProbabilities, nonZeroProbMaximizerActions] = ...
            findNonZeroProbabilitiesAndActions(maximizerProbabilities, maximizerActions, valuePrecision);
        
        nonZeroMaximizerProbabilities = nonZeroMaximizerProbabilities / sum(nonZeroMaximizerProbabilities);
        
        [bestMinimizerResponseAction, bestMinimizerResponse] = findBestMinimizerResponseAction(...
            nonZeroMaximizerProbabilities, nonZeroProbMaximizerActions, lagrangeMultipliers, IDCG);
%         [bestMinimizerResponseActions_e, bestMinimizerResponse_e] = findBestMinimizerResponseActionExhaust(...
%             nonZeroMaximizerProbabilities, nonZeroProbMaximizerActions, lagrangeMultipliers, IDCG, valuePrecision);
%         if abs(bestMinimizerResponse-bestMinimizerResponse_e) > valuePrecision
%             warning('Best minimizer finder get different game value %0.6f as optimal solution %0.6f',bestMinimizerResponse,bestMinimizerResponse_e);
%         end
%         if ~ismember(bestMinimizerResponseAction',bestMinimizerResponseActions_e','rows')
%             warning('Best minimizer finder fails in finding the optimal action');
%             fprintf('Best response got is:');
%             disp(bestMinimizerResponseAction');
%             fprintf('Optimal best responses are:');
%             disp(bestMinimizerResponseActions_e');
%         end
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
        
        if errorFlag2
            flag = 2;
            warning('The game value cannot be solved.');
            break;
        end
        
        [nonZeroMinimizerProbabilities, nonZeroProbMinimizerActions] = ...
            findNonZeroProbabilitiesAndActions(minimizerProbabilities, minimizerActions, valuePrecision);
        
        nonZeroMinimizerProbabilities = nonZeroMinimizerProbabilities / sum(nonZeroMinimizerProbabilities);
        
        [bestMaximizerResponseAction, bestMaximizerResponse] = findBestMaximizerResponseAction(...
            nonZeroMinimizerProbabilities, nonZeroProbMinimizerActions, lagrangeMultipliers, IDCG);
%         [bestMaximizerResponseActions_e, bestMaximizerResponse_e] = findBestMaximizerResponseActionExhaust(...
%             nonZeroMinimizerProbabilities, nonZeroProbMinimizerActions, lagrangeMultipliers, IDCG, valuePrecision);
%         if abs(bestMaximizerResponse-bestMaximizerResponse_e) > valuePrecision
%             warning('Best maximizer finder get different game value %0.6f as optimal solution %0.6f',bestMaximizerResponse,bestMaximizerResponse_e);
%         end
%         if ~ismember(bestMaximizerResponseAction',bestMaximizerResponseActions_e','rows')
%             warningStr = strcat(warningStr,'Best response got is:');
%             warningStr = strcat(warningStr,mat2str(bestMaximizerResponseAction'));
%             warningStr = strcat(warningStr,'\nOptimal best responses are:\n');
%             [~,nActions] = size(bestMaximizerResponseActions_e);
%             for i = 1:nActions
%                 warningStr = strcat(warningStr,mat2str((bestMaximizerResponseActions_e(:,i))'));
%                 warningStr = strcat(warningStr,'\n');
%             end
%             warning('Best maximizer finder fails in finding the optimal action.%s',strcat('\n',warningStr));
%         end
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