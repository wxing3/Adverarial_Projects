function [minimizerActions, maximizerActions] = buildInitialDCGActions(nDoc, nRel, fullmtx)
    % FIXME we are using the smallest and largest actions as initial ones
    % column vectors
    ascendOrder = 1:nDoc;
    descendOrder = nDoc:-1:1;
    
    if nRel == 1 
        nRel = 2; % Adjust for binary case
    end
    
    lowRel = zeros(nDoc,1);
    highRel = ones(nDoc,1)*(nRel-1);
    
    if fullmtx
        minimizerActions = [];
        for i = 0:2^nDoc-1
            str = dec2bin(i, nDoc);
            minimizerAction = zeros(nDoc,1);
            for j = 1:nDoc
                minimizerAction(j) = str2num(str(j));
            end
            minimizerActions = [minimizerActions, minimizerAction];
        end
        
        maximizerActions = [perms([nDoc:-1:1])]';
    else
        % minimizer is freely choose any action configuration
        minimizerActions = [lowRel, highRel];
        maximizerActions = [ascendOrder', descendOrder'];
        
        % minimizer is freely choose any action configuration
        %     minimizerActions = [randi(nRel,nDoc,2)]-1;
        %     maximizerActions = [randperm(nDoc)', randperm(nDoc)'];
    end
    
end

