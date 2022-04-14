function [minimizerActions, maximizerActions] = buildInitialDCGActions(nDoc, nRel_real)
    % FIXME we are using the smallest and largest actions as initial ones
    % column vectors
    
    if nRel_real == 1 
        nRel_real = 2; % Adjust for binary case
    end
    
%     ascendOrder = 1:nDoc;
%     descendOrder = nDoc:-1:1;
%     
%     lowRel = zeros(nDoc,1);
%     highRel = ones(nDoc,1)*(nRel_real-1);
%     
%     minimizerActions = [lowRel, highRel];
%     maximizerActions = [ascendOrder', descendOrder'];

    order1 = randperm(nDoc);
    order2 = randperm(nDoc);
    while norm(order1-order2)==0
        order2 = randperm(nDoc);
    end
    maximizerActions = [order1', order2'];
    
    rel1 = randi(nRel_real,nDoc,1)-1;
    rel2 = randi(nRel_real,nDoc,1)-1;

    while norm(rel1-rel2)==0
        rel2 = randi(nRel_real,nDoc,1)-1;
    end
    minimizerActions = [rel1, rel2];
   % minimizer is freely choose any action configuration
%     minimizerActions = [randi(k,numOfBits,2)]-1;
%     maximizerActions = [randperm(numOfBits)', randperm(numOfBits)'];
end

