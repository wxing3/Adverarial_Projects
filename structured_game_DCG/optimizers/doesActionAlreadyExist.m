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