function [minimizerActions, maximizerActions] = buildInitialRecallActions(numOfBits, k)
    % FIXME we are using the smallest and largest actions as initial ones
    % column vectors
    kBitOnes = ones(k, 1);
    restBitZeros = zeros(numOfBits - k, 1);
    lowerOneBits = [kBitOnes; restBitZeros];
    upperOneBits = [restBitZeros; kBitOnes];
    if(isempty(restBitZeros)) % no zero, but all ones
        % then lowerOneBits and upperOneBits are the same one
        maximizerActions = upperOneBits;
    elseif(isempty(kBitOnes)) % no ones, but all zeros
        maximizerActions = lowerOneBits;
    else
        % each column is a action
        maximizerActions = [lowerOneBits, upperOneBits];
    end
    
    nonBits = zeros(numOfBits, 1);
    allBits = ones(numOfBits, 1);
    % minimizer is freely choose any action configuration
    minimizerActions = [nonBits, allBits];
end

