function [minimizerActions, maximizerActions] = buildInitialF1Actions(numOfBits, ~)
    % column vectors
    nonBits = zeros(numOfBits, 1);
    allBits = ones(numOfBits, 1);
    % each column is a action, e.g.
    % [0 1;
    %  0 1;
    %  0 1]
    minimizerActions = [nonBits, allBits];
    maximizerActions = [nonBits, allBits];
end
