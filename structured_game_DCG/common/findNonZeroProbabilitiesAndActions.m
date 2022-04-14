function [nonZeroProbabilities, nonZeroProbActions] = findNonZeroProbabilitiesAndActions(...
        probabilities, actions, valuePrecision)
    probabilities(abs(probabilities) < valuePrecision) = 0;
    nonZeroProbInnerIndices = find(probabilities); % indices in 'probabilities'
    nonZeroProbabilities = probabilities(nonZeroProbInnerIndices);
    nonZeroProbActions = actions(:, nonZeroProbInnerIndices); % actions is column-wise
end
